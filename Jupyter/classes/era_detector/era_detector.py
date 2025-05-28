import pandas as pd
import ruptures as rpt
import sqlalchemy
import numpy as np 
import scipy.stats 
import torch 

# -----------------------------------------------------------------
# pomegranate 0.14.x  <-->  pomegranate >=1.0  compatibility layer
# (Based on provided web analysis)
# -----------------------------------------------------------------
try:                                # Newer pomegranate (>=1.0)
    from pomegranate.hmm import DenseHMM
    from pomegranate.distributions import Normal
    HiddenMarkovModel = DenseHMM     # <-- alias so the rest of the code stays unchanged
    print("pomegranate >=1.0 detected (using DenseHMM + Normal)")
except ImportError:                  # Older pomegranate (<1.0)
    from pomegranate.hmm import HiddenMarkovModel
    from pomegranate.distributions import NormalDistribution as Normal # Alias old NormalDistribution to Normal for consistency
    print("pomegranate 0.14.x or older detected (using HiddenMarkovModel + NormalDistribution as Normal)")
# -----------------------------------------------------------------

import functools 

# ---- SciPy alias fix: make bayesian_changepoint_detection happy ----
import types, scipy            # must run *before* the BOCPD import
if not hasattr(scipy, "misc"):
    scipy.misc = types.ModuleType("misc")
if not hasattr(scipy.misc, "comb"):          # alias vanished in SciPy 1.3
    from scipy.special import comb           # new canonical location
    scipy.misc.comb = comb
    print("Shimmed scipy.misc.comb for bayesian_changepoint_detection")
# --------------------------------------------------------------------
import bayesian_changepoint_detection.offline_changepoint_detection as bocd_offline
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE} for HMM (if applicable)")

def fetch_data_from_timescaledb(db_params: dict, query: str) -> pd.DataFrame:
    """
    Fetches data from a TimescaleDB/PostgreSQL database using SQLAlchemy Core.

    Args:
        db_params: Dictionary with database connection parameters
                     (e.g., {'username': 'user', 'password': 'pass', 'host': 'localhost', 
                             'port': '5432', 'database': 'dbname'})
        query: The SQL query string to execute.

    Returns:
        A pandas DataFrame containing the query results.
        The DataFrame should have a DatetimeIndex.
    """
    try:
        # Construct the database URL for SQLAlchemy
        # postgresql+psycopg2://user:password@host:port/database
        db_url = sqlalchemy.engine.URL.create(
            drivername="postgresql+psycopg2",
            username=db_params.get('user'),      # Changed from 'user' to 'username' to match common SQLAlchemy usage
            password=db_params.get('password'),
            host=db_params.get('host'),
            port=db_params.get('port'),
            database=db_params.get('dbname')    # Changed from 'dbname' to 'database' for clarity with URL
        )
        
        engine = sqlalchemy.create_engine(db_url)
        
        print("Successfully created SQLAlchemy engine.")
        
        # Using a connection context manager
        with engine.connect() as connection:
            df = pd.read_sql_query(sql=sqlalchemy.text(query), con=connection, index_col=0) # Assumes first col is for index
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        print(f"Data fetched successfully using SQLAlchemy. DataFrame shape: {df.shape}")
        return df
    
    except sqlalchemy.exc.SQLAlchemyError as e: # Catch generic SQLAlchemy errors
        print(f"SQLAlchemy Database error: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"An error occurred while fetching data with SQLAlchemy: {e}")
        return pd.DataFrame()
    finally:
        # SQLAlchemy engine handles connection pooling and disposal, so explicit conn.close() is less common here
        # if using the context manager for connections.
        if 'engine' in locals() and engine is not None:
            engine.dispose()
            print("SQLAlchemy engine disposed.")

def _convert_changepoints_to_interval_index(series_index: pd.DatetimeIndex, changepoint_indices: list[int]) -> pd.IntervalIndex:
    """Helper function to convert a list of changepoint indices to a pandas IntervalIndex."""
    if not changepoint_indices or series_index.empty:
        return pd.IntervalIndex([])

    break_dates = [series_index[0].normalize()] # Start date
    for idx in sorted(list(set(changepoint_indices))): # Ensure unique, sorted indices
        if 0 <= idx < len(series_index):
            break_dates.append(series_index[idx].normalize())
    break_dates.append((series_index[-1].normalize() + pd.Timedelta(days=1))) # End date
    
    final_break_points = sorted(list(set(break_dates))) # Ensure unique and sorted boundary points
    
    if len(final_break_points) > 1:
        return pd.IntervalIndex.from_breaks(final_break_points, closed='left')
    else:
        return pd.IntervalIndex([])

def detect_eras_ruptures(counts_series: pd.Series, pen_value: float = 10.0) -> pd.IntervalIndex:
    """
    Detects eras using Ruptures PELT.

    Args:
        counts_series: pandas Series of daily counts (index=DatetimeIndex, values=counts).
        pen_value: Penalty for the PELT algorithm. Higher values -> fewer changepoints.

    Returns:
        A pandas.IntervalIndex representing the detected eras.
    """
    if counts_series.empty:
        return pd.IntervalIndex([])

    counts_values = counts_series.values.reshape(-1, 1)
    model = rpt.Pelt(model="poisson", min_size=7).fit(counts_values)
    # break_idx from ruptures are the first index of the new segment (exclusive end of previous)
    # The last element is n_samples.
    break_idx_indices = model.predict(pen=pen_value)[:-1] # Exclude the last one (n_samples)
    return _convert_changepoints_to_interval_index(counts_series.index, break_idx_indices)

def detect_eras_hmm(counts_series: pd.Series, n_states: int = 3, random_state_seed: int = 42) -> tuple[pd.IntervalIndex | None, HiddenMarkovModel | None]:
    """
    Detects eras using a Hidden Markov Model (HMM) from pomegranate.
    Uses DenseHMM for pomegranate >= 1.0.0 and moves model to DEVICE.
    The compatibility shim handles HiddenMarkovModel and Normal aliasing.

    Args:
        counts_series: pandas Series of daily counts.
        n_states: Number of hidden states for the HMM.
        random_state_seed: Seed for reproducibility (torch.manual_seed might be used globally).

    Returns:
        A tuple containing: 
            - pandas.IntervalIndex representing the detected eras (or None if error).
            - The trained HMM model (or None if error).
    """
    if counts_series.empty or len(counts_series) < n_states * 2: # Need enough data
        print("HMM: Data series is empty or too short for the number of states.")
        return None, None

    try:
        # Ensure data is float32 for torch, reshape. The patch uses float32.
        data_for_hmm = counts_series.values.astype(np.float32).reshape(-1, 1)
        
        print(f"Fitting HMM with {n_states} states on device: {DEVICE}...")
        # For reproducibility with torch backend, seed torch if specific initial weights are critical
        # torch.manual_seed(random_state_seed) 

        dists = [Normal() for _ in range(n_states)] # Parameters will be learned during fit
        
        # HiddenMarkovModel is now an alias from the shim (points to DenseHMM for v1.x)
        # DenseHMM() constructor does not take random_state.
        model = HiddenMarkovModel().to(DEVICE) 
        model.add_distributions(dists) 

        model.fit(data_for_hmm, batch_size=8192, n_jobs=1, verbose=False, inertia=0.1)
        print("HMM fitting complete.")

        predicted_states = model.predict(data_for_hmm, batch_size=16384)
        
        changepoint_indices = np.where(np.diff(predicted_states) != 0)[0] + 1
        eras = _convert_changepoints_to_interval_index(counts_series.index, changepoint_indices.tolist())
        return eras, model
    except Exception as e:
        print(f"Error during HMM processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def detect_eras_with_bocpd(counts_series: pd.Series, hazard_lambda: int = 200) -> tuple[pd.IntervalIndex | None, np.ndarray | None]:
    """
    Detects changepoints using Bayesian Offline Changepoint Detection (BOCPD).
    Uses a Gaussian observation model as a practical placeholder for count data.

    Args:
        counts_series: pandas Series of daily counts.
        hazard_lambda: Expected run length (hazard prior), e.g., 200 days.

    Returns:
        A tuple containing:
            - pandas.IntervalIndex representing the detected eras (or None if error).
            - The Pcp matrix (posterior probabilities of changepoints), for potential further analysis.
    """
    if counts_series.empty or len(counts_series) < hazard_lambda / 10: # Ensure some minimal data length
        print("BOCPD: Data series is empty or too short.")
        return None, None

    print(f"\n--- Running Bayesian Offline Changepoint Detection (Gaussian Approx.) ---")
    print(f"Hazard Lambda: {hazard_lambda}")
    print("NOTE: Using Gaussian observation model as an approximation for count data. Results may be suboptimal for low counts.")

    data = counts_series.values.astype(float) # Ensure float for Gaussian model

    # Parameters for the Gaussian observation model (prior beliefs)
    # These are illustrative. Proper priors would be data-informed or weakly informative.
    # For bocd_offline.gaussian_obs_log_likelihood, it implicitly uses a Normal-Gamma prior.
    # The function itself does not take these mu0, kappa0 etc. directly. It's assumed by the model.
    # We use a simplified approach here if we were to write our own likelihood, but the library's internal model is used.
    # mu0 = np.mean(data) if len(data) > 0 else 0
    # kappa0 = 0.01
    # alpha0 = 0.01 # Inverse Gamma shape for variance
    # beta0 = 0.01  # Inverse Gamma scale for variance

    try:
        # The library's gaussian_obs_log_likelihood is designed to work with the offline_changepoint_detection framework.
        # It implicitly assumes a Normal-Gamma conjugate prior structure for the mean and variance.
        # No explicit prior parameters are passed to gaussian_obs_log_likelihood here.
        Q, P, Pcp = bocd_offline.offline_changepoint_detection(
            data,
            functools.partial(bocd_offline.constant_hazard, hazard_lambda),
            bocd_offline.gaussian_obs_log_likelihood 
            # For custom priors with Gaussian, one would typically need to define a custom likelihood function
            # that incorporates those priors into the marginal likelihood calculation for a segment.
        )
        print(f"BOCPD Pcp matrix shape: {Pcp.shape}")

        # Extract changepoints from Pcp matrix
        # A common way: find where the run length with max probability resets to 0 (or near 0).
        # The Pcp matrix stores P(runlength_t | data_1:t). Max prob often at R_t=0 after a changepoint.
        # An alternative: changepoints = np.where(np.diff(np.argmax(Pcp, axis=0)) < 0)[0]
        # The argmax gives the most likely run length at each time t.
        # A decrease in this most likely run length often signals a changepoint.
        # Let's try finding peaks in Pcp[0,:] which indicates high prob of run length 0 (new segment)
        
        # Heuristic: Changepoint if prob of run length 0 is high (e.g. > threshold)
        # Or where argmax of run length probability significantly drops.
        # For simplicity and robustness, let's use the diff(argmax) approach here. indices are t-1 where change happens at t.
        # We add +1 to align with the definition of start of new segment.
        changepoint_indices = np.where(np.diff(np.argmax(Pcp, axis=0)) < - (len(Pcp)//10) )[0] + 1 # Heuristic threshold for drop
        # A simpler way often cited for this library (though not always perfect):
        # changepoint_indices = np.where(Pcp[0,1:] > 0.5)[0] # if P(runlength=0 at t) > 0.5, then t-1 is a changepoint
        # The Pcp[0,t] is P(r_t=0 | x_{1:t}). If this is high, a change just occurred.

        # Using a slightly more robust method: find where the MAP run length significantly drops.
        map_run_lengths = np.argmax(Pcp, axis=0)
        diff_map_run_lengths = np.diff(map_run_lengths)
        # A changepoint is suspected where the most likely run length drops significantly (becomes small again)
        # The threshold here (-10) is arbitrary and needs tuning or a more principled approach.
        # Or, simply where it's not just incrementing by 1.
        changepoint_indices = np.where(diff_map_run_lengths < 0)[0] + 1 # Add 1 to align index with data point *after* change.
        
        print(f"BOCPD detected potential changepoint indices (raw): {changepoint_indices}")
        eras = _convert_changepoints_to_interval_index(counts_series.index, changepoint_indices.tolist())
        return eras, Pcp

    except Exception as e:
        print(f"Error during BOCPD processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_era_report(eras_ruptures: pd.IntervalIndex = None, 
                        eras_hmm: pd.IntervalIndex = None, 
                        eras_bocpd: pd.IntervalIndex = None, 
                        output_file_path: str = "era_report.txt") -> None:
    """
    Generates a text report from the detected eras from multiple methods.
    """
    report_lines = []
    report_lines.append("Data Availability Era Report")
    report_lines.append("=" * 30)

    def format_eras_for_report(method_name: str, eras_index: pd.IntervalIndex):
        if eras_index is None or eras_index.empty:
            report_lines.append(f"\n--- {method_name} ---")
            report_lines.append("No eras were detected or data insufficient.")
            return
        
        report_lines.append(f"\n--- {method_name} (Detected: {len(eras_index)} eras) ---")
        for i, era in enumerate(eras_index):
            start_date_str = era.left.strftime('%Y-%m-%d')
            end_date_str = (era.right - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            duration_days = (era.right - era.left).days
            report_lines.append(f"Era {i+1}: {start_date_str} to {end_date_str} ({duration_days} days)")

    if eras_ruptures is not None:
        format_eras_for_report("Ruptures (PELT)", eras_ruptures)
    if eras_hmm is not None:
        format_eras_for_report("Hidden Markov Model (HMM)", eras_hmm)
    if eras_bocpd is not None:
        format_eras_for_report("Bayesian Changepoint Detection (BOCPD)", eras_bocpd)

    if not any([eras_ruptures is not None and not eras_ruptures.empty, 
                eras_hmm is not None and not eras_hmm.empty, 
                eras_bocpd is not None and not eras_bocpd.empty]):
        report_lines.append("\nNo eras detected by any method.")

    try:
        with open(output_file_path, 'w') as f:
            for line in report_lines:
                f.write(line + "\n")
        print(f"Era report successfully saved to: {output_file_path}")
    except IOError as e:
        print(f"Error saving era report to {output_file_path}: {e}")

if __name__ == '__main__':
    # --- 1. Configure Database Connection and Query ---
    db_connection_params = {
        'dbname': 'postgres',   # Will be used as 'database' for SQLAlchemy URL
        'user': 'postgres', # Will be used as 'username'
        'password': 'postgres',
        'host': 'host.docker.internal', # For Docker connection to host machine
        'port': '5432'
    }

    # List of relevant data columns from your input
    # Excluding metadata-like fields for era detection based on sensor data presence
    data_columns = [
        "lamp_group",
        "air_temp_c",
        "air_temp_middle_c",
        "outside_temp_c",
        "relative_humidity_percent",
        "humidity_deficit_g_m3",
        "radiation_w_m2",
        "light_intensity_lux",
        "light_intensity_umol",
        "outside_light_w_m2",
        "co2_measured_ppm",
        "co2_required_ppm",
        "co2_dosing_status",
        "co2_status",
        "rain_status",
        "vent_pos_1_percent",
        "vent_pos_2_percent",
        "vent_lee_afd3_percent",
        "vent_wind_afd3_percent",
        "vent_lee_afd4_percent",
        "vent_wind_afd4_percent",
        "curtain_1_percent",
        "curtain_2_percent",
        "curtain_3_percent",
        "curtain_4_percent",
        "window_1_percent",
        "window_2_percent",
        "lamp_grp1_no3_status",
        "lamp_grp2_no3_status",
        "lamp_grp3_no3_status",
        "lamp_grp4_no3_status",
        "lamp_grp1_no4_status",
        "lamp_grp2_no4_status",
        "measured_status_bool",
        "heating_setpoint_c",
        "pipe_temp_1_c",
        "pipe_temp_2_c",
        "flow_temp_1_c",
        "flow_temp_2_c",
        "temperature_forecast_c",
        "sun_radiation_forecast_w_m2",
        "temperature_actual_c",
        "sun_radiation_actual_w_m2",
        "vpd_hpa",
        "humidity_deficit_afd3_g_m3",
        "relative_humidity_afd3_percent",
        "humidity_deficit_afd4_g_m3",
        "relative_humidity_afd4_percent",
        "dli_sum"
    ]

    # Ensure column names are quoted in case they contain special characters or are keywords
    quoted_data_columns = [f'"{col}"' for col in data_columns]
    
    # The timestamp column must be first in the SELECT statement for index_col=0 in pd.read_sql_query
    # Assuming 'time' is your timestamp column and needs to be quoted as well.
    timestamp_column = '"time"' 
    table_name = 'your_greenhouse_data_table'  # !!! REPLACE THIS with your actual table name !!!

    # Construct the SQL query
    # It's good practice to explicitly list columns rather than SELECT *
    sql_query = f"SELECT {timestamp_column}, {', '.join(quoted_data_columns)} FROM {table_name} ORDER BY {timestamp_column} ASC;"
    
    print(f"Generated SQL Query: {sql_query}") # For verification

    print("Attempting to fetch data from the database...")
    main_df = fetch_data_from_timescaledb(db_connection_params, sql_query)

    counts_series = pd.Series(dtype=float) # Initialize empty series
    if not main_df.empty:
        if not isinstance(main_df.index, pd.DatetimeIndex):
            try:
                main_df.index = pd.to_datetime(main_df.index)
                print("DataFrame index converted to DatetimeIndex.")
            except Exception as e:
                print(f"Error: Could not convert DataFrame index to DatetimeIndex: {e}")
                main_df = pd.DataFrame() # Invalidate df
        
        if not main_df.empty:
            print("\nCalculating daily counts of non-null columns...")
            full_date_range = pd.date_range(start=main_df.index.min().normalize(), 
                                            end=main_df.index.max().normalize(), 
                                            freq='D')
            daily_sum_of_non_na_counts = main_df.notna().sum(axis=1).resample('D').sum()
            counts_series = daily_sum_of_non_na_counts.reindex(full_date_range, fill_value=0)
            print(f"Counts series calculated. Length: {len(counts_series)}, From {counts_series.index.min()} to {counts_series.index.max()}")
    else:
        print("Failed to fetch data from the database, or the query returned no data. Cannot proceed.")

    # --- Run Era Detection Methods ---
    detected_eras_ruptures = None
    detected_eras_hmm = None      # Placeholder for HMM results
    hmm_model_trained = None # To store the HMM model for analysis
    detected_eras_bocpd = None # For IntervalIndex
    bocpd_pcp_matrix = None    # For storing the Pcp matrix

    if not counts_series.empty:
        print("\n--- Running Ruptures (PELT) Detection ---")
        ruptures_pen_value = 10.0 # Example, can be tuned
        print(f"Using Ruptures PELT with penalty: {ruptures_pen_value}")
        detected_eras_ruptures = detect_eras_ruptures(counts_series.copy(), pen_value=ruptures_pen_value)
        if detected_eras_ruptures is not None and not detected_eras_ruptures.empty:
            print(f"Ruptures detected {len(detected_eras_ruptures)} eras.")
        else:
            print("Ruptures detected no eras or data was insufficient.")

        print("\n--- Running HMM Detection ---")
        hmm_n_states = 3 # Example, can be tuned
        print(f"Using HMM with {hmm_n_states} states.")
        detected_eras_hmm, hmm_model_trained = detect_eras_hmm(counts_series.copy(), n_states=hmm_n_states)
        if detected_eras_hmm is not None and not detected_eras_hmm.empty:
            print(f"HMM detected {len(detected_eras_hmm)} eras.")
            if hmm_model_trained:
                print("Trained HMM State Emission Parameters:")
                # Check if we are using the new pomegranate (torch-based distributions)
                is_new_pomegranate = hasattr(hmm_model_trained.distributions[0], 'summarize') # summarize is a method in new dists
                # Or check type: isinstance(hmm_model_trained.distributions[0], torch.nn.Module)
                # A simpler check might be based on the name derived from the shim
                
                for i, state_obj in enumerate(hmm_model_trained.states):
                    dist = state_obj.distribution
                    try:
                        # Attempt to access parameters assuming pomegranate >= 1.0 (torch backend)
                        # The `Normal` from the shim should align with pomegranate.distributions.Normal for v1.x
                        if hasattr(dist, 'parameters') and isinstance(dist.parameters, list) and len(dist.parameters) >= 2 and isinstance(dist.parameters[0], torch.Tensor):
                            learned_mean = dist.parameters[0].cpu().item() 
                            learned_std = torch.sqrt(dist.parameters[1][0][0]).cpu().item() 
                            dist_name_for_print = f"{dist.name} (PyTorch)" 
                        else: # Fallback for older pomegranate or different structure
                            # For older pomegranate.distributions.NormalDistribution, params might be .parameters[0] for mean, .parameters[1] for stdev
                            # Or .mu, .sigma, or .mean(), .stddev() methods. This needs to be robust or specific to version.
                            # Let's assume a common older pattern if not torch tensor params
                            if hasattr(dist, 'mu') and hasattr(dist, 'sigma'): # Common for older simple distributions
                                 learned_mean = float(dist.mu)
                                 learned_std = float(dist.sigma)
                            elif hasattr(dist, 'parameters') and isinstance(dist.parameters, (list, tuple)) and len(dist.parameters) >=2 : # e.g. older NormalDistribution
                                learned_mean = float(dist.parameters[0])
                                learned_std = float(dist.parameters[1])
                            else: # Can't determine parameters easily
                                learned_mean = float('nan')
                                learned_std = float('nan')
                            dist_name_for_print = f"{dist.name} (Old API?)"

                        data_for_predict = counts_series.values.astype(np.float32).reshape(-1,1)
                        all_predicted_states = hmm_model_trained.predict(data_for_predict)
                        state_days_counts = counts_series[all_predicted_states == i]
                        actual_mean_in_state = state_days_counts.mean() if not state_days_counts.empty else float('nan')
                        print(f"  State {i} ({dist_name_for_print}): Learned Dist: mu={learned_mean:.2f}, sigma={learned_std:.2f}. Actual counts mean: {actual_mean_in_state:.2f}")
                    except Exception as e:
                        print(f"  State {i}: Error retrieving or processing parameters for distribution '{dist.name}': {e}")
        else:
            print("HMM detected no eras or an error occurred.")

        print("\n--- Running BOCPD Detection ---")
        bocpd_hazard_lambda = 200 # Default, can be tuned
        # Note: The detect_eras_with_bocpd now returns (eras, pcp_matrix)
        detected_eras_bocpd, bocpd_pcp_matrix = detect_eras_with_bocpd(counts_series.copy(), hazard_lambda=bocpd_hazard_lambda)
        if detected_eras_bocpd is not None and not detected_eras_bocpd.empty:
            print(f"BOCPD detected {len(detected_eras_bocpd)} eras (using Gaussian approx.)")
        else:
            print("BOCPD detected no eras or an error occurred (using Gaussian approx.)")

    else:
        print("Counts series is empty, skipping era detection.")

    # --- Generate Report --- 
    print("\n--- Generating Report ---")
    report_filepath = "era_detection_report.txt"
    generate_era_report(
        eras_ruptures=detected_eras_ruptures, 
        eras_hmm=detected_eras_hmm,
        eras_bocpd=detected_eras_bocpd, # Pass BOCPD eras to report
        output_file_path=report_filepath
    )

    # --- Plotting --- 
    if not counts_series.empty:
        plt.figure(figsize=(18, 7)) # Wider for more methods
        counts_series.plot(label='Daily Non-Null Column Count', alpha=0.6, color='gray')
        
        # Plot Ruptures eras
        if detected_eras_ruptures is not None and not detected_eras_ruptures.empty:
            for i, era in enumerate(detected_eras_ruptures):
                plt.axvline(era.left, color='red', linestyle='--', linewidth=1.2, label='Ruptures Era' if i == 0 else None)
            if not detected_eras_ruptures.empty: 
                 plt.axvline(detected_eras_ruptures[-1].right - pd.Timedelta(days=1), color='red', linestyle='--', linewidth=1.2)

        # Plot HMM eras
        if detected_eras_hmm is not None and not detected_eras_hmm.empty:
            for i, era in enumerate(detected_eras_hmm):
                plt.axvline(era.left, color='blue', linestyle=':', linewidth=1.2, label='HMM Era' if i == 0 else None)
            if not detected_eras_hmm.empty:
                 plt.axvline(detected_eras_hmm[-1].right - pd.Timedelta(days=1), color='blue', linestyle=':', linewidth=1.2)

        # Plot BOCPD eras (if detected)
        if detected_eras_bocpd is not None and not detected_eras_bocpd.empty:
            for i, era in enumerate(detected_eras_bocpd):
                plt.axvline(era.left, color='green', linestyle='-', linewidth=1.2, label='BOCPD Era' if i == 0 else None)
            if not detected_eras_bocpd.empty:
                 plt.axvline(detected_eras_bocpd[-1].right - pd.Timedelta(days=1), color='green', linestyle='-', linewidth=1.2)

        # Add plotting for BOCPD eras here in later phases
        
        plt.title('Detected Eras on Daily Column Counts')
        plt.xlabel('Date')
        plt.ylabel('Count of Non-Null Columns')
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1)) # Adjust legend if too many items
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside
        plt.savefig("era_plot_all_methods.png") # New name
        print("\nPlot saved to era_plot_all_methods.png")
    else:
        print("Counts series empty, skipping plot generation.")

    print("\nScript finished.")

