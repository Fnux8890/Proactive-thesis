import pandas as pd
from pathlib import Path
from ruptures.detection import Pelt
# from sktime.dists_kernels import RBF # RBF is a kernel, usually specified by string in Pelt model
import numpy as np

from functools import partial
# import torch
# import pyro

# --- Level-C Imports (Now uses rusthmm) ---
try:
    from rusthmm import viterbi_path
    RUSTHMM_AVAILABLE = True
    print("Successfully imported viterbi_path from rusthmm.")
except ImportError:
    print("Warning: rusthmm module not found. Level C will be skipped.")
    RUSTHMM_AVAILABLE = False

from tqdm import tqdm
import argparse # Added for command-line arguments

# --- Configuration ---
# Default input path, can be overridden by --input-parquet argument
DEFAULT_PROCESSED_SEGMENT_PARQUET_PATH = Path("/app/data/processed/your_segment_file.parquet")

# Output directory for era label files (Levels A, B, C)
# This should align with where EraFeatureGenerator in preprocess.py expects them
# (i.e., feature_extraction/data/processed/ on the host, /app/data/processed in container)
OUTPUT_DIR = Path("/app/data/processed/")

# Columns to be used for segmentation, as per the game plan
# TODO: Verify these column names exist in your processed segment files.
#       Adjust if necessary (e.g., dli_sum might have a different name like dli_daily_aggregated).
SEGMENTATION_COLUMNS = [
    "dli_sum", 
    "pipe_temp_1_c", 
    "pipe_temp_2_c",
    "outside_temp_c", 
    "radiation_w_m2", 
    "co2_measured_ppm"
]

def load_and_prepare_data_slice(parquet_path: Path, seg_cols: list) -> tuple:
    """
    Loads data from a Parquet file, selects segmentation columns,
    and interpolates missing values in those columns.
    """
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
        print("Please update PROCESSED_SEGMENT_PARQUET_PATH in the script or provide a valid --input-parquet path.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Loading data from: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Successfully loaded DataFrame. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        time_col_name = 'time' 
        
        if time_col_name not in df.columns:
            if df.index.name == time_col_name and isinstance(df.index, pd.DatetimeIndex):
                print(f"'{time_col_name}' is already the DatetimeIndex.")
            else:
                print(f"Error: Time column '{time_col_name}' not found in DataFrame columns or as DatetimeIndex.")
                print(f"Available columns: {df.columns.tolist()}, Index name: {df.index.name}")
                return pd.DataFrame(), pd.DataFrame()
        else:
            print(f"Setting '{time_col_name}' as DatetimeIndex...")
            df[time_col_name] = pd.to_datetime(df[time_col_name])
            df = df.set_index(time_col_name)

        missing_seg_cols = [col for col in seg_cols if col not in df.columns]
        if missing_seg_cols:
            print(f"Error: The following segmentation columns are missing from the DataFrame: {missing_seg_cols}")
            print(f"Available columns after setting index: {df.columns.tolist()}")
            return pd.DataFrame(), pd.DataFrame()
            
        X = df[seg_cols].copy() 
        print(f"Selected segmentation columns: {seg_cols}. Shape of X: {X.shape}")
        
        print("Interpolating missing values in segmentation columns (limit_direction='both')...")
        X.interpolate(method='linear', limit_direction='both', inplace=True)
        
        if X.isnull().any().any():
            print("Warning: Segmentation features (X) still contain NaNs after interpolation:")
            print(X.isnull().sum())
        else:
            print("Segmentation features (X) are now complete (no NaNs).")
            
        print(f"Prepared data slice X. Shape: {X.shape}")
        return X, df

    except Exception as e:
        print(f"An error occurred during data loading or preparation: {e}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Era Detection Script")
    parser.add_argument(
        "--input-parquet", 
        type=Path,
        default=DEFAULT_PROCESSED_SEGMENT_PARQUET_PATH,
        help="Path to the processed segment Parquet file."
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="your_segment_file", # Default suffix if none provided
        help="Suffix to use for output era label files (e.g., 'MegaEra1'). Output will be <suffix>_era_labels_levelX.parquet."
    )
    args = parser.parse_args()

    # Ensure output directory exists (moved here to run once)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    processed_segment_path = args.input_parquet
    output_file_suffix = args.output_suffix.replace(".parquet", "") # Ensure .parquet is not in suffix itself

    print(f"--- Era Detection Script ---")
    print(f"Input Parquet: {processed_segment_path}")
    print(f"Output Suffix: {output_file_suffix}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    X_segmentation_features, df_full_segment = load_and_prepare_data_slice(
        processed_segment_path, 
        SEGMENTATION_COLUMNS
    )

    if not X_segmentation_features.empty and not df_full_segment.empty:
        print("\nSuccessfully prepared X_segmentation_features and df_full_segment.")
        print("X_segmentation_features head:")
        print(X_segmentation_features.head())
        print("\ndf_full_segment (original with all columns) head:")
        print(df_full_segment.head())
        
        # Next steps (Level A, B, C) will use X_segmentation_features
        # and add era labels back to df_full_segment.
        
        print("\nTODO: Implement Level A - PELT (30 min)")
        # --- Level A: PELT Implementation ---
        print("\n--- Level A: PELT Segmentation ---")
        try:
            # Ensure X_segmentation_features has a simple RangeIndex for Pelt if it expects one
            # However, sktime usually handles DataFrames with various index types.
            # If X_segmentation_features has a DatetimeIndex, Pelt should work.
            
            # Initialize PELT detector
            # model="rbf" implies a Gaussian model with an RBF kernel for change point detection.
            # min_size=48: minimum size of a segment (e.g., if data is 5T, 48*5 = 240 min = 4 hours)
            # jump=1: check every point as a potential changepoint
            detector_A = Pelt(model="rbf", min_size=48, jump=1) 
            print("PELT detector initialized.")

            # Fit the detector first
            detector_A.fit(X_segmentation_features)
            # Then predict with a penalty value. This was missing the 'pen' argument.
            # Common penalty values might range from log(n_samples) to 3*log(n_samples) or be tuned.
            # Using a placeholder penalty, e.g., 10, adjust as needed.
            penalty_value_A = 10 
            print(f"Fitting PELT detector and predicting breakpoints with penalty={penalty_value_A}...")
            bkps_indices_A = detector_A.predict(pen=penalty_value_A)
            print(f"Detected {len(bkps_indices_A)} breakpoints at row indices: {bkps_indices_A}")

            # Label eras in the original DataFrame
            df_full_segment['era_level_A'] = 0 # Default to era 0
            start_idx = 0
            for i, bkp_row_idx in enumerate(bkps_indices_A):
                # bkp_row_idx is the first index of the NEW segment
                df_full_segment.iloc[start_idx:bkp_row_idx, df_full_segment.columns.get_loc('era_level_A')] = i
                start_idx = bkp_row_idx
            # Label the last segment
            df_full_segment.iloc[start_idx:, df_full_segment.columns.get_loc('era_level_A')] = len(bkps_indices_A)
            
            print("\n'era_level_A' column added to df_full_segment:")
            print(df_full_segment['era_level_A'].value_counts().sort_index())

            # Persist era labels
            # The EraFeatureGenerator expects 'time' as a column or index.
            # Since df_full_segment has 'time' as index, we reset it for saving.
            df_to_save_A = df_full_segment[['era_level_A']].reset_index()
            time_col_name = df_to_save_A.columns[0] 
            
            era_labels_A_path = OUTPUT_DIR / f"{output_file_suffix}_era_labels_levelA.parquet" # Use suffix
            df_to_save_A.to_parquet(era_labels_A_path, index=False)
            print(f"Saved Level A era labels (columns: ['{time_col_name}', 'era_level_A']) to {era_labels_A_path}")

        except Exception as e:
            print(f"Error during Level A (PELT) processing: {e}")
        # ... (Code for Level A will go here) ...
        # era_labels_A_path = OUTPUT_DIR / "era_labels_levelA.parquet"
        # df_full_segment[['time_col_name_if_not_index', 'era']].to_parquet(era_labels_A_path) 
        # print(f"Saved Level A era labels to {era_labels_A_path}")

        print("\nTODO: Implement Level B - Bayesian CPD (45 min)")
        # --- Level B: Bayesian Online CPD via changepoint library --- 
        print("\n--- Level B: Bayesian Changepoint Detection ---")
        try:
            # Select a univariate signal for Bayesian CPD as per plan
            # TODO: Ensure 'dli_sum' is the correct and available column name in X_segmentation_features
            signal_B_col_name = "dli_sum" 
            if signal_B_col_name not in X_segmentation_features.columns:
                print(f"Error: Column '{signal_B_col_name}' not found in X_segmentation_features for Level B. Skipping.")
                raise KeyError(f"Column '{signal_B_col_name}' not available for Bayesian CPD.")

            signal_B = X_segmentation_features[signal_B_col_name].values
            print(f"Using signal '{signal_B_col_name}' for Bayesian CPD. Length: {len(signal_B)}")

            # Expected segment length (e.g., ~two weeks if data is hourly)
            # Adjust 'l' based on your data's actual frequency and expected era duration
            # Example: If 5T data, 24*12 = 288 samples/day. 14 days = 288*14 = 4032.
            # For now, using a generic value; adjust based on X_segmentation_features frequency.
            # Assuming min_size=48 for PELT was for ~2 days on hourly, let's use 24*7 for a week roughly.
            # This 'l' parameter is the lambda for the exponential prior on segment lengths.
            # If your data has frequency 'F' (e.g., '5T'), calculate expected_samples_per_day.
            # Then samples_for_two_weeks = expected_samples_per_day * 14.
            # For now, a placeholder, assuming it might be daily or similar. Adjust if very high frequency.
            expected_segment_len_samples = 24 * 7 # Placeholder: 1 week of hourly data
            if X_segmentation_features.index.inferred_freq:
                freq_str = X_segmentation_features.index.inferred_freq
                try:
                    samples_per_day_approx = pd.Timedelta(days=1) / pd.to_timedelta(freq_str)
                    expected_segment_len_samples = int(samples_per_day_approx * 14) # ~two weeks
                    print(f"  Inferred frequency: {freq_str}. Using expected segment length: {expected_segment_len_samples} samples for prior.")
                except ValueError:
                    print(f"  Could not parse inferred frequency '{freq_str}' to calculate samples per day. Using default prior length.")
            else:
                print("  Could not infer frequency from X_segmentation_features index. Using default prior length.")

            # --- Level B: Bayesian Online CPD via changepoint library --- 
            from changepoint import Bocpd, NormalGamma # NormalGamma is typical for unknown mean/variance
            print("Initializing BOCPD with NormalGamma model from 'changepoint' library...")

            # For changepoint 0.3.2, the observation model is the first positional arg.
            bocpd_model_B = Bocpd(NormalGamma(), lam=float(expected_segment_len_samples)) 

            cp_posterior_prob = np.empty(len(signal_B), dtype=float)
            print(f"Running online BOCPD updates with changepoint.step() for {len(signal_B)} data points...")
            for t, x_val in enumerate(signal_B):
                # Bocpd.step() returns the changepoint probability for the current point t
                bocpd_model_B.step(float(x_val)) 
                # The probability P(R_t=0|x_{1:t}) is stored in model.beliefs
                # This is probability of current point being a changepoint (i.e., run length is 0)
                cp_posterior_prob[t] = bocpd_model_B.rt # P(r_t=0 | x_{1:t}), probability current point is a changepoint

            print("Online Bayesian changepoint detection (changepoint lib) completed.")
            # cp_posterior_prob now contains P(cp at t | x_1:t)
            df_full_segment['cp_prob_level_B'] = cp_posterior_prob
            
            # Create eras based on thresholding cp_prob (MAP eras)
            df_full_segment['era_level_B'] = (df_full_segment['cp_prob_level_B'] > 0.5).cumsum()

            print("\n'cp_prob_level_B' and 'era_level_B' columns added to df_full_segment.")
            print("Summary of 'cp_prob_level_B':")
            print(df_full_segment['cp_prob_level_B'].describe())
            print("Value counts for 'era_level_B':")
            print(df_full_segment['era_level_B'].value_counts().sort_index())

            # Persist era labels and probabilities
            df_to_save_B = df_full_segment[['cp_prob_level_B', 'era_level_B']].reset_index()
            time_col_name_B = df_to_save_B.columns[0] 

            era_labels_B_path = OUTPUT_DIR / f"{output_file_suffix}_era_labels_levelB.parquet" # Use suffix
            df_to_save_B.to_parquet(era_labels_B_path, index=False)
            print(f"Saved Level B era labels (columns: ['{time_col_name_B}', 'cp_prob_level_B', 'era_level_B']) to {era_labels_B_path}")

        except KeyError as ke:
            # Handled already by the check for signal_B_col_name
            print(f"Skipping Level B due to missing column: {ke}")
        except Exception as e:
            print(f"Error during Level B (Bayesian CPD) processing: {e}")
            import traceback
            traceback.print_exc()
        # ... (Code for Level B will go here) ...
        # era_labels_B_path = OUTPUT_DIR / "era_labels_levelB.parquet"
        # df_full_segment[['time_col_name_if_not_index', 'cp_prob', 'era_B']].to_parquet(era_labels_B_path)
        # print(f"Saved Level B era labels to {era_labels_B_path}")


        print("\nTODO: Implement Level C - Sticky HDP-HMM (60 min)")
        # --- Level C: Sticky HDP-HMM --- 
        print("\n--- Level C: Sticky HDP-HMM Segmentation ---")
        # Guard for HDPHMM availability
        if not RUSTHMM_AVAILABLE:
            print("⚠️  rusthmm.viterbi_path not available – Level C skipped.")
        else:
            try:
                # For rusthmm, we need a single 1D numpy array of discrete observations.
                # We'll use 'dli_sum' as the primary signal for Level C as well, similar to Level B.
                # The values need to be discretized. For simplicity, we can round and cast to uint8.
                # A more robust approach would be to use pd.qcut or KMeans for discretization.
                signal_C_col_name = "dli_sum"
                if signal_C_col_name not in X_segmentation_features.columns:
                    print(f"Error: Column '{signal_C_col_name}' not found for Level C. Skipping.")
                    raise KeyError(f"Column '{signal_C_col_name}' not available for HMM.")

                # Discretize the signal: scale, round, and cast to uint8.
                # This is a simple discretization. Adjust as needed.
                signal_C_continuous = X_segmentation_features[signal_C_col_name].values
                signal_C_min = signal_C_continuous.min()
                signal_C_max = signal_C_continuous.max()
                if signal_C_max == signal_C_min: # Avoid division by zero if signal is constant
                    signal_C_discrete = np.zeros_like(signal_C_continuous, dtype=np.uint8)
                else:
                    # Scale to 0-254 to leave room for 255 as a potential max symbol if needed, and to handle float precision.
                    signal_C_scaled = 254 * (signal_C_continuous - signal_C_min) / (signal_C_max - signal_C_min)
                    signal_C_discrete = np.round(signal_C_scaled).astype(np.uint8)

                print(f"Discretized signal '{signal_C_col_name}' for HMM. Min: {signal_C_discrete.min()}, Max: {signal_C_discrete.max()}")

                n_states_C = 10 # Number of hidden states for the HMM (tuneable)
                n_iter_C = 30   # Number of Baum-Welch iterations (tuneable)
                print(f"Running Viterbi path with rusthmm: n_states={n_states_C}, n_iter={n_iter_C}...")

                states_np = viterbi_path(
                    signal_C_discrete, 
                    n_states_C, 
                    n_iter_C
                )
                print("Viterbi path calculation completed.")

                num_inferred_states = n_states_C # rusthmm doesn't dynamically adjust number of states like HDPHMM

                # Add Viterbi path (most likely era sequence)
                df_full_segment['era_level_C'] = states_np

                print(f"\nInferred {num_inferred_states} states for Level C.")
                print("'era_level_C' column added to df_full_segment.") # Note: rusthmm only provides Viterbi path, not state probabilities gamma
                print("Value counts for 'era_level_C':")
                print(df_full_segment['era_level_C'].value_counts().sort_index())
                
                # Persist era labels (just the Viterbi path for now)
                cols_to_save_C = ['era_level_C'] 
                df_to_save_C = df_full_segment[cols_to_save_C].reset_index()
                time_col_name_C = df_to_save_C.columns[0] 

                era_labels_C_path = OUTPUT_DIR / f"{output_file_suffix}_era_labels_levelC.parquet" # Use suffix
                df_to_save_C.to_parquet(era_labels_C_path, index=False)
                print(f"Saved Level C era labels (columns: ['{time_col_name_C}'] + {cols_to_save_C}) to {era_labels_C_path}")

            except ImportError as ie:
                print(f"Could not import Pyro/Torch for Level C. Ensure they are installed. Error: {ie}")
            except Exception as e:
                print(f"Error during Level C (HDP-HMM) processing: {e}")
                import traceback
                traceback.print_exc()

    else:
        print("\nFailed to load or prepare data. Please check the path and column names.")

    print("\n--- Era Detection Script Finished ---") 

if __name__ == "__main__":
    main() 