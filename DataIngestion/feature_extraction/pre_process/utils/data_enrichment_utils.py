import json
import logging
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pvlib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

LITERATURE_PHENOTYPES_TABLE_NAME = "literature_kalanchoe_phenotypes"


def load_literature_phenotypes(engine) -> pd.DataFrame:
    print(
        f"Loading literature Kalanchoe phenotypes from table: {LITERATURE_PHENOTYPES_TABLE_NAME}..."
    )
    query = f"SELECT * FROM public.{LITERATURE_PHENOTYPES_TABLE_NAME} WHERE phenotype_value IS NOT NULL;"
    try:
        pheno_df = pd.read_sql_query(query, engine)
        print(f"DEBUG: Columns in pheno_df after read_sql_query: {pheno_df.columns.tolist()}")
        print("DEBUG: pheno_df.info() after read_sql_query:")
        pheno_df.info()
        if pheno_df.empty:
            print(
                "DEBUG: pheno_df is empty after read_sql_query (possibly due to 'WHERE phenotype_value IS NOT NULL' and no non-null values)."
            )

        cols_to_convert = [
            "phenotype_value",
            "environment_temp_day_c",
            "environment_temp_night_c",
            "environment_photoperiod_h",
            "environment_dli_mol_m2_d",
            "environment_light_intensity_umol_m2_s",
            "environment_co2_ppm",
            "environment_rh_percent",
        ]

        valid_cols_to_convert = [col for col in cols_to_convert if col in pheno_df.columns]
        print(f"DEBUG: Valid columns for numeric conversion: {valid_cols_to_convert}")

        for col in valid_cols_to_convert:
            pheno_df[col] = pd.to_numeric(pheno_df[col], errors="coerce")

        essential_cols_for_proxy = [
            "phenotype_value",
            "environment_dli_mol_m2_d",
            "environment_temp_day_c",
            "environment_photoperiod_h",
            "species",
            "phenotype_name",
            "phenotype_unit",
        ]

        valid_essential_cols_for_proxy = [
            col for col in essential_cols_for_proxy if col in pheno_df.columns
        ]
        print(f"DEBUG: Valid essential columns for dropna: {valid_essential_cols_for_proxy}")

        if not valid_essential_cols_for_proxy:
            print(
                f"DEBUG: No essential columns for proxy model found in pheno_df. Columns available: {pheno_df.columns.tolist()}"
            )

        if valid_essential_cols_for_proxy:
            pheno_df.dropna(subset=valid_essential_cols_for_proxy, inplace=True)
        else:
            print(
                "DEBUG: Skipping dropna because no valid essential_cols_for_proxy were found in the DataFrame."
            )

        print(f"Loaded and cleaned {len(pheno_df)} rows from {LITERATURE_PHENOTYPES_TABLE_NAME}.")
        return pheno_df
    except Exception as e:
        print(f"Error loading literature Kalanchoe phenotypes: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def generate_proxy_phenotype_labels(
    era_aggregated_gh_data: pd.DataFrame,
    literature_phenotypes_df: pd.DataFrame,
    target_phenotype_info: dict[str, str],
    literature_model_features: list[str],
    greenhouse_data_features: list[str],
) -> pd.Series:
    phenotype_name = target_phenotype_info["name"]
    phenotype_unit = target_phenotype_info["unit"]
    target_species = target_phenotype_info["species"]

    print(f"  Generating proxy labels for: {target_species} - {phenotype_name} ({phenotype_unit})")

    pheno_subset = literature_phenotypes_df[
        (
            literature_phenotypes_df["species"].str.contains(
                target_species, case=False, na=False, regex=False
            )
        )
        & (literature_phenotypes_df["phenotype_name"] == phenotype_name)
        & (literature_phenotypes_df["phenotype_unit"] == phenotype_unit)
    ]

    if (
        pheno_subset.empty
        or pheno_subset[literature_model_features + ["phenotype_value"]].isnull().all().any()
    ):
        print(
            f"    Not enough data or missing required columns in literature for {target_species}, {phenotype_name}. Required: {literature_model_features}. Available in pheno_subset: {pheno_subset.columns.tolist()}"
        )
        return pd.Series(
            dtype="float64",
            index=era_aggregated_gh_data.index,
            name=f"{phenotype_name}_proxy_pred_error",
        )

    pheno_subset = pheno_subset.dropna(subset=literature_model_features + ["phenotype_value"])
    if len(pheno_subset) < 3:
        print(
            f"    Too few ({len(pheno_subset)}) valid data points in literature for {target_species}, {phenotype_name} to train proxy model. Skipping."
        )
        return pd.Series(
            dtype="float64",
            index=era_aggregated_gh_data.index,
            name=f"{phenotype_name}_proxy_pred_insufficient_data",
        )

    X_lit = pheno_subset[literature_model_features]
    y_lit = pheno_subset["phenotype_value"]

    model = KNeighborsRegressor(n_neighbors=min(3, len(X_lit)))
    try:
        model.fit(X_lit, y_lit)
    except Exception as e:
        print(f"    Error fitting proxy model for {phenotype_name}: {e}")
        return pd.Series(
            dtype="float64",
            index=era_aggregated_gh_data.index,
            name=f"{phenotype_name}_proxy_pred_fit_error",
        )

    if not all(f in era_aggregated_gh_data.columns for f in greenhouse_data_features):
        print(
            f"    Missing one or more required greenhouse features for prediction: {greenhouse_data_features}. Available: {era_aggregated_gh_data.columns.tolist()}"
        )
        return pd.Series(
            dtype="float64",
            index=era_aggregated_gh_data.index,
            name=f"{phenotype_name}_proxy_pred_missing_gh_feat",
        )

    X_gh = era_aggregated_gh_data[greenhouse_data_features].copy()

    for i, lit_col_name in enumerate(literature_model_features):
        gh_col_name = greenhouse_data_features[i]
        if X_gh[gh_col_name].isnull().any():
            fill_value = X_lit[lit_col_name].mean()
            X_gh.loc[:, gh_col_name] = X_gh[gh_col_name].fillna(fill_value)
            print(
                f"    Filled NaNs in greenhouse feature '{gh_col_name}' with mean from literature training data ({fill_value:.2f})."
            )

    try:
        predicted_values = model.predict(X_gh)
        print(f"    Successfully generated proxy labels for {phenotype_name}.")
        return pd.Series(
            predicted_values,
            index=era_aggregated_gh_data.index,
            name=f"{phenotype_name}_proxy_pred",
        )
    except Exception as e:
        print(f"    Error during prediction for {phenotype_name}: {e}")
        return pd.Series(
            dtype="float64",
            index=era_aggregated_gh_data.index,
            name=f"{phenotype_name}_proxy_pred_predict_error",
        )


def calculate_daily_weekly_aggregates(
    df_era_processed: pd.DataFrame, era_identifier: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_era_processed.empty or not isinstance(df_era_processed.index, pd.DatetimeIndex):
        print(
            f"  (Era: {era_identifier}) DataFrame empty or not DatetimeIndexed, skipping daily/weekly aggregation."
        )
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Calculating daily and weekly aggregated features for Era '{era_identifier}'...")
    numeric_cols_for_agg = df_era_processed.select_dtypes(include=np.number).columns.tolist()

    daily_aggs_dict = {}
    weekly_aggs_dict = {}

    for col in numeric_cols_for_agg:
        daily_aggs_dict[f"{col}_daily_mean"] = (col, "mean")
        daily_aggs_dict[f"{col}_daily_min"] = (col, "min")
        daily_aggs_dict[f"{col}_daily_max"] = (col, "max")
        daily_aggs_dict[f"{col}_daily_std"] = (col, "std")

        weekly_aggs_dict[f"{col}_weekly_mean"] = (col, "mean")
        weekly_aggs_dict[f"{col}_weekly_min"] = (col, "min")
        weekly_aggs_dict[f"{col}_weekly_max"] = (col, "max")
        weekly_aggs_dict[f"{col}_weekly_std"] = (col, "std")

        if (
            "dli" in col.lower()
            or "radiation" in col.lower()
            or "light" in col.lower()
            or "dosing_status" in col.lower()
        ):
            daily_aggs_dict[f"{col}_daily_sum"] = (col, "sum")
            weekly_aggs_dict[f"{col}_weekly_sum"] = (col, "sum")

    df_daily_features = pd.DataFrame()
    df_weekly_features = pd.DataFrame()

    if daily_aggs_dict:
        df_daily_features = df_era_processed.groupby(df_era_processed.index.normalize()).agg(
            **{k: v for k, v in daily_aggs_dict.items() if v[0] in df_era_processed.columns}
        )
    if weekly_aggs_dict:
        df_weekly_features = df_era_processed.groupby(pd.Grouper(freq="W-MON")).agg(
            **{k: v for k, v in weekly_aggs_dict.items() if v[0] in df_era_processed.columns}
        )

    print(
        f"  Daily aggregates shape: {df_daily_features.shape}, Weekly aggregates shape: {df_weekly_features.shape}"
    )
    return df_daily_features, df_weekly_features


class EraFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, era_file_path: Path):
        self.era_file_path = era_file_path
        self.era_df = None
        try:
            print(f"Attempting to load era features from: {self.era_file_path}")
            self.era_df = pd.read_parquet(self.era_file_path)

            if "time" in self.era_df.columns:
                self.era_df["time"] = pd.to_datetime(self.era_df["time"], errors="coerce", utc=True)
                self.era_df = self.era_df.set_index("time")
            elif self.era_df.index.name == "time":
                if not isinstance(self.era_df.index, pd.DatetimeIndex):
                    self.era_df.index = pd.to_datetime(self.era_df.index, errors="coerce", utc=True)
                elif self.era_df.index.tz is None:
                    self.era_df.index = self.era_df.index.tz_localize("UTC")
            else:
                raise ValueError(
                    f"'time' column or index not found in era file: {self.era_file_path}"
                )

            self.era_df.dropna(axis=1, how="all", inplace=True)
            self.era_df = self.era_df[self.era_df.index.notna()]

            print(
                f"Successfully loaded era features. Index type: {type(self.era_df.index)}, Columns: {self.era_df.columns.tolist()}"
            )
            if self.era_df.empty:
                print(
                    f"Warning: Era feature DataFrame is empty after loading and time processing from {self.era_file_path}."
                )

        except FileNotFoundError:
            print(
                f"Warning: Era feature file not found at {self.era_file_path}. Era features will not be added."
            )
            self.era_df = pd.DataFrame()
        except Exception as e:
            print(f"Error loading or processing era feature file {self.era_file_path}: {e}")
            self.era_df = pd.DataFrame()

    def fit(self):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.era_df is None or self.era_df.empty:
            print("Era DataFrame not loaded or empty, returning original DataFrame.")
            return X

        if not isinstance(X.index, pd.DatetimeIndex):
            print(
                "Warning: Input DataFrame X to EraFeatureGenerator does not have a DatetimeIndex. Transformation might fail or be incorrect."
            )
            return X

        X_work = X.copy()
        if self.era_df.index.tz is not None and X_work.index.tz is None:
            print(
                f"Warning: Era DF index is TZ-aware ({self.era_df.index.tz}) but input X index is naive. Localizing X to UTC for join."
            )
            X_work.index = X_work.index.tz_localize("UTC", ambiguous="infer", nonexistent="NaT")
            X = X[X_work.index.notna()]
        elif self.era_df.index.tz is None and X_work.index.tz is not None:
            print(
                f"Warning: Era DF index is naive but input X index is TZ-aware ({X_work.index.tz}). Making X naive for join."
            )
            X_work.index = X_work.index.tz_convert(None)

        try:
            X_transformed = X.join(self.era_df, how="left", rsuffix="_era")

            era_cols_potentially_added = self.era_df.columns.tolist()
            newly_added_cols = [
                c
                for c in X_transformed.columns
                if c not in X.columns
                and (
                    c in era_cols_potentially_added
                    or c.replace("_era", "") in era_cols_potentially_added
                )
            ]

            if newly_added_cols:
                print(
                    f"Joined era features: {newly_added_cols}. Applying ffill/bfill to these columns."
                )
                X_transformed[newly_added_cols] = X_transformed[newly_added_cols].ffill().bfill()
            else:
                print(
                    "No new era columns seem to have been added by the join, or they were all NaNs."
                )

            print(f"Shape after attempting to join era features: {X_transformed.shape}")
            return X_transformed
        except Exception as e:
            print(f"Error during era feature transformation (join): {e}")
            return X


def scale_data_for_era(
    df: pd.DataFrame,
    era_identifier: str,
    era_config: dict[str, Any],
    global_config: dict[str, Any],
    scalers_dir: Path,
    fit_scalers: bool = True,
    existing_scalers: dict[str, MinMaxScaler] = None,
) -> tuple[pd.DataFrame, dict[str, MinMaxScaler]]:
    if df.empty:
        print(f"(Era: {era_identifier}) DataFrame is empty, skipping scaling.")
        return df, {}

    print(f"\n--- Scaling Data for Era: {era_identifier} (fit_scalers={fit_scalers}) ---")
    scalers_dir.mkdir(parents=True, exist_ok=True)
    era_scalers_path = scalers_dir / f"{era_identifier}_scalers.joblib"

    cols_to_scale = []
    boolean_cols_converted = era_config.get(
        "boolean_columns_to_int",
        global_config.get("common_settings", {}).get("boolean_columns_to_int", []),
    )

    defined_cols_in_rules = set()
    outlier_rules_ref = era_config.get("outlier_rules_ref", "default_outlier_rules")
    imputation_rules_ref = era_config.get("imputation_rules_ref", "default_imputation_rules")
    outlier_rules = global_config.get("preprocessing_rules", {}).get(outlier_rules_ref, [])
    imputation_rules = global_config.get("preprocessing_rules", {}).get(imputation_rules_ref, [])
    for rule in outlier_rules:
        defined_cols_in_rules.add(rule["column"])
    for rule in imputation_rules:
        defined_cols_in_rules.add(rule["column"])

    for col in df.columns:
        if col in boolean_cols_converted:
            print(
                f"  Skipping MinMaxScaling for column '{col}' (identified as boolean/status 0/1)."
            )
            continue

        if (
            df[col].dtype == "float64"
            or df[col].dtype == "float32"
            or (
                col in defined_cols_in_rules
                and pd.api.types.is_numeric_dtype(df[col])
                and col not in cols_to_scale
            )
        ):
            cols_to_scale.append(col)

    print(f"Identified columns for MinMaxScaling to [-1,1]: {cols_to_scale}")

    df_scaled = df.copy()
    current_scalers = existing_scalers if existing_scalers is not None else {}

    if fit_scalers:
        for col in cols_to_scale:
            if col in df_scaled.columns and not df_scaled[col].isnull().all():
                data_to_scale = df_scaled[[col]].astype(float)
                if data_to_scale.dropna().empty:
                    print(
                        f"  Column '{col}' contains all NaN or non-convertible values. Skipping scaling."
                    )
                    continue
                scaler = MinMaxScaler(feature_range=(-1, 1))
                df_scaled[col] = scaler.fit_transform(data_to_scale)
                current_scalers[col] = scaler
                print(f"  Fitted and transformed column: {col}")
        if current_scalers:
            joblib.dump(current_scalers, era_scalers_path)
            print(f"Saved fitted scalers for Era '{era_identifier}' to {era_scalers_path}")
    else:
        if not current_scalers and era_scalers_path.exists():
            current_scalers = joblib.load(era_scalers_path)
            print(f"Loaded scalers for Era '{era_identifier}' from {era_scalers_path}")
        elif not current_scalers and not era_scalers_path.exists():
            print(
                f"Warning: fit_scalers is False and Scaler file not found at {era_scalers_path}. Cannot transform data."
            )
            return df_scaled, {}

        for col in cols_to_scale:
            if (
                col in df_scaled.columns
                and col in current_scalers
                and not df_scaled[col].isnull().all()
            ):
                data_to_transform = df_scaled[[col]].astype(float)
                if data_to_transform.dropna().empty:
                    print(
                        f"  Column '{col}' contains all NaN or non-convertible values after astype(float). Skipping transform."
                    )
                    continue
                df_scaled[col] = current_scalers[col].transform(data_to_transform)
                print(f"  Transformed column '{col}' using provided/loaded scaler.")
            elif col in df_scaled.columns and not df_scaled[col].isnull().all():
                print(
                    f"  Warning: No scaler provided/loaded for column '{col}'. Skipping scaling for it."
                )

    return df_scaled, current_scalers


def synthesize_light_data(
    df: pd.DataFrame,
    lat: float = 56.2661,
    lon: float = 10.064,
    use_cloud: bool = True,
    era_identifier: str = "UnknownEra",
    dli_scale_clip_min: float = 0.25,
    dli_scale_clip_max: float = 4.0,
) -> pd.DataFrame:
    """Return df with 'par_synth_umol' and a status dict.

    Synthesizes Photosynthetically Active Radiation (PAR) using a clear-sky model,
    adjusted for night time and optionally DLI and cloud cover.

    Args:
        df: Input DataFrame with a DatetimeIndex. Expected to have a discernible frequency (e.g., 5T, 15T).
            May contain 'dli_sum' (mol m-2 day-1) and 'radiation_w_m2' (W m-2) for adjustments.
        lat: Latitude for solar position calculation (default Hinnerup: 56.2661 N).
        lon: Longitude for solar position calculation (default Hinnerup: 10.064 E).
        use_cloud: Boolean flag to enable cloud correction using 'radiation_w_m2' if available.
        era_identifier: Identifier for the current era, for logging.
        dli_scale_clip_min: Minimum value for DLI scaling factor clip.
        dli_scale_clip_max: Maximum value for DLI scaling factor clip.

    Returns:
        DataFrame with added 'par_synth_umol' (µmol m-2 s-1) and 'light_synthesis_status' (JSON string) columns.
    """
    # Note: The imports pvlib, numpy, pandas, json, astral are expected to be at the top of the file.
    # Ensure df.index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for light synthesis.")

    logger = logging.getLogger(__name__)
    logger.info("Attempting light synthesis for era '%s' @ (%.4f, %.4f)", era_identifier, lat, lon)
    df_synthesized = df.copy()
    status = {}

    # 1. Clear-sky irradiance
    try:
        solpos = pvlib.solarposition.get_solarposition(df_synthesized.index, lat, lon)
        # Using pvlib's default for linke_turbidity by not specifying or passing None if the function handles it.
        # pvlib.clearsky.ineichen typically defaults Linke turbidity to 3 and pressure to 101325 Pa if not provided.
        # Explicitly setting linke_turbidity=3 as a common default, can be parameterized later if needed.
        clearsky = pvlib.clearsky.ineichen(solpos["apparent_zenith"], linke_turbidity=3)
        ghi_cs = clearsky["ghi"].fillna(
            0
        )  # W m-2, fill NaNs from clearsky model (e.g. extreme sun angles)
        par_cs = ghi_cs * 4.57  # Convert GHI (W m-2) to PAR (µmol m-2 s-1)
        status["clearsky_model"] = "Ineichen-Perez_pvlib"
    except Exception as e:
        logger.error(
            f"  Error in clear-sky model calculation for era {era_identifier}: {e}", exc_info=True
        )
        status["clearsky_model_error"] = str(e)
        df_synthesized["par_synth_umol"] = np.nan
        df_synthesized["light_synthesis_status"] = json.dumps(status)
        return df_synthesized

    # 2. Zero at night (astronomical definition: sun zenith < 90 degrees)
    # Ensure astral.LocationInfo can get timezone if df.index is timezone-aware, or assume UTC/naive
    # For simplicity, if df.index.tz is None, astral might use system's local timezone for sunrise/sunset,
    # which could be an issue if df.index is naive UTC. Best if df.index is tz-aware UTC.
    # The solarposition zenith angle is generally sufficient and timezone-agnostic if index is UTC.
    day_mask = solpos["zenith"] < 90  # True during daylight hours
    par_cs = par_cs.where(day_mask, 0.0)
    status["night_masking"] = "applied_zenith_lt_90"

    # 3. Daily scale to DLI if 'dli_sum' is available
    par_adj = par_cs.copy()  # Initialize adjusted PAR with clear-sky, night-masked PAR
    if "dli_sum" in df_synthesized.columns and not df_synthesized["dli_sum"].isnull().all():
        # Determine interval for integration (e.g., 5 minutes = 300 seconds)
        interval_s = None
        if df_synthesized.index.freq:
            interval_s = pd.to_timedelta(df_synthesized.index.freq).total_seconds()
        elif len(df_synthesized.index) > 1:
            interval_s = (df_synthesized.index[1] - df_synthesized.index[0]).total_seconds()

        if interval_s and interval_s > 0:
            # Calculate synthetic DLI from par_cs (µmol m-2 s-1 -> mol m-2 day-1)
            # Sum(par_cs_values_in_day) * interval_s / 1,000,000
            synthetic_dli_mol_daily = (
                par_cs.groupby(df_synthesized.index.date).sum() * interval_s
            ) / 1e6
            synthetic_dli_mol_daily = (
                synthetic_dli_mol_daily.reindex(df_synthesized.index.date).ffill().bfill()
            )

            # Get observed DLI, ensuring one value per day, broadcasted
            observed_dli_daily = (
                df_synthesized["dli_sum"].groupby(df_synthesized.index.date).first()
            )
            observed_dli_daily = (
                observed_dli_daily.reindex(df_synthesized.index.date).ffill().bfill()
            )

            # Create a daily scaling factor where observed DLI is available
            # Avoid division by zero or near-zero synthetic DLI if par_cs was all zero for a day (e.g. polar night)
            scale_factor_daily = (
                (observed_dli_daily / synthetic_dli_mol_daily.where(synthetic_dli_mol_daily > 1e-6))
                .clip(dli_scale_clip_min, dli_scale_clip_max)
                .fillna(1.0)
            )

            # Apply daily scale factor to the timeseries par_cs values
            # Map daily scale_factor back to the original DataFrame index
            daily_scale_map = df_synthesized.index.to_series().dt.date.map(scale_factor_daily)
            par_adj = par_cs * daily_scale_map
            status["dli_anchor"] = True
            logger.info(
                f"  (Era: {era_identifier}) DLI anchoring applied using 'dli_sum'. Min/Max scale: {scale_factor_daily.min():.2f}/{scale_factor_daily.max():.2f}"
            )
        else:
            logger.warning(
                f"  (Era: {era_identifier}) Could not determine data frequency for DLI scaling. Skipping DLI anchor."
            )
            status["dli_anchor"] = False
            status["dli_anchor_issue"] = "Could not determine data frequency"
    else:
        status["dli_anchor"] = False
        logger.info(
            f"  (Era: {era_identifier}) 'dli_sum' not available or all NaNs. Skipping DLI anchor."
        )

    # 4. Optional cloud correction using 'radiation_w_m2'
    if use_cloud:
        if (
            "radiation_w_m2" in df_synthesized.columns
            and not df_synthesized["radiation_w_m2"].isnull().all()
        ):
            # ghi_cs is clear-sky GHI (W m-2). df_synthesized['radiation_w_m2'] is observed GHI.
            # kt is the clearness index. Limit to [0,1] physically.
            # Ensure ghi_cs is not zero to prevent division errors; where ghi_cs is 0, kt is undefined or 1 if radiation_w_m2 is also 0.
            kt = (df_synthesized["radiation_w_m2"] / ghi_cs.where(ghi_cs > 1e-3)).clip(0, 1)
            # Fill NaNs in kt (e.g., from missing radiation_w_m2 or ghi_cs=0) with 1.0 (clear sky) or a mean/median kt if preferred.
            # For simplicity, filling with 1.0 assumes periods of missing data were clear, or relies on DLI anchor.
            par_adj *= kt.fillna(1.0)
            status["cloud_correction"] = "empirical_kt_radiation_w_m2"
            logger.info(
                f"  (Era: {era_identifier}) Cloud correction applied using 'radiation_w_m2'. Mean kt: {kt.mean():.2f}"
            )
        else:
            status["cloud_correction"] = "skipped_no_radiation_w_m2_data"
            logger.info(
                f"  (Era: {era_identifier}) Cloud correction skipped: 'radiation_w_m2' not available or all NaNs."
            )
    else:
        status["cloud_correction"] = "disabled_by_user_flag"

    # Add final synthesized PAR column and status column
    df_synthesized["par_synth_umol"] = par_adj.astype("float32")
    df_synthesized["light_synthesis_status"] = json.dumps(status)

    logger.info(f"--- (Era: {era_identifier}) Light Synthesis Complete ---")
    return df_synthesized
