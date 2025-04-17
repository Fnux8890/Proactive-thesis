import os
import json
import polars as pl
import asyncpg
import pyarrow as pa
import logging
from prefect import flow, task
from prefect.logging.loggers import get_run_logger as prefect_get_run_logger
from prefect.exceptions import MissingContextError
from pathlib import Path
from datetime import datetime, timedelta, date
import great_expectations as gx

from config import load_config, PlantConfig, OptimalRange

def get_logger() -> logging.Logger:
    """Return a Prefect run logger if in context, else a standard Python logger."""
    try:
        return prefect_get_run_logger()
    except MissingContextError:
        return logging.getLogger("pipeline")

# --- Data Access Object ---
class SensorRepository:
    def __init__(self, db_url: str):
        self._pool = None
        self.db_url = db_url

    async def __aenter__(self):
        logger = get_logger()
        logger.info(f"Connecting to database...")
        try:
            self._pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=5)
            logger.info("Database pool created.")
            # Test connection
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            logger.info("Database connection successful.")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._pool:
            await self._pool.close()
            get_logger().info("Database pool closed.")

    async def get_sensor_data(self, start_time: datetime, end_time: datetime) -> pa.Table:
        logger = get_logger()
        query = """
        SELECT * FROM sensor_data
        WHERE time >= $1 AND time < $2
        ORDER BY time;
        """
        logger.info(f"Fetching data from {start_time} to {end_time}")
        try:
            async with self._pool.acquire() as conn:
                # Using fetch for simplicity, consider copy_to_arrow for large data
                records = await conn.fetch(query, start_time, end_time)
                if not records:
                    logger.warning("No data found for the specified time range.")
                    return None

                # Manually construct pyarrow Table
                data_dict = {key: [r[key] for r in records] for key in records[0].keys()}
                # Explicitly handle potential None values if necessary column by column
                # For simplicity, assuming basic types are handled by pyarrow
                arrow_table = pa.table(data_dict)
                logger.info(f"Fetched {len(records)} records.")
                return arrow_table
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

# --- Feature Engineering Task ---
@task
def transform_features(df: pl.DataFrame, config: PlantConfig) -> pl.DataFrame:
    logger = get_logger()
    logger.info(f"Starting feature transformation on DataFrame with shape {df.shape}")

    if df.is_empty():
        logger.warning("Input DataFrame is empty, skipping transformations.")
        return df

    # --- Pre-checks and Setup ---
    required_cols = {"time", "air_temp_c", "relative_humidity_percent"}
    # Add required columns based on config features
    if config.dif_parameters.day_definition == "lamp_status":
        required_cols.update(config.dif_parameters.lamp_status_columns)
    par_col = "light_par_umol_m2_s" # Standardized PAR column name
    if par_col in config.feature_parameters.rolling_average_cols or par_col in config.feature_parameters.rate_of_change_cols: # Check if DLI calc needed indirectly
         required_cols.add(par_col)

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns for transformation: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure time column is datetime and sort
    df = df.with_columns(pl.col("time").cast(pl.Datetime)).sort("time")
    df = df.with_columns(pl.col("time").dt.date().alias("date")) # Add date column for joins

    # --- Feature Calculations ---

    # 1. VPD (Vapor Pressure Deficit)
    # Unit Test Target: (25 °C, 60 % RH ⇒ 1.26 kPa)
    logger.info("Calculating VPD (Buck's 1981)...")
    T = pl.col("air_temp_c")
    RH = pl.col("relative_humidity_percent")
    # Saturation Vapor Pressure (svp) in kPa using Buck's formula
    # Note: Use the expression method .exp() for exponential
    svp_expr = 0.61121 * ((17.502 * T) / (T + 240.97)).exp()
    # Actual Vapor Pressure (avp) in kPa
    avp_expr = svp_expr * (RH / 100.0)
    df = df.with_columns((svp_expr - avp_expr).alias("vpd_kpa"))

    # 2. DLI (Daily Light Integral)
    # Unit Test Target: 1hr constant 500 µmol -> 1.8 mol/m²/d
    logger.info("Calculating DLI...")
    if par_col in df.columns:
        dli_daily = (
            df.select(["time", "date", par_col]) # Select time, date, and PAR column
            .sort("time")
            .with_columns(
                # Calculate time delta in seconds, fill first null with 0
                pl.col("time").diff().dt.total_seconds().fill_null(0).alias("delta_t_s")
            )
            .with_columns(
                # Calculate moles for the interval (PAR * delta_t / 1e6)
                (pl.col(par_col) * pl.col("delta_t_s") / 1_000_000).alias("mol_chunk")
            )
            .group_by_dynamic(
                index_column="time",
                every="1d",
                closed="left", # Include start, exclude end of the day interval
                group_by="date" # Group by precomputed date for joining
            )
            .agg(
                pl.sum("mol_chunk").alias("DLI_mol_m2_d")
            )
            .select(["date", "DLI_mol_m2_d"]) # Keep only date and DLI for join
        )
        # Join daily DLI back to the main dataframe
        df = df.join(dli_daily, on="date", how="left")
    else:
        logger.warning(f"Column '{par_col}' not found, skipping DLI calculation.")

    # 3. GDD (Growing Degree Days - Daily & Cumulative)
    # Unit Test Target: Day (mean=20C, base=10C) -> 14 GDD (based on config)
    logger.info("Calculating GDD...")
    gdd_profile = config.gdd_parameters.profiles[config.gdd_parameters.crop_profile]
    t_base = gdd_profile.t_base_celsius
    t_cap = gdd_profile.t_cap_celsius
    df = df.with_columns(
        pl.when(T > t_base)
        .then(
            pl.min_horizontal(T, pl.lit(t_cap)) # Cap temperature
            - t_base
        )
        .otherwise(0)
        .alias("gdd_increment")
    )
    # Calculate daily GDD sum
    gdd_daily_sum = (
        df.group_by("date", maintain_order=True)
        .agg(pl.sum("gdd_increment").alias("GDD_daily"))
    )
    # Join daily GDD sum back
    df = df.join(gdd_daily_sum, on="date", how="left")
    # Calculate cumulative GDD
    df = df.with_columns(
        pl.col("GDD_daily").cum_sum().over("date").alias("GDD_cumulative_within_day")
        # For true cumulative GDD across days, this needs state or reading previous day's value
    )
    logger.warning("GDD_cumulative calculated within day; true cumulative needs historical context.")

    # 4. DIF (Day-minus-Night Temperature) - Explicit Calculation
    # Unit Test Target: Day=25C, Night=18C -> +7C
    logger.info("Calculating DIF (Explicit Method)...")
    if config.dif_parameters.day_definition == "lamp_status":
        lamp_cols = config.dif_parameters.lamp_status_columns
        if not lamp_cols:
            logger.warning("DIF calc by lamp_status specified, but no lamp_status_columns defined.")
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("DIF_daily")) # Add null column
        else:
            # Create masks
            day_mask_expr = pl.any_horizontal([pl.col(c) == 1 for c in lamp_cols])
            night_mask_expr = ~day_mask_expr

            # Calculate daily mean day temperature
            day_temps_daily = (
                df.filter(day_mask_expr)
                .group_by("date", maintain_order=True)
                .agg(pl.mean("air_temp_c").alias("_dayT_mean"))
            )

            # Calculate daily mean night temperature
            night_temps_daily = (
                df.filter(night_mask_expr)
                .group_by("date", maintain_order=True)
                .agg(pl.mean("air_temp_c").alias("_nightT_mean"))
            )

            # Join day and night means together
            # Use outer_coalesce if polars >= 0.20.11, otherwise use outer and fillna
            try:
                # Try outer_coalesce first
                dif_stats_daily = day_temps_daily.join(
                    night_temps_daily, on="date", how="outer_coalesce"
                )
            except Exception: # Broad exception for potential older polars version
                logger.warning("outer_coalesce join failed, falling back to outer join for DIF stats.")
                dif_stats_daily = day_temps_daily.join(
                    night_temps_daily, on="date", how="outer"
                )

            # Calculate DIF daily, filling missing day/night means with 0 before subtraction
            dif_stats_daily = dif_stats_daily.with_columns(
                (pl.col("_dayT_mean").fill_null(0.0) - pl.col("_nightT_mean").fill_null(0.0)).alias("DIF_daily")
            ).select(["date", "DIF_daily"]) # Keep only date and final DIF

            # Join the calculated daily DIF back to the main DataFrame
            df = df.join(dif_stats_daily, on="date", how="left")

    else:
        logger.warning(f"DIF calculation method '{config.dif_parameters.day_definition}' not implemented.")
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("DIF_daily")) # Add null column

    # 5. Advanced Flags
    logger.info("Calculating advanced flags...")
    # Rolling SD / Lag (Using duration syntax)
    if config.advanced_feature_parameters.rolling_std_dev_cols:
        for col, window_minutes_obj in config.advanced_feature_parameters.rolling_std_dev_cols.items():
            # Skip non-integer window sizes (e.g., comments)
            if not isinstance(window_minutes_obj, int):
                logger.warning(f"Skipping rolling std dev for '{col}': window size '{window_minutes_obj}' is not an integer.")
                continue
            window_minutes = window_minutes_obj
            if col in df.columns:
                window_duration = f"{window_minutes}m"
                # Create a temporary rolling aggregation DataFrame
                try:
                    # Use general rolling for time-based std dev
                    rolling_agg_df = df.select(["time", col]) \
                                        .rolling(index_column="time", period=window_duration, closed="left") \
                                        .agg(pl.std(col).alias(f"{col}_rolling_std"))
                    # Join the result back (handle potential missing time col after agg)
                    # Need to check how `rolling().agg()` preserves index
                    # Safer join strategy might be needed
                    df = df.join(rolling_agg_df.select(["time", f"{col}_rolling_std"]), on="time", how="left") \
                          .rename({f"{col}_rolling_std": f"{col}_rolling_std_{window_minutes}m"})
                except Exception as e:
                     logger.error(f"Failed rolling std dev calculation for {col}: {e}")

    if config.advanced_feature_parameters.lag_features:
         for col, lag_minutes_obj in config.advanced_feature_parameters.lag_features.items():
            # Skip non-integer lag values
            if not isinstance(lag_minutes_obj, int):
                logger.warning(f"Skipping lag feature for '{col}': lag value '{lag_minutes_obj}' is not an integer.")
                continue
            lag_minutes = lag_minutes_obj
            if col in df.columns:
                # Polars lag is by row, not time duration directly. Needs resampling or window functions.
                # Placeholder: Implement time-based lag if needed.
                # df = df.with_columns(pl.col(col).shift(periods=lag_minutes * 60 // sample_interval_seconds).alias(f"{col}_lag_{lag_minutes}m"))
                logger.warning(f"Time-based lag for '{col}' requires resampling or more complex windowing; row-based shift not implemented.")

    # Distance-from-midpoint & in-range flag
    logger.info("Calculating distance-from-midpoint and in-range flags...")
    if config.advanced_feature_parameters.distance_from_optimal_midpoint:
        for target_key, input_col in config.advanced_feature_parameters.distance_from_optimal_midpoint.items():
            if not isinstance(input_col, str) or not input_col:
                 logger.warning(f"Skipping distance-from-midpoint for '{target_key}': invalid input column '{input_col}'.")
                 continue
            if input_col not in df.columns:
                 logger.warning(f"Skipping distance-from-midpoint for '{target_key}': input column '{input_col}' not found.")
                 continue

            # Find corresponding optimal range
            optimal_range = config.objective_function_parameters.optimal_ranges.get(target_key)
            if isinstance(optimal_range, OptimalRange):
                lower = optimal_range.lower
                upper = optimal_range.upper
                midpoint = (lower + upper) / 2.0
                df = df.with_columns(
                    (pl.col(input_col) - midpoint).abs().alias(f"{input_col}_dist_opt_mid")
                )
            else:
                logger.warning(f"Skipping distance-from-midpoint for '{target_key}': Optimal range not found or invalid in config.")

    if config.advanced_feature_parameters.in_optimal_range_flag:
        for target_key, input_col in config.advanced_feature_parameters.in_optimal_range_flag.items():
            if not isinstance(input_col, str) or not input_col:
                 logger.warning(f"Skipping in-optimal-range flag for '{target_key}': invalid input column '{input_col}'.")
                 continue
            if input_col not in df.columns:
                 logger.warning(f"Skipping in-optimal-range flag for '{target_key}': input column '{input_col}' not found.")
                 continue

            # Find corresponding optimal range
            optimal_range = config.objective_function_parameters.optimal_ranges.get(target_key)
            if isinstance(optimal_range, OptimalRange):
                lower = optimal_range.lower
                upper = optimal_range.upper
                df = df.with_columns(
                    pl.when((pl.col(input_col) >= lower) & (pl.col(input_col) <= upper))
                    .then(1)
                    .otherwise(0)
                    .alias(f"{input_col}_in_opt_range")
                )
            else:
                logger.warning(f"Skipping in-optimal-range flag for '{target_key}': Optimal range not found or invalid in config.")

    # Night-stress flag
    if config.dif_parameters.day_definition == "lamp_status" and config.advanced_feature_parameters.night_stress_flags:
        # Ensure night_mask_expr is defined for this scope
        lamp_cols = config.dif_parameters.lamp_status_columns
        night_mask_expr = None # Default to None
        if lamp_cols:
            day_mask_expr = pl.any_horizontal([pl.col(c) == 1 for c in lamp_cols])
            night_mask_expr = ~day_mask_expr # Define it here for use below

        if night_mask_expr is not None:
            logger.info("Calculating night stress flags...")
            for flag_name, details in config.advanced_feature_parameters.night_stress_flags.items():
                 if isinstance(details, dict):
                    # Assuming structure like: { "input_temp_col": "air_temp_c", "threshold_config_key": ..., "threshold_sub_key": ...}
                    input_col = details.get("input_temp_col", "air_temp_c")
                    thresh_key = details.get("threshold_config_key")
                    sub_key = details.get("threshold_sub_key")
                    try:
                        threshold_val = getattr(getattr(config.stress_thresholds, thresh_key), sub_key)
                        df = df.with_columns(
                            # Ensure night_mask_expr is valid before using
                            pl.when(night_mask_expr & (pl.col(input_col) > threshold_val))
                            .then(1)
                            .otherwise(0)
                            .alias(flag_name)
                        )
                    except AttributeError:
                         logger.warning(f"Could not find threshold '{thresh_key}.{sub_key}' in config for night stress flag '{flag_name}'.")
                    except Exception as e:
                         logger.warning(f"Error processing night stress flag '{flag_name}': {e}")
                 else:
                     logger.warning(f"Skipping night stress flag '{flag_name}': Invalid config format.")
        else:
            logger.warning("Cannot calculate night stress flags: Lamp status columns not defined.")

    # --- Final Cleanup --- (Optional)
    df = df.drop(["date"]) # Drop intermediate date col

    logger.info(f"Finished feature transformation. Final shape: {df.shape}")
    return df

# --- Validation Task ---
@task
def validate_data(df: pl.DataFrame) -> bool:
    """Validates the transformed DataFrame using Great Expectations."""
    logger = get_logger()
    logger.info("Starting data validation with Great Expectations...")
    if df.is_empty():
        logger.warning("DataFrame is empty, skipping validation.")
        return True # Or False? Decide based on requirements - returning True for now

    ge_root_dir = "/app/great_expectations"
    suite_name = "greenhouse_features"
    datasource_name = "runtime_pandas_datasource"
    data_asset_name = f"features_{datetime.utcnow().date().isoformat()}" # Unique asset name per run

    try:
        # 1. Initialize DataContext
        logger.info(f"Initializing Great Expectations DataContext at {ge_root_dir}")
        context = gx.get_context(context_root_dir=ge_root_dir)

        # 2. Ensure Runtime Datasource Exists
        try:
            context.get_datasource(datasource_name)
            logger.info(f"Using existing GE datasource: {datasource_name}")
        except gx.exceptions.DatasourceError:
            logger.info(f"Adding runtime datasource '{datasource_name}' to GE context")
            # Add a Runtime datasource configured for Pandas
            context.add_datasource(
                name=datasource_name,
                class_name="Datasource",
                execution_engine={"class_name": "PandasExecutionEngine"},
                data_connectors={
                    "runtime_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["run_id"], # Simple identifier
                    }
                },
            )
            logger.info(f"Added runtime datasource: {datasource_name}")

        # 3. Prepare Batch Request
        logger.info("Preparing GE Runtime Batch Request...")
        pandas_df = df.to_pandas() # Convert Polars to Pandas for validation
        batch_request = context.get_datasource(datasource_name).get_batch_request(
            batch_data=pandas_df,
            data_asset_name=data_asset_name, # Helps identify data in results
        )

        # 4. Create Checkpoint for validation
        logger.info(f"Running validation against suite: {suite_name}")
        checkpoint = context.add_or_update_checkpoint(
            name="runtime_feature_check",
            batch_request=batch_request,
            expectation_suite_name=suite_name,
        )

        # 5. Run Checkpoint
        # Result type is CheckpointResult, but we mainly need the success flag
        result = checkpoint.run()

        # 6. Check and Log Results
        success = bool(result.success)
        if success:
            logger.info("Great Expectations validation PASSED.")
        else:
            logger.error("Great Expectations validation FAILED.")
            # Log specific failure details
            try:
                stats = result.run_results[next(iter(result.run_results))]["validation_result"]["statistics"]
                logger.error(f"  Validation Stats: {stats}")
                failed_expectations = [res for res in result.list_validation_results() if not res["success"]] # GE >= 0.18 API
                # For older GE: failed_expectations = [r for r in results["results"] if not r["success"]]
                if failed_expectations:
                     logger.error("  Failed Expectations:")
                     for failure in failed_expectations:
                         exp = failure["expectation_config"]
                         details = failure["result"]
                         logger.error(f"    - {exp['expectation_type']}({exp['kwargs']}): {details.get('unexpected_list', details.get('partial_unexpected_list', details.get('observed_value', '')))}")
            except Exception as log_e:
                 logger.error(f"  Error logging GE failure details: {log_e}")

        # Optional: Save results (can fill up disk quickly)
        # context.save_expectation_suite(validator.get_expectation_suite(discard_failed_expectations=False))
        # context.build_data_docs()

        return success

    except Exception as e:
        logger.error(f"Error during Great Expectations validation: {e}", exc_info=True)
        return False # Fail flow if GE step encounters an error

# --- Loading Task ---
@task
def load_data(df: pl.DataFrame, run_date: date):
    # TODO: Add unit tests for load_data
    logger = get_logger()
    if df.is_empty():
        logger.warning("DataFrame is empty, skipping load step.")
        return

    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use date partitioning for parquet files
    partition_date_str = run_date.strftime("%Y-%m-%d")
    output_path = output_dir / f"features_{partition_date_str}.parquet"

    logger.info(f"Saving features to {output_path}")
    try:
        # Consider compression options (e.g., compression='zstd')
        df.write_parquet(output_path, use_pyarrow=True)
        logger.info("Features saved successfully.")
        # Optional: Register with Feast Feature Store here
        # TODO: Implement Feast registration if needed
    except Exception as e:
        logger.error(f"Failed to save features to {output_path}: {e}")
        raise

# --- Main Flow ---
@flow(log_prints=True)
async def main_feature_flow(run_date_str: str = "auto"):
    # TODO: Add integration tests for main_feature_flow
    logger = get_logger()
    logger.info("Starting main feature flow...")

    # Determine date range
    if run_date_str == "auto":
        # Default to yesterday for typical daily run
        run_date = (datetime.utcnow() - timedelta(days=1)).date()
        logger.info(f"'run_date_str' is auto, using yesterday's date: {run_date}")
    else:
        try:
            run_date = datetime.strptime(run_date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid date format '{run_date_str}'. Use YYYY-MM-DD.")
            return

    # Define time range for the target day
    start_datetime = datetime.combine(run_date, datetime.min.time())
    end_datetime = start_datetime + timedelta(days=1)

    # Load configuration
    try:
        config_path = "/app/plant_config.json"
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Get Database URL from environment variable
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set!")
        return

    # Execute pipeline within the DAO context
    raw_df_pl = None
    try:
        async with SensorRepository(db_url) as repo:
            arrow_table = await repo.get_sensor_data(start_datetime, end_datetime)
            if arrow_table:
                raw_df_pl = pl.from_arrow(arrow_table)
                logger.info("Successfully converted Arrow table to Polars DataFrame.")
            else:
                logger.warning("No data returned from repository for the period.")
                logger.info("Exiting flow due to no data.")
                return

    except Exception as e:
        logger.error(f"Error during data extraction: {e}")
        return # Exit flow on extraction error

    # Proceed only if data extraction was successful and yielded data
    if raw_df_pl is not None and not raw_df_pl.is_empty():
        try:
            # TODO: Add unit tests for transform_features
            transformed_df = transform_features(raw_df_pl, config)

            # TODO: Implement and run Great Expectations validation properly
            validation_passed = validate_data(transformed_df)

            if validation_passed:
                load_data(transformed_df, run_date)
                logger.info(f"Feature pipeline completed successfully for date: {run_date}.")
            else:
                logger.error(f"Data validation failed for date: {run_date}. Features not loaded.")
                # Potentially raise an error here to make the Prefect run fail
                # raise ValueError("Data validation failed!")

        except Exception as e:
            logger.error(f"Error during transform/validate/load for date {run_date}: {e}")
            # Potentially raise error to fail the flow
    else:
        # This case should ideally be caught by the earlier check after repo call
        logger.warning("Raw DataFrame is None or empty after extraction attempt. Skipping transform/validate/load.")

# Allow running the flow directly for testing
if __name__ == "__main__":
    import asyncio
    # Example run for yesterday
    # Ensure DATABASE_URL is set in your local environment if running this way
    # export DATABASE_URL='postgresql://user:pass@host:port/db'
    yesterday_str = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Running feature flow locally for date: {yesterday_str}")
    asyncio.run(main_feature_flow(run_date_str=yesterday_str))
