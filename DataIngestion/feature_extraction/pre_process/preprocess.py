#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "psycopg2-binary", "SQLAlchemy", "pyarrow", "fastparquet", "scikit-learn", "joblib", "ijson"]
# ///

import pandas as pd
import json
import os
import sys
from pathlib import Path
# Removed unused imports like re, BaseEstimator, TransformerMixin, MinMaxScaler, joblib, KNeighborsRegressor as they are in utils
from sqlalchemy import create_engine, text # Keep create_engine and text if used directly for engine creation
import subprocess
import traceback

# Internal utilities
from processing_steps import OutlierHandler, ImputationHandler, DataSegmenter
from database_operations import (
    fetch_source_data, 
    run_sql_script, 
    verify_table_exists, 
    save_to_timescaledb,
    fetch_and_prepare_external_weather_for_era,
    fetch_and_prepare_energy_prices_for_era
)
from data_preparation_utils import (
    load_config, 
    sort_and_prepare_df, 
    resample_data_for_era, 
    save_data, 
    generate_summary_report
)
from data_enrichment_utils import (
    load_literature_phenotypes,
    generate_proxy_phenotype_labels,
    calculate_daily_weekly_aggregates,
    EraFeatureGenerator, # This class is imported
    scale_data_for_era,
    synthesize_light_data # Added for light synthesis
)


# Define paths
CONFIG_PATH = os.getenv('APP_CONFIG_PATH', '/app/config/data_processing_config.json')
OUTPUT_DATA_DIR = Path(os.getenv('APP_OUTPUT_DIR', '/app/data/output'))
SCALERS_DIR = OUTPUT_DATA_DIR / "scalers" 
SUMMARY_REPORT_FILENAME_TEMPLATE = "preprocessing_summary_report_{era_identifier}.txt"
OUTPUT_FILENAME_TEMPLATE = "{era_identifier}_processed_segment_{segment_num}.parquet"
# LITERATURE_PHENOTYPES_TABLE_NAME is now in data_enrichment_utils

# Database connection details (defaults, might be overridden by config)
DB_USER_ENV = os.getenv("DB_USER", "postgres")
DB_PASSWORD_ENV = os.getenv("DB_PASSWORD", "postgres")
DB_HOST_ENV = os.getenv("DB_HOST", "db") 
DB_PORT_ENV = os.getenv("DB_PORT", "5432")
DB_NAME_ENV = os.getenv("DB_NAME", "postgres")


# Start of the main execution block
if __name__ == "__main__":
    print("--- Starting Preprocessing Stage --- ")
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    current_script_dir = Path(__file__).parent
    fetch_weather_script_path = current_script_dir / "fetch_external_weather.py"
    fetch_energy_script_path = current_script_dir / "fetch_energy.py"
    phenotype_ingest_script_path = current_script_dir / "phenotype_ingest.py"
    sql_script_path = current_script_dir / "create_preprocessed_hypertable.sql" 

    scripts_to_run = {
        "External Weather Data Fetch": fetch_weather_script_path,
        "Energy Price Data Fetch": fetch_energy_script_path,
        "Phenotype Data Ingest": phenotype_ingest_script_path,
    }

    app_config_for_subprocess_env = {}
    try:
        with open(CONFIG_PATH, 'r') as f_temp_cfg:
            app_config_for_subprocess_env = json.load(f_temp_cfg)
    except Exception as e_cfg_sub:
        print(f"Warning: Could not load config for subprocess env setup: {e_cfg_sub}")

    subprocess_env = None
    if app_config_for_subprocess_env:
        db_conn_settings_for_subprocess = app_config_for_subprocess_env.get("database_connection", {})
        subprocess_env = os.environ.copy()
        subprocess_env["DB_USER"] = db_conn_settings_for_subprocess.get("user", DB_USER_ENV)
        subprocess_env["DB_PASSWORD"] = db_conn_settings_for_subprocess.get("password", DB_PASSWORD_ENV)
        subprocess_env["DB_HOST"] = db_conn_settings_for_subprocess.get("host", DB_HOST_ENV)
        subprocess_env["DB_PORT"] = str(db_conn_settings_for_subprocess.get("port", DB_PORT_ENV))
        subprocess_env["DB_NAME"] = db_conn_settings_for_subprocess.get("dbname", DB_NAME_ENV)
    else:
        print("Critical Error: app_config_for_subprocess_env is empty. Subprocess env not set.")


    if subprocess_env:
        for script_name, script_path_val in scripts_to_run.items():
            print(f"\n--- Attempting to run: {script_name} from {script_path_val} ---")
            if script_path_val.exists():
                try:
                    process = subprocess.run(
                        [sys.executable, "-m", "uv", "run", "--isolated", str(script_path_val)],
                        capture_output=True, text=True, check=False, 
                        cwd=current_script_dir, env=subprocess_env
                    )
                    print(f"Output from {script_name}:\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")
                    if process.returncode != 0:
                        print(f"Warning: {script_name} exited with code {process.returncode}")
                except Exception as e:
                    print(f"An unexpected error occurred while trying to run {script_path_val}: {e}")
                    traceback.print_exc()
            else:
                print(f"Warning: Script not found at {script_path_val}. Skipping execution of {script_name}.")
    else:
        print("Warning: Subprocess environment not set up due to config load issue. Skipping external script runs.")

    try:
        app_config = load_config(Path(CONFIG_PATH)) 
    except Exception as e: 
        print(f"Failed to load main application config: {e}. Exiting.")
        sys.exit(1)

    db_conn_settings = app_config.get("database_connection", {})
    db_user = db_conn_settings.get("user", DB_USER_ENV)
    db_password = db_conn_settings.get("password", DB_PASSWORD_ENV)
    db_host = db_conn_settings.get("host", DB_HOST_ENV)
    db_port = db_conn_settings.get("port", DB_PORT_ENV)
    db_name = db_conn_settings.get("dbname", DB_NAME_ENV)
    
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = None
    try:
        engine = create_engine(db_url, pool_size=5, max_overflow=10, pool_timeout=30, pool_recycle=1800)
        if not run_sql_script(engine, sql_script_path):
             print(f"Warning: SQL script execution failed or script not found: {sql_script_path}")
        if not verify_table_exists(engine, "preprocessed_features"):
             print("Warning: Table 'preprocessed_features' does not exist after script execution attempt.")
        
        literature_phenotypes_df = load_literature_phenotypes(engine)

    except Exception as e:
        print(f"Error during DB setup or literature phenotype loading: {e}")
        traceback.print_exc()
        literature_phenotypes_df = pd.DataFrame() 
        if engine is None: 
            print("Critical: Could not establish database engine. Exiting.")
            sys.exit(1)

    extra_cfg_path_str = os.getenv("EXTRA_ERA_CONFIG")
    if extra_cfg_path_str:
        extra_cfg_path = Path(extra_cfg_path_str)
        if extra_cfg_path.exists():
            print(f"Found EXTRA_ERA_CONFIG: {extra_cfg_path}. Attempting to merge auto era definitions.")
            try:
                with open(extra_cfg_path, 'r') as f_extra_cfg:
                    extra_cfg_data = json.load(f_extra_cfg)
                auto_era_definitions = extra_cfg_data.get("auto_era_definitions")
                if auto_era_definitions and isinstance(auto_era_definitions, dict):
                    if 'era_definitions' not in app_config: app_config['era_definitions'] = {}
                    elif not isinstance(app_config['era_definitions'], dict):
                        print("Warning: 'era_definitions' in main config is not a dict, overwriting with empty dict before merge.")
                        app_config['era_definitions'] = {}
                    app_config["era_definitions"].update(auto_era_definitions)
                    print(f"Successfully merged {len(auto_era_definitions)} auto-detected eras.")
                else:
                    print(f"Warning: 'auto_era_definitions' key not found or not a dict in {extra_cfg_path}.")
            except Exception as e_merge:
                print(f"Error merging auto era config from {extra_cfg_path}: {e_merge}.")
        else:
            print(f"Warning: EXTRA_ERA_CONFIG path {extra_cfg_path} does not exist.")
    else:
        print("INFO: EXTRA_ERA_CONFIG environment variable not set. No auto eras to merge.")

    eras_to_process_config = app_config.get('era_definitions', {})
    if not eras_to_process_config:
        print("No 'era_definitions' found in config after potential merge. Exiting.")
        sys.exit(1)

    single_era_env = os.getenv("PROCESS_ERA_IDENTIFIER")
    eras_to_run_keys = []
    if single_era_env:
        if single_era_env in eras_to_process_config:
            eras_to_run_keys = [single_era_env]
            print(f"PROCESS_ERA_IDENTIFIER is set to '{single_era_env}'. Processing only this era.")
        else:
            print(f"Warning: PROCESS_ERA_IDENTIFIER '{single_era_env}' not found in era_definitions. Processing all defined eras.")
            eras_to_run_keys = list(eras_to_process_config.keys())
    else:
        eras_to_run_keys = list(eras_to_process_config.keys())
        print("PROCESS_ERA_IDENTIFIER not set. Processing all defined eras.")
    
    print(f"Eras to be processed: {eras_to_run_keys}")

    for era_id in eras_to_run_keys:
        era_conf_details = eras_to_process_config.get(era_id)
        if not era_conf_details: 
            print(f"Critical Error: Era configuration for '{era_id}' not found. Skipping.")
            continue
            
        print(f"\n===== PROCESSING ERA: {era_id} =====")
        current_era_summary_items = [("Era Identifier", era_id), 
                                     ("Era Configuration Snapshot", json.dumps(era_conf_details, indent=2))]
        
        source_df = fetch_source_data(era_identifier=era_id, era_config=era_conf_details, global_config=app_config, engine=engine)
        current_era_summary_items.append(("Source Data Shape", source_df.shape))
        if source_df.empty:
            print(f"No source data fetched for Era '{era_id}'. Skipping further processing for this era.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue

        prepared_df = sort_and_prepare_df(source_df, app_config, era_identifier=era_id)
        current_era_summary_items.append(("Prepared Data Shape (after sort, ID, bool_to_int)", prepared_df.shape))
        if prepared_df.empty: 
            print(f"DataFrame became empty after initial preparation for Era '{era_id}'. Skipping.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue
        
        time_col_name_common = app_config.get('common_settings',{}).get('time_col', 'time')
        if time_col_name_common in prepared_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(prepared_df[time_col_name_common]):
                 prepared_df[time_col_name_common] = pd.to_datetime(prepared_df[time_col_name_common], errors='coerce', utc=True)
                 prepared_df.dropna(subset=[time_col_name_common], inplace=True) 
            if not prepared_df.empty:
                prepared_df = prepared_df.set_index(time_col_name_common)
        elif prepared_df.index.name != time_col_name_common:
            print(f"Warning: Time column '{time_col_name_common}' not found as column or index in prepared_df for Era '{era_id}'. Subsequent steps might fail.")
        
        if prepared_df.empty: 
            print(f"DataFrame became empty after setting index for Era '{era_id}'. Skipping.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue

        df_with_weather = prepared_df.copy() 
        common_settings_cfg = app_config.get('common_settings', {})
        era_target_freq = era_conf_details.get("target_frequency", common_settings_cfg.get("default_target_frequency", "15T"))

        # Load light synthesis configurations globally for the run
        enable_light_synthesis_flag = common_settings_cfg.get('enable_light_synthesis', False)
        light_synthesis_columns_config = common_settings_cfg.get('light_synthesis_columns', [])
        synth_latitude = common_settings_cfg.get('latitude', 56.2661)  # Default Hinnerup
        synth_longitude = common_settings_cfg.get('longitude', 10.064) # Default Hinnerup
        synth_use_cloud_correction = common_settings_cfg.get('use_cloud_correction', True)
        synth_dli_clip_min = common_settings_cfg.get('dli_scale_clip_min', 0.25)
        synth_dli_clip_max = common_settings_cfg.get('dli_scale_clip_max', 4.0)

        if common_settings_cfg.get("include_external_weather", False):
            print("Attempting to fetch and merge external weather data...")
            era_start_date_str = era_conf_details.get("start_date")
            era_end_date_str = era_conf_details.get("end_date")
            
            if era_start_date_str and era_end_date_str:
                external_weather_df = fetch_and_prepare_external_weather_for_era(
                    era_start_date_str, era_end_date_str, 
                    era_target_freq, time_col_name_common, engine
                )
                if not external_weather_df.empty:
                    if df_with_weather.index.tz is None: df_with_weather.index = df_with_weather.index.tz_localize('UTC')
                    if external_weather_df.index.tz is None: external_weather_df.index = external_weather_df.index.tz_localize('UTC')
                    
                    df_with_weather = df_with_weather.join(external_weather_df, how='left', rsuffix='_ext_weather')
                    print(f"Shape after attempting to join external weather: {df_with_weather.shape}")
                    current_era_summary_items.append(("External Weather Data Merged", True if not external_weather_df.empty else False))
                    current_era_summary_items.append(("Shape after weather merge", df_with_weather.shape))
                else:
                    print("No external weather data fetched or prepared. Proceeding without it.")
                    current_era_summary_items.append(("External Weather Data Merged", False))
            else:
                print("Start/End date for era not found, cannot fetch external weather.")
                current_era_summary_items.append(("External Weather Data Fetch Skipped", "Missing era start/end dates"))
        else:
            print("Skipping external weather data based on config.")
            current_era_summary_items.append(("External Weather Data Merged", "Skipped by config"))

        df_final_merged = df_with_weather.copy() 
        if common_settings_cfg.get("include_energy_prices", False):
            print("Attempting to fetch and merge external energy prices...")
            era_start_date_str = era_conf_details.get("start_date")
            era_end_date_str = era_conf_details.get("end_date")

            if era_start_date_str and era_end_date_str:
                energy_prices_df = fetch_and_prepare_energy_prices_for_era(
                    era_start_date_str, era_end_date_str,
                    era_target_freq, common_settings_cfg, engine
                )
                if not energy_prices_df.empty:
                    if df_final_merged.index.tz is None: df_final_merged.index = df_final_merged.index.tz_localize('UTC')
                    if energy_prices_df.index.tz is None: energy_prices_df.index = energy_prices_df.index.tz_localize('UTC')

                    df_final_merged = df_final_merged.join(energy_prices_df, how='left', rsuffix='_energy_price')
                    output_spot_price_col_name = "spot_price_dkk_mwh" 
                    
                    cols_to_ffill_energy = [col for col in df_final_merged.columns if output_spot_price_col_name in col]
                    if cols_to_ffill_energy:
                        print(f"Forward filling merged energy price columns: {cols_to_ffill_energy}")
                        df_final_merged[cols_to_ffill_energy] = df_final_merged[cols_to_ffill_energy].ffill()
                    
                    print(f"Shape after attempting to join energy prices: {df_final_merged.shape}")
                    current_era_summary_items.append(("Energy Price Data Merged", True if not energy_prices_df.empty else False))
                    current_era_summary_items.append(("Shape after energy price merge", df_final_merged.shape))
                else:
                    print("No energy price data fetched or prepared. Proceeding without it.")
                    current_era_summary_items.append(("Energy Price Data Merged", False))
            else:
                print("Start/End date for era not found, cannot fetch energy prices.")
                current_era_summary_items.append(("Energy Price Data Fetch Skipped", "Missing era start/end dates"))
        else:
            print("Skipping energy price data based on config.")
            current_era_summary_items.append(("Energy Price Data Merged", "Skipped by config"))
        
        df_resampled = resample_data_for_era(df_final_merged, era_id, era_conf_details, common_settings_cfg)
        current_era_summary_items.append(("Resampled Data Shape", df_resampled.shape))
        if df_resampled.empty:
            print(f"DataFrame became empty after resampling for Era '{era_id}'. Skipping.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue
        
        df_enriched = df_resampled.copy()
        
        # Light Synthesis Block
        if common_settings_cfg.get("enable_light_synthesis", False):
            print(f"\n--- Synthesizing Light Data for Era '{era_id}' ---")
            light_synth_lat = common_settings_cfg.get("latitude", 56.2661)
            light_synth_lon = common_settings_cfg.get("longitude", 10.064)
            light_synth_use_cloud = common_settings_cfg.get("use_cloud_correction", True)
            light_synth_dli_clip_min = common_settings_cfg.get("dli_scale_clip_min", 0.25)
            light_synth_dli_clip_max = common_settings_cfg.get("dli_scale_clip_max", 4.0)
            
            if df_enriched.empty:
                print(f"  Skipping light synthesis for Era '{era_id}': input DataFrame is empty before synthesis.")
                current_era_summary_items.append(("Light Synthesis Attempted", "Skipped, empty input DF"))
            else:
                try:
                    print(f"  Input shape for light synthesis: {df_enriched.shape}")
                    df_synthesized = synthesize_light_data(
                        df=df_enriched.copy(), # Use .copy() for safety
                        lat=light_synth_lat,
                        lon=light_synth_lon,
                        use_cloud=light_synth_use_cloud,
                        era_identifier=era_id,
                        dli_scale_clip_min=light_synth_dli_clip_min,
                        dli_scale_clip_max=light_synth_dli_clip_max
                    )
                    
                    if df_synthesized is not None and not df_synthesized.empty:
                        df_enriched = df_synthesized
                        print(f"  Successfully synthesized light data. DataFrame shape after synthesis: {df_enriched.shape}")
                        current_era_summary_items.append(("Light Synthesis Status", "Success"))
                        if 'par_synth_umol' in df_enriched.columns:
                             current_era_summary_items.append(("Synthetic PAR Column", "par_synth_umol added"))
                        else:
                             current_era_summary_items.append(("Synthetic PAR Column", "par_synth_umol NOT added"))
                    else:
                        print(f"  Light synthesis for Era '{era_id}' resulted in an empty or None DataFrame. Original data retained.")
                        current_era_summary_items.append(("Light Synthesis Status", "Failed - empty/None result"))
                except Exception as e:
                    print(f"  Error during light synthesis for Era '{era_id}': {e}")
                    traceback.print_exc()
                    current_era_summary_items.append(("Light Synthesis Status", f"Failed - Exception: {type(e).__name__}"))
        else:
            print(f"\n--- Skipping Light Synthesis (disabled in config) ---")
            current_era_summary_items.append(("Light Synthesis Attempted", "Skipped by config"))
        # End Light Synthesis Block

        proxy_phenotype_config = app_config.get('proxy_phenotypes')
        if proxy_phenotype_config and not literature_phenotypes_df.empty:
            print(f"\n--- Generating Proxy Phenotype Labels ---")
            for pheno_target_info in proxy_phenotype_config.get('targets', []):
                lit_features = pheno_target_info.get('literature_model_features')
                gh_features = pheno_target_info.get('greenhouse_data_features')
                if lit_features and gh_features:
                    # Erroneous light synthesis logic previously here (lines 317-338) has been removed.
                    # The actual light synthesis will be inserted before this proxy phenotype block.
                    proxy_series = generate_proxy_phenotype_labels(
                    era_aggregated_gh_data=df_enriched, 
                    literature_phenotypes_df=literature_phenotypes_df,
                    target_phenotype_info=pheno_target_info,
                    literature_model_features=lit_features,
                    greenhouse_data_features=gh_features
                )
                if not proxy_series.empty:
                    df_enriched[proxy_series.name] = proxy_series
                    current_era_summary_items.append((f"Proxy Phenotype Added: {proxy_series.name}", True))
                    print(f"  Added proxy phenotype: {proxy_series.name}")
                else:
                    current_era_summary_items.append((f"Proxy Phenotype Skipped: {pheno_target_info.get('name')}", "No data or error"))
            current_era_summary_items.append(("Shape after proxy phenotypes", df_enriched.shape))
        else:
            print(f"\n--- Skipping Proxy Phenotype Generation (no config or no literature data) ---")

        if common_settings_cfg.get("calculate_time_aggregates", False):
            print(f"\n--- Calculating Daily and Weekly Aggregates ---")
            df_daily_agg, df_weekly_agg = calculate_daily_weekly_aggregates(df_enriched, era_id) # This function is now imported
            if not df_daily_agg.empty:
                df_enriched = df_enriched.join(df_daily_agg, how='left') 
                df_enriched[df_daily_agg.columns] = df_enriched[df_daily_agg.columns].ffill() 
                current_era_summary_items.append(("Daily Aggregates Added", True))
            if not df_weekly_agg.empty:
                df_enriched = df_enriched.join(df_weekly_agg, how='left') 
                df_enriched[df_weekly_agg.columns] = df_enriched[df_weekly_agg.columns].ffill() 
                current_era_summary_items.append(("Weekly Aggregates Added", True))
            current_era_summary_items.append(("Shape after time aggregates", df_enriched.shape))
        else:
            print(f"\n--- Skipping Daily/Weekly Aggregate Calculation (config) ---")
        
        df_for_further_processing = df_enriched.copy()

        era_outlier_rules_ref = era_conf_details.get('outlier_rules_ref', 'default_outlier_rules')
        outlier_rules = app_config.get('preprocessing_rules', {}).get(era_outlier_rules_ref, [])
        outlier_handler = OutlierHandler(outlier_rules, rules_cfg_dict=common_settings_cfg.get('outlier_configs', {}))
        df_after_outliers = outlier_handler.clip_outliers(df_for_further_processing)
        current_era_summary_items.append(("Outlier Clipped Data Shape", df_after_outliers.shape))

        segmenter = DataSegmenter(era_conf_details, common_config=app_config.get('common_settings',{}))
        if df_after_outliers.index.name != time_col_name_common and time_col_name_common in df_after_outliers.columns:
             if not pd.api.types.is_datetime64_any_dtype(df_after_outliers[time_col_name_common]):
                 df_after_outliers[time_col_name_common] = pd.to_datetime(df_after_outliers[time_col_name_common], errors='coerce', utc=True)
                 df_after_outliers.dropna(subset=[time_col_name_common], inplace=True)
             if not df_after_outliers.empty:
                df_after_outliers = df_after_outliers.set_index(time_col_name_common)
        elif not isinstance(df_after_outliers.index, pd.DatetimeIndex) and df_after_outliers.index.name == time_col_name_common:
            df_after_outliers.index = pd.to_datetime(df_after_outliers.index, utc=True)
        
        if df_after_outliers.empty:
            print(f"DataFrame became empty before segmentation for Era '{era_id}'. Skipping.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue
            
        data_segments = segmenter.segment_by_availability(df_after_outliers)
        current_era_summary_items.append(("Number of Segments Found", len(data_segments)))

        if not data_segments:
            print(f"No data segments found for Era '{era_id}'. Skipping further processing for this era.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue
        
        processed_segment_paths_era = []
        for i, segment_df_original in enumerate(data_segments):
            segment_label = f"Era {era_id} - Segment {i+1}/{len(data_segments)}"
            print(f"\n--- Processing {segment_label} ---")
            current_era_summary_items.append((f"Segment {i+1} Initial Shape", segment_df_original.shape))
            if segment_df_original.empty:
                print(f"{segment_label} is empty. Skipping.")
                current_era_summary_items.append((f"Segment {i+1} Status", "Empty, Skipped"))
                continue

            era_imputation_rules_ref = era_conf_details.get('imputation_rules_ref', 'default_imputation_rules')
            imputation_rules = app_config.get('preprocessing_rules', {}).get(era_imputation_rules_ref, [])
            imputation_handler = ImputationHandler(imputation_rules)
            df_after_imputation = imputation_handler.impute_data(segment_df_original)
            current_era_summary_items.append((f"Segment {i+1} Shape after Imputation", df_after_imputation.shape))
            
            df_with_lamp_features = df_after_imputation.copy()
            lamp_cols = [col for col in df_with_lamp_features.columns if 'lamp' in col.lower() and 'status' in col.lower()]
            if lamp_cols:
                try:
                    for lamp_col in lamp_cols:
                        df_with_lamp_features[lamp_col] = pd.to_numeric(df_with_lamp_features[lamp_col], errors='coerce').fillna(0)
                    
                    df_with_lamp_features['total_lamps_on'] = df_with_lamp_features[lamp_cols].sum(axis=1)
                    print(f"  Calculated 'total_lamps_on' from columns: {lamp_cols}")
                    current_era_summary_items.append((f"Segment {i+1} Total Lamps Feature Added", True))
                except Exception as e_lamp:
                    print(f"  Error calculating total_lamps_on: {e_lamp}")
                    current_era_summary_items.append((f"Segment {i+1} Total Lamps Feature Error", str(e_lamp)))
            else:
                 current_era_summary_items.append((f"Segment {i+1} Total Lamps Feature Added", "No lamp status cols found"))

            df_processed_segment = df_with_lamp_features 
            skip_era_feature_env = os.getenv("SKIP_ERA_FEATURE", "false").lower()
            should_skip_era_feature_generation = skip_era_feature_env == "true" or skip_era_feature_env == "1"

            if not should_skip_era_feature_generation:
                default_era_c_label_file = f"{era_id}_era_labels_levelC.parquet"
                era_feature_file_name = era_conf_details.get("era_feature_file", 
                                                       app_config.get('common_settings',{}).get("era_feature_file", default_era_c_label_file))
                
                era_feature_file_path = OUTPUT_DATA_DIR / era_feature_file_name
                
                print(f"Attempting to add era features using file: {era_feature_file_path} for {segment_label}")
                era_feature_generator = EraFeatureGenerator(era_file_path=era_feature_file_path)
                df_temp_for_era_feat = df_processed_segment.copy() 
                df_processed_segment = era_feature_generator.transform(df_temp_for_era_feat)
                
                shape_change_info = f"Shape before EraFeat: {df_temp_for_era_feat.shape}, After: {df_processed_segment.shape}"
                current_era_summary_items.append((f"Segment {i+1} Era Feature Gen", shape_change_info))
            else:
                skip_info = f"SKIP_ERA_FEATURE is set. Skipped for {segment_label}."
                print(skip_info)
                current_era_summary_items.append((f"Segment {i+1} Era Feature Gen", skip_info))

            df_scaled, _ = scale_data_for_era(df_processed_segment, era_id, era_conf_details, app_config, SCALERS_DIR, fit_scalers=True)
            current_era_summary_items.append((f"Segment {i+1} Shape after Scaling", df_scaled.shape))
            
            df_to_save = df_scaled.reset_index() if isinstance(df_scaled.index, pd.DatetimeIndex) else df_scaled
            
            output_filename = OUTPUT_FILENAME_TEMPLATE.format(era_identifier=era_id, segment_num=i+1)
            output_path = OUTPUT_DATA_DIR / output_filename 
            save_data(df_to_save, output_path) 
            processed_segment_paths_era.append(str(output_path))
            current_era_summary_items.append((f"Segment {i+1} Saved Path", str(output_path)))
            
            if engine and verify_table_exists(engine, "preprocessed_features"):
                time_col_for_saving = app_config.get('common_settings',{}).get('time_col', 'time')
                save_to_timescaledb(df=df_scaled, era_identifier=era_id, engine=engine, time_col=time_col_for_saving)
        
        current_era_summary_items.append(("Processed Segment Paths", processed_segment_paths_era))
        generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))

    if engine:
        engine.dispose()
        print(f"\nDatabase engine disposed.")
    print("\n===== ALL SPECIFIED ERAS PROCESSED =====")

