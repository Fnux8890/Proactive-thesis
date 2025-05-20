import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Any, Tuple, Dict, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
import traceback

LITERATURE_PHENOTYPES_TABLE_NAME = "literature_kalanchoe_phenotypes"

def load_literature_phenotypes(engine) -> pd.DataFrame:
    print(f"Loading literature Kalanchoe phenotypes from table: {LITERATURE_PHENOTYPES_TABLE_NAME}...")
    query = f"SELECT * FROM public.{LITERATURE_PHENOTYPES_TABLE_NAME} WHERE phenotype_value IS NOT NULL;"
    try:
        pheno_df = pd.read_sql_query(query, engine)
        print(f"DEBUG: Columns in pheno_df after read_sql_query: {pheno_df.columns.tolist()}") 
        print(f"DEBUG: pheno_df.info() after read_sql_query:") 
        pheno_df.info() 
        if pheno_df.empty:
            print("DEBUG: pheno_df is empty after read_sql_query (possibly due to 'WHERE phenotype_value IS NOT NULL' and no non-null values).")

        cols_to_convert = ['phenotype_value', 'environment_temp_day_c', 'environment_temp_night_c', 
                           'environment_photoperiod_h', 'environment_dli_mol_m2_d', 'environment_light_intensity_umol_m2_s',
                           'environment_co2_ppm', 'environment_rh_percent']
        
        valid_cols_to_convert = [col for col in cols_to_convert if col in pheno_df.columns]
        print(f"DEBUG: Valid columns for numeric conversion: {valid_cols_to_convert}") 

        for col in valid_cols_to_convert: 
            pheno_df[col] = pd.to_numeric(pheno_df[col], errors='coerce')
        
        essential_cols_for_proxy = ['phenotype_value', 'environment_dli_mol_m2_d', 
                                    'environment_temp_day_c', 'environment_photoperiod_h', 
                                    'species', 'phenotype_name', 'phenotype_unit']
        
        valid_essential_cols_for_proxy = [col for col in essential_cols_for_proxy if col in pheno_df.columns]
        print(f"DEBUG: Valid essential columns for dropna: {valid_essential_cols_for_proxy}") 
        
        if not valid_essential_cols_for_proxy:
            print(f"DEBUG: No essential columns for proxy model found in pheno_df. Columns available: {pheno_df.columns.tolist()}")
        
        if valid_essential_cols_for_proxy:
             pheno_df.dropna(subset=valid_essential_cols_for_proxy, inplace=True)
        else:
            print("DEBUG: Skipping dropna because no valid essential_cols_for_proxy were found in the DataFrame.")

        print(f"Loaded and cleaned {len(pheno_df)} rows from {LITERATURE_PHENOTYPES_TABLE_NAME}.")
        return pheno_df
    except Exception as e:
        print(f"Error loading literature Kalanchoe phenotypes: {e}")
        traceback.print_exc() 
        return pd.DataFrame()

def generate_proxy_phenotype_labels(
    era_aggregated_gh_data: pd.DataFrame,
    literature_phenotypes_df: pd.DataFrame,
    target_phenotype_info: Dict[str, str], 
    literature_model_features: List[str], 
    greenhouse_data_features: List[str]   
) -> pd.Series:

    phenotype_name = target_phenotype_info['name']
    phenotype_unit = target_phenotype_info['unit']
    target_species = target_phenotype_info['species']

    print(f"  Generating proxy labels for: {target_species} - {phenotype_name} ({phenotype_unit})")
    
    pheno_subset = literature_phenotypes_df[
        (literature_phenotypes_df['species'].str.contains(target_species, case=False, na=False, regex=False)) &
        (literature_phenotypes_df['phenotype_name'] == phenotype_name) &
        (literature_phenotypes_df['phenotype_unit'] == phenotype_unit)
    ]

    if pheno_subset.empty or pheno_subset[literature_model_features + ['phenotype_value']].isnull().all().any():
        print(f"    Not enough data or missing required columns in literature for {target_species}, {phenotype_name}. Required: {literature_model_features}. Available in pheno_subset: {pheno_subset.columns.tolist()}")
        return pd.Series(dtype='float64', index=era_aggregated_gh_data.index, name=f"{phenotype_name}_proxy_pred_error")

    pheno_subset = pheno_subset.dropna(subset=literature_model_features + ['phenotype_value'])
    if len(pheno_subset) < 3: 
        print(f"    Too few ({len(pheno_subset)}) valid data points in literature for {target_species}, {phenotype_name} to train proxy model. Skipping.")
        return pd.Series(dtype='float64', index=era_aggregated_gh_data.index, name=f"{phenotype_name}_proxy_pred_insufficient_data")

    X_lit = pheno_subset[literature_model_features]
    y_lit = pheno_subset['phenotype_value']

    model = KNeighborsRegressor(n_neighbors=min(3, len(X_lit)))
    try:
        model.fit(X_lit, y_lit)
    except Exception as e:
        print(f"    Error fitting proxy model for {phenotype_name}: {e}")
        return pd.Series(dtype='float64', index=era_aggregated_gh_data.index, name=f"{phenotype_name}_proxy_pred_fit_error")

    if not all(f in era_aggregated_gh_data.columns for f in greenhouse_data_features):
        print(f"    Missing one or more required greenhouse features for prediction: {greenhouse_data_features}. Available: {era_aggregated_gh_data.columns.tolist()}")
        return pd.Series(dtype='float64', index=era_aggregated_gh_data.index, name=f"{phenotype_name}_proxy_pred_missing_gh_feat")
        
    X_gh = era_aggregated_gh_data[greenhouse_data_features].copy()
    
    for i, lit_col_name in enumerate(literature_model_features):
        gh_col_name = greenhouse_data_features[i]
        if X_gh[gh_col_name].isnull().any():
            fill_value = X_lit[lit_col_name].mean()
            X_gh.loc[:, gh_col_name] = X_gh[gh_col_name].fillna(fill_value)
            print(f"    Filled NaNs in greenhouse feature '{gh_col_name}' with mean from literature training data ({fill_value:.2f}).")
    
    try:
        predicted_values = model.predict(X_gh)
        print(f"    Successfully generated proxy labels for {phenotype_name}.")
        return pd.Series(predicted_values, index=era_aggregated_gh_data.index, name=f"{phenotype_name}_proxy_pred")
    except Exception as e:
        print(f"    Error during prediction for {phenotype_name}: {e}")
        return pd.Series(dtype='float64', index=era_aggregated_gh_data.index, name=f"{phenotype_name}_proxy_pred_predict_error")

def calculate_daily_weekly_aggregates(df_era_processed: pd.DataFrame, era_identifier: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_era_processed.empty or not isinstance(df_era_processed.index, pd.DatetimeIndex):
        print(f"  (Era: {era_identifier}) DataFrame empty or not DatetimeIndexed, skipping daily/weekly aggregation.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Calculating daily and weekly aggregated features for Era '{era_identifier}'...")
    numeric_cols_for_agg = df_era_processed.select_dtypes(include=np.number).columns.tolist()
    
    daily_aggs_dict = {}
    weekly_aggs_dict = {}

    for col in numeric_cols_for_agg:
        daily_aggs_dict[f'{col}_daily_mean'] = (col, 'mean')
        daily_aggs_dict[f'{col}_daily_min'] = (col, 'min')
        daily_aggs_dict[f'{col}_daily_max'] = (col, 'max')
        daily_aggs_dict[f'{col}_daily_std'] = (col, 'std')

        weekly_aggs_dict[f'{col}_weekly_mean'] = (col, 'mean')
        weekly_aggs_dict[f'{col}_weekly_min'] = (col, 'min')
        weekly_aggs_dict[f'{col}_weekly_max'] = (col, 'max')
        weekly_aggs_dict[f'{col}_weekly_std'] = (col, 'std')

        if 'dli' in col.lower() or 'radiation' in col.lower() or 'light' in col.lower() or 'dosing_status' in col.lower():
            daily_aggs_dict[f'{col}_daily_sum'] = (col, 'sum')
            weekly_aggs_dict[f'{col}_weekly_sum'] = (col, 'sum')
    
    df_daily_features = pd.DataFrame()
    df_weekly_features = pd.DataFrame()

    if daily_aggs_dict:
        df_daily_features = df_era_processed.groupby(df_era_processed.index.normalize()).agg(
            **{k: v for k, v in daily_aggs_dict.items() if v[0] in df_era_processed.columns}
        )
    if weekly_aggs_dict:
        df_weekly_features = df_era_processed.groupby(pd.Grouper(freq='W-MON')).agg(
            **{k: v for k, v in weekly_aggs_dict.items() if v[0] in df_era_processed.columns}
        )
    
    print(f"  Daily aggregates shape: {df_daily_features.shape}, Weekly aggregates shape: {df_weekly_features.shape}")
    return df_daily_features, df_weekly_features

class EraFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, era_file_path: Path):
        self.era_file_path = era_file_path
        self.era_df = None
        try:
            print(f"Attempting to load era features from: {self.era_file_path}")
            self.era_df = pd.read_parquet(self.era_file_path)
            
            if 'time' in self.era_df.columns:
                self.era_df['time'] = pd.to_datetime(self.era_df['time'], errors='coerce', utc=True)
                self.era_df = self.era_df.set_index('time')
            elif self.era_df.index.name == 'time':
                if not isinstance(self.era_df.index, pd.DatetimeIndex):
                    self.era_df.index = pd.to_datetime(self.era_df.index, errors='coerce', utc=True)
                elif self.era_df.index.tz is None: 
                    self.era_df.index = self.era_df.index.tz_localize('UTC')
            else: 
                 raise ValueError(f"'time' column or index not found in era file: {self.era_file_path}")

            self.era_df.dropna(axis=1, how='all', inplace=True)
            self.era_df = self.era_df[self.era_df.index.notna()]

            print(f"Successfully loaded era features. Index type: {type(self.era_df.index)}, Columns: {self.era_df.columns.tolist()}")
            if self.era_df.empty:
                print(f"Warning: Era feature DataFrame is empty after loading and time processing from {self.era_file_path}.")

        except FileNotFoundError:
            print(f"Warning: Era feature file not found at {self.era_file_path}. Era features will not be added.")
            self.era_df = pd.DataFrame()
        except Exception as e:
            print(f"Error loading or processing era feature file {self.era_file_path}: {e}")
            self.era_df = pd.DataFrame()

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.era_df is None or self.era_df.empty:
            print("Era DataFrame not loaded or empty, returning original DataFrame.")
            return X
        
        if not isinstance(X.index, pd.DatetimeIndex):
            print("Warning: Input DataFrame X to EraFeatureGenerator does not have a DatetimeIndex. Transformation might fail or be incorrect.")
            return X 
        
        X_work = X.copy()
        if self.era_df.index.tz is not None and X_work.index.tz is None:
            print(f"Warning: Era DF index is TZ-aware ({self.era_df.index.tz}) but input X index is naive. Localizing X to UTC for join.")
            X_work.index = X_work.index.tz_localize('UTC', ambiguous='infer', nonexistent='NaT')
            X = X[X_work.index.notna()]
        elif self.era_df.index.tz is None and X_work.index.tz is not None:
            print(f"Warning: Era DF index is naive but input X index is TZ-aware ({X_work.index.tz}). Making X naive for join.")
            X_work.index = X_work.index.tz_convert(None)

        try:
            X_transformed = X.join(self.era_df, how='left', rsuffix='_era')
            
            era_cols_potentially_added = self.era_df.columns.tolist()
            newly_added_cols = [c for c in X_transformed.columns if c not in X.columns and (c in era_cols_potentially_added or c.replace('_era','') in era_cols_potentially_added)]
            
            if newly_added_cols:
                print(f"Joined era features: {newly_added_cols}. Applying ffill/bfill to these columns.")
                X_transformed[newly_added_cols] = X_transformed[newly_added_cols].ffill().bfill()
            else:
                 print("No new era columns seem to have been added by the join, or they were all NaNs.")
                   
            print(f"Shape after attempting to join era features: {X_transformed.shape}")
            return X_transformed
        except Exception as e:
            print(f"Error during era feature transformation (join): {e}")
            return X

def scale_data_for_era(df: pd.DataFrame, era_identifier: str, era_config: Dict[str, Any], global_config: Dict[str, Any], scalers_dir: Path, fit_scalers: bool = True, existing_scalers: Dict[str, MinMaxScaler] = None) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    if df.empty:
        print(f"(Era: {era_identifier}) DataFrame is empty, skipping scaling.")
        return df, {}

    print(f"\n--- Scaling Data for Era: {era_identifier} (fit_scalers={fit_scalers}) ---")
    scalers_dir.mkdir(parents=True, exist_ok=True)
    era_scalers_path = scalers_dir / f"{era_identifier}_scalers.joblib"

    cols_to_scale = []
    boolean_cols_converted = era_config.get("boolean_columns_to_int", 
                                          global_config.get('common_settings',{}).get("boolean_columns_to_int", []))

    defined_cols_in_rules = set()
    outlier_rules_ref = era_config.get('outlier_rules_ref', 'default_outlier_rules')
    imputation_rules_ref = era_config.get('imputation_rules_ref', 'default_imputation_rules')
    outlier_rules = global_config.get('preprocessing_rules', {}).get(outlier_rules_ref, [])
    imputation_rules = global_config.get('preprocessing_rules', {}).get(imputation_rules_ref, [])
    for rule in outlier_rules: defined_cols_in_rules.add(rule['column'])
    for rule in imputation_rules: defined_cols_in_rules.add(rule['column'])
    
    for col in df.columns:
        if col in boolean_cols_converted:
            print(f"  Skipping MinMaxScaling for column '{col}' (identified as boolean/status 0/1).")
            continue
        
        if df[col].dtype == 'float64' or df[col].dtype == 'float32':
            cols_to_scale.append(col)
        elif col in defined_cols_in_rules and pd.api.types.is_numeric_dtype(df[col]):
             if col not in cols_to_scale: cols_to_scale.append(col)
    
    print(f"Identified columns for MinMaxScaling to [-1,1]: {cols_to_scale}")

    df_scaled = df.copy()
    current_scalers = existing_scalers if existing_scalers is not None else {}

    if fit_scalers:
        for col in cols_to_scale:
            if col in df_scaled.columns and not df_scaled[col].isnull().all():
                data_to_scale = df_scaled[[col]].astype(float)
                if data_to_scale.dropna().empty:
                    print(f"  Column '{col}' contains all NaN or non-convertible values. Skipping scaling.")
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
            print(f"Warning: fit_scalers is False and Scaler file not found at {era_scalers_path}. Cannot transform data.")
            return df_scaled, {}
        
        for col in cols_to_scale:
            if col in df_scaled.columns and col in current_scalers and not df_scaled[col].isnull().all():
                data_to_transform = df_scaled[[col]].astype(float)
                if data_to_transform.dropna().empty:
                    print(f"  Column '{col}' contains all NaN or non-convertible values after astype(float). Skipping transform.")
                    continue
                df_scaled[col] = current_scalers[col].transform(data_to_transform)
                print(f"  Transformed column '{col}' using provided/loaded scaler.")
            elif col in df_scaled.columns and not df_scaled[col].isnull().all():
                print(f"  Warning: No scaler provided/loaded for column '{col}'. Skipping scaling for it.")

    return df_scaled, current_scalers 
