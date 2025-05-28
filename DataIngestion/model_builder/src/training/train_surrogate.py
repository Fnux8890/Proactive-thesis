import logging
import os
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import mlflow
from pytorch_lightning.loggers import MLFlowLogger
import matplotlib.pyplot as plt

# New imports for refactored structure
from ..config import GlobalConfig, DataConfig # Relative import
from ..models.components import LSTMBackbone
from ..models.lstm_regressor import LSTMRegressor

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants moved to Config dataclasses
# SEQUENCE_LENGTH = 24
# BATCH_SIZE = 64
# LEARNING_RATE = 1e-3

# --- Data Handling ---
class TimeSeriesDataset(Dataset):
    """Simple Dataset wrapper for time series sequences and targets."""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

def create_sequences(data, seq_length):
    """Creates sequences and targets from time series data."""
    sequences = []
    targets = []
    if len(data) <= seq_length:
        logger.warning(f"Data length ({len(data)}) <= sequence length ({seq_length}). Cannot create sequences.")
        return np.array([]), np.array([])

    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

def load_and_preprocess_data(data_dir: Path, cfg: DataConfig, smoke_test: bool):
    """Loads Parquet files, preprocesses, scales, splits, and creates sequences."""
    logger.info(f"Loading data from {data_dir} using config: {cfg}")
    all_files = list(data_dir.glob('*.parquet'))
    if not all_files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")

    if smoke_test:
        logger.warning("Smoke test enabled: Using only the first Parquet file.")
        all_files = all_files[:1]

    df_list = []
    for f in all_files:
        try:
            table = pq.read_table(f)
            df_list.append(table.to_pandas())
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            continue

    if not df_list:
        raise ValueError("No valid Parquet files could be loaded.")

    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Loaded combined data with shape: {df.shape}")

    initial_rows = len(df)
    df = df.ffill().fillna(0)
    rows_after_fillna = len(df)
    if rows_after_fillna != initial_rows:
        logger.warning(f"Row count changed during fillna (shouldn't happen): {initial_rows} -> {rows_after_fillna}")
    if df.empty:
        raise ValueError("DataFrame is empty after attempting to fill NaNs.")
    logger.info("NaN values handled using forward fill and zero fill.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in the data for training.")
    logger.info(f"Using numeric columns as features/targets: {numeric_cols}")

    feature_columns = numeric_cols
    target_columns = numeric_cols
    num_features = len(feature_columns)
    num_targets = len(target_columns)

    data_to_process = df[feature_columns].values

    min_rows_for_test = 50 # Increase minimum rows needed if splitting 3 ways
    if smoke_test and len(data_to_process) < min_rows_for_test:
        raise ValueError(f"Insufficient data ({len(data_to_process)} rows) for smoke test with 3 splits (need ~{min_rows_for_test}).")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_process)
    logger.info("Features scaled using StandardScaler.")

    sequences, targets = create_sequences(scaled_data, cfg.sequence_length)
    if sequences.size == 0:
         raise ValueError("Not enough data to create sequences after processing.")
    logger.info(f"Created sequences of shape: {sequences.shape}, targets of shape: {targets.shape}")

    if targets.shape[1] != num_targets:
         raise ValueError(f"Target shape mismatch. Expected {num_targets} features, got {targets.shape[1]}." )

    # --- Train/Validation/Test Split (Chronological 70/15/15) ---
    n_total = len(sequences)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    # Ensure train and val splits have at least 1 sample if possible
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    # Ensure test split doesn't overlap or underflow
    n_test = n_total - n_train - n_val
    if n_test < 1 and n_total > n_train: # If no test samples left, steal one from val
        n_val = max(1, n_val - 1)
        n_test = 1
    if n_train + n_val + n_test > n_total: # Recalculate test if adjustments caused overflow
         n_test = n_total - n_train - n_val

    if n_train <= 0 or n_val <=0 or n_test <= 0:
        raise ValueError(f"Data too small for 70/15/15 split. Total sequences: {n_total}. Calculated splits: Train={n_train}, Val={n_val}, Test={n_test}")

    X_train = sequences[:n_train]
    y_train = targets[:n_train]

    X_val = sequences[n_train : n_train + n_val]
    y_val = targets[n_train : n_train + n_val]

    X_test = sequences[n_train + n_val :]
    y_test = targets[n_train + n_val :]

    logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # Return all splits
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, num_features, num_targets

# --- Main Training Function ---
def train_model(args, cfg: GlobalConfig):
    """Main function to orchestrate the training process using configuration."""
    logger.info("Starting model training process...")
    pl.seed_everything(42, workers=True) # Seed for reproducibility

    # --- Determine Output Directory based on GPU Tag --- 
    gpu_tag = os.getenv("GPU_TAG", "default_gpu") # Get tag from env var, default if not set
    # Check if actually running on CPU despite tag
    if not torch.cuda.is_available():
        gpu_tag = "cpu"
    base_model_dir = Path(args.model_dir if args.model_dir else cfg.model_dir)
    model_dir = base_model_dir / gpu_tag # Create subdirectory
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using GPU Tag: {gpu_tag}")
    logger.info(f"Model artifacts will be saved to: {model_dir}")

    # Use data path from config or args
    data_dir = Path(args.data_dir if args.data_dir else cfg.data_dir)
    logger.info(f"Using data directory: {data_dir}")


    # --- Load and Prepare Data (including test set now) ---
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, n_features, n_targets = load_and_preprocess_data(
            data_dir, cfg.data, args.smoke_test
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Data preparation failed: {e}")
        return

    # Save the scaler to the GPU-specific directory
    scaler_path = model_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # --- Create Datasets and DataLoaders ---
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test) # Create test dataset

    # Adjust num_workers based on CPU count if default is too high
    num_workers = min(cfg.data.num_workers, os.cpu_count() // 2 if os.cpu_count() else 1)
    logger.info(f"Using num_workers={num_workers} for DataLoaders.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers if num_workers > 0 else False
    )
    # Create DataLoader for the test set
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers if num_workers > 0 else False
    )

    # --- Configure MLflow Logger ---
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    experiment_name = f"SurrogateModelTraining_{gpu_tag}"
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_tracking_uri,
        log_model=False # Disable automatic checkpoint logging
    )
    # Log hyperparameters defined in our config
    # Filter out complex objects like default_epochs_by_gpu for logging
    hparams_to_log = {
         "data": cfg.data.__dict__,
         "model": cfg.model.__dict__,
         "trainer": {k:v for k,v in cfg.trainer.__dict__.items() if k != 'max_epochs'}, # log determined epochs separately
         "epochs_run": cfg.trainer.max_epochs, # Log the actual epochs configured for this run
         "gpu_tag": gpu_tag,
         # Add specific args if needed
         "smoke_test": args.smoke_test
    }
    # MLflowLogger doesn't automatically log all hparams like TensorBoard might
    # We might need to log them manually at the start or rely on params logged by Trainer
    # For explicit logging:
    # mlflow.log_params(flatten_dict(hparams_to_log)) # Requires a utility to flatten nested dicts


    # --- Initialize Model and Trainer ---
    backbone = LSTMBackbone(
        n_features=n_features,
        n_targets=n_targets,
        hidden_units=cfg.model.hidden_units,
        num_layers=cfg.model.num_layers
    )
    model = LSTMRegressor(
        backbone=backbone,
        learning_rate=cfg.trainer.learning_rate
    )

    # --- Callbacks ---
    # Keep ModelCheckpoint for saving locally
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        # Keep a simple filename for local saving
        filename='best_model_epoch_{epoch}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min"
    )

    # Max epochs already determined in the main block
    max_epochs = cfg.trainer.max_epochs
    # Log smoke test only if applicable
    if args.smoke_test:
        logger.warning("Smoke test: Limiting training to 1 epoch.")
    else:
        logger.info(f"Training for {max_epochs} epochs.") # Log the determined epochs

    # Precision already determined in the main block
    precision_setting = cfg.trainer.precision
    # Log GPU fallback if needed
    if "mixed" in precision_setting and not torch.cuda.is_available():
        logger.warning(f"Mixed precision ({precision_setting}) requested but no GPU found. Using FP32.")
        precision_setting = 32

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=mlf_logger,
        deterministic=cfg.trainer.deterministic,
        precision=precision_setting,
        gradient_clip_val=0.5, # Add gradient clipping (common for LSTMs)
        enable_progress_bar=False # Disable the progress bar log clutter
    )

    # --- Train the Model ---
    logger.info(f"Starting training with config: {cfg.trainer}")
    logger.info(f"MLflow Run ID (for tracking): {mlf_logger.run_id}") # Log run ID
    best_model_path = None
    try:
        mlf_logger.log_hyperparams(hparams_to_log)
        trainer.fit(model, train_loader, val_loader)
        logger.info("Training finished.")
        # Store the best path after fit completes
        if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
            best_model_path = checkpoint_callback.best_model_path
            logger.info(f"Best model checkpoint saved locally to: {best_model_path}")
        else:
            logger.warning("Could not determine best model path from checkpoint callback.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        return

    # --- Log Best Checkpoint Manually to MLflow ---
    if best_model_path and Path(best_model_path).exists():
        try:
            logger.info(f"Logging best checkpoint artifact to MLflow: {best_model_path}")
            # Use mlflow.log_artifact directly
            mlflow.log_artifact(best_model_path, artifact_path="checkpoints") 
            logger.info("Successfully logged checkpoint to MLflow artifacts.")
        except Exception as e:
             logger.error(f"Failed to log checkpoint artifact {best_model_path} to MLflow: {e}", exc_info=True)

    # --- Test the Best Model ---
    test_results = None # Initialize test_results
    if best_model_path and Path(best_model_path).exists():
        logger.info(f"Loading best model from {best_model_path} for testing...")
        try:
             test_results = trainer.test(model=model, dataloaders=test_loader, ckpt_path=best_model_path)
             logger.info(f"Test results (scaled): {test_results}")
             if test_results: 
                 mlf_logger.log_metrics({f"test_{k}": v for k, v in test_results[0].items()})
        except Exception as e:
             logger.error(f"An error occurred during testing: {e}", exc_info=True)
    elif model: 
         logger.warning("No best checkpoint found. Testing the model's last state (might not be optimal).")
         try:
              test_results = trainer.test(model=model, dataloaders=test_loader)
              logger.info(f"Test results (scaled, last state): {test_results}")
              if test_results:
                  mlf_logger.log_metrics({f"test_{k}_last_state": v for k, v in test_results[0].items()})
         except Exception as e:
              logger.error(f"An error occurred during testing (last state): {e}", exc_info=True)
    else:
        logger.error("Testing skipped: No model available (training might have failed early).")

    # --- Calculate & Log Original Scale Metrics --- 
    # Requires predictions generated from the test set using the evaluated model
    if test_results: # Only proceed if testing ran
        try:
            logger.info("Calculating metrics on original data scale...")
            # Reload the model checkpoint that was tested
            tested_ckpt_path = best_model_path if best_model_path and Path(best_model_path).exists() else None
            
            if tested_ckpt_path:
                 backbone_for_eval = LSTMBackbone(n_features, n_targets, cfg.model.hidden_units, cfg.model.num_layers)
                 model_for_eval = LSTMRegressor.load_from_checkpoint(
                     tested_ckpt_path, 
                     backbone=backbone_for_eval,
                     map_location=torch.device('cpu') if not torch.cuda.is_available() else None
                 ).eval()
            elif model: # Use last state if no checkpoint path was tested
                model_for_eval = model.eval()
            else: 
                raise RuntimeError("Model for evaluation not available.")

            # Generate predictions on the test set (again)
            test_preds_scaled = []
            test_actuals_scaled = []
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    x = x.to(model_for_eval.device)
                    y_hat = model_for_eval(x)
                    test_preds_scaled.append(y_hat.cpu().numpy())
                    test_actuals_scaled.append(y.cpu().numpy())
            test_preds_scaled = np.concatenate(test_preds_scaled, axis=0)
            test_actuals_scaled = np.concatenate(test_actuals_scaled, axis=0)

            # Inverse transform
            n_scaler_features = scaler.n_features_in_
            n_prediction_features = test_preds_scaled.shape[-1]
            if n_prediction_features == n_scaler_features:
                test_preds_orig = scaler.inverse_transform(test_preds_scaled)
                test_actuals_orig = scaler.inverse_transform(test_actuals_scaled)
                
                # Calculate metrics per target
                metrics_original = {}
                # TODO: Get actual column names instead of indices
                original_col_names = [f"Target_{i}" for i in range(n_targets)]
                for i in range(n_targets):
                    col_name = original_col_names[i]
                    col_actuals = test_actuals_orig[:, i]
                    col_preds = test_preds_orig[:, i]
                    mae_orig = np.mean(np.abs(col_actuals - col_preds))
                    mse_orig = np.mean((col_actuals - col_preds)**2)
                    rmse_orig = np.sqrt(mse_orig)
                    metrics_original[f"test_mae_orig_{col_name}"] = mae_orig
                    metrics_original[f"test_rmse_orig_{col_name}"] = rmse_orig
                
                if metrics_original:
                    logger.info(f"Original scale test metrics (sample): {list(metrics_original.items())[:5]}...")
                    mlf_logger.log_metrics(metrics_original)
                else:
                     logger.warning("No original scale metrics were calculated.")
            else:
                 logger.warning("Cannot calculate original scale metrics: Scaler feature mismatch.")

        except Exception as e:
             logger.error(f"Failed to calculate or log original scale metrics: {e}", exc_info=True)

    # --- Log Scaler as Artifact ---
    if scaler_path.exists():
        try:
            logger.info(f"Logging scaler artifact: {scaler_path}")
            # Use mlflow.log_artifact directly
            mlflow.log_artifact(scaler_path, artifact_path="scaler") 
            logger.info("Successfully logged scaler artifact to MLflow.")
        except Exception as e:
            logger.error(f"Failed to log scaler artifact {scaler_path} to MLflow: {e}", exc_info=True)

    # --- Plot Test Results ---
    plot_path = None # Initialize plot_path
    if test_results: # Only plot if testing actually ran and produced results
        try:
            logger.info("Generating test prediction plot...")
            # Reload the model used for testing (either best or last state)
            test_ckpt_path = best_model_path if best_model_path and Path(best_model_path).exists() else None
            if test_ckpt_path:
                 # Instantiate backbone needed for loading
                 backbone_for_plot = LSTMBackbone(
                     n_features=n_features,
                     n_targets=n_targets,
                     hidden_units=cfg.model.hidden_units,
                     num_layers=cfg.model.num_layers
                 )
                 predictor_for_plot = LSTMRegressor.load_from_checkpoint(
                     test_ckpt_path,
                     backbone=backbone_for_plot, # Pass the backbone instance
                     map_location=torch.device('cpu') if not torch.cuda.is_available() else None
                 ).eval()
            elif model: 
                # If testing last state, the model object is already available
                predictor_for_plot = model.eval()
            else: 
                raise RuntimeError("Cannot plot: Model is not available.")

            test_preds_scaled = []
            test_actuals_scaled = []
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    x = x.to(predictor_for_plot.device)
                    y_hat = predictor_for_plot(x)
                    test_preds_scaled.append(y_hat.cpu().numpy())
                    test_actuals_scaled.append(y.cpu().numpy())

            test_preds_scaled = np.concatenate(test_preds_scaled, axis=0)
            test_actuals_scaled = np.concatenate(test_actuals_scaled, axis=0)

            n_scaler_features = scaler.n_features_in_
            n_prediction_features = test_preds_scaled.shape[-1]

            if n_prediction_features == n_scaler_features:
                test_preds_orig = scaler.inverse_transform(test_preds_scaled)
                test_actuals_orig = scaler.inverse_transform(test_actuals_scaled)
            else:
                 logger.warning("Plotting scaled data: Cannot inverse transform if prediction dim != scaler features.")
                 test_preds_orig = test_preds_scaled
                 test_actuals_orig = test_actuals_scaled

            num_plots = min(5, n_targets)
            plot_steps = min(500, len(test_actuals_orig))
            fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3 * num_plots), sharex=True)
            if num_plots == 1:
                 axes = [axes]
            column_names = [f'Target_{i}' for i in range(num_plots)]

            for i in range(num_plots):
                axes[i].plot(test_actuals_orig[:plot_steps, i], label='Actual')
                axes[i].plot(test_preds_orig[:plot_steps, i], label='Predicted', linestyle='--')
                axes[i].set_title(f"{column_names[i]}")
                axes[i].legend()
            axes[-1].set_xlabel("Time Step (in Test Set)")
            fig.suptitle("Test Set Predictions vs Actuals (First N Steps)", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            plot_path = model_dir / "test_predictions_plot.png"
            fig.savefig(plot_path)
            logger.info(f"Test prediction plot saved to {plot_path}")

            if plot_path.exists():
                 try:
                      # Use mlflow.log_artifact directly
                      mlflow.log_artifact(plot_path, artifact_path="plots") 
                      logger.info("Successfully logged plot artifact to MLflow.")
                 except Exception as e:
                      logger.error(f"Failed to log plot artifact {plot_path} to MLflow: {e}", exc_info=True)
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to generate prediction plot: {e}", exc_info=True)
    else:
         logger.warning("Skipping plot generation as no test results were available.")

    logger.info("Model training script finished.")

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM surrogate model.")
    parser.add_argument('--data-dir', type=str, default=None, help='Override data directory path.')
    parser.add_argument('--model-dir', type=str, default=None, help='Override base model output directory.')
    parser.add_argument('--epochs', type=int, default=None, help='Explicitly set number of training epochs, overrides GPU defaults.')
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test (1 epoch, limited data).')
    args = parser.parse_args()

    # Load base config
    config = GlobalConfig()

    # Override paths from args if provided
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.model_dir:
        config.model_dir = args.model_dir

    # Determine GPU tag (used for output path AND default epochs)
    gpu_tag = os.getenv("GPU_TAG", "default_gpu")
    if not torch.cuda.is_available():
        gpu_tag = "cpu" # Override tag if no GPU detected

    # Determine max_epochs
    if args.epochs is not None:
        # CLI argument takes precedence
        config.trainer.max_epochs = args.epochs
        logger.info(f"Using explicit epochs from CLI: {config.trainer.max_epochs}")
    elif args.smoke_test:
        # Smoke test always uses 1 epoch
        config.trainer.max_epochs = 1
        # No need to log here, train_model will log the smoke test warning
    else:
        # Look up default epochs based on GPU tag
        config.trainer.max_epochs = config.default_epochs_by_gpu.get(gpu_tag, config.default_epochs_by_gpu["default_gpu"])
        logger.info(f"Using default epochs for GPU tag '{gpu_tag}': {config.trainer.max_epochs}")

    # Final check for max_epochs (should always be set by now)
    if config.trainer.max_epochs is None or config.trainer.max_epochs < 1:
         logger.error("Invalid max_epochs determined. Setting to 1.")
         config.trainer.max_epochs = 1

    # Log GPU info (moved after determining gpu_tag)
    if torch.cuda.is_available():
         logger.info(f"Found {torch.cuda.device_count()} CUDA devices. Using: {torch.cuda.get_device_name(0)}")
         # Check for BF16 support (Ampere or newer) for mixed precision advice
         if torch.cuda.is_bf16_supported() and "bf16" in config.trainer.precision:
              logger.info("BF16 is supported and configured.")
         elif "16-mixed" == config.trainer.precision:
              logger.info("Using FP16 for mixed precision.")
    else:
         logger.warning("No CUDA devices found. Training will run on CPU (FP32).")

    # Pass the populated config object
    train_model(args, config) 