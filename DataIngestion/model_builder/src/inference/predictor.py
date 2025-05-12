import torch
import joblib
import numpy as np
from pathlib import Path
import os # Import os to read environment variable
from typing import Tuple, Optional, List
import logging

# Use numpy typing if available (requires numpy >= 1.20)
try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = np.ndarray # Fallback for older numpy

# New imports for refactored structure
from ..models.lstm_regressor import LSTMRegressor # LightningModule for loading checkpoint
# We might not need config here if model/scaler implicitly contain necessary info
# from ..config import GlobalConfig

logger = logging.getLogger(__name__)

class SurrogatePredictor:
    """Handles loading the surrogate model/scaler and running inference."""
    
    model: Optional[LSTMRegressor] = None
    scaler: Optional[joblib.numpy_pickle.NumpyPickler] = None
    base_model_dir: Path # Store the base directory provided
    gpu_specific_model_dir: Path # Store the actual directory with artifacts
    # Store feature/target info if needed for complex inverse scaling
    # feature_columns: Optional[List[str]] = None 
    # target_columns: Optional[List[str]] = None 

    def __init__(self, model_dir: Path, gpu_tag: Optional[str] = None):
        """Initializes the predictor by loading the model and scaler once.

        Args:
            model_dir (Path): The base directory where GPU-specific subdirectories exist.
            gpu_tag (Optional[str]): The specific GPU tag (e.g., 'gtx1660') to load from.
                                      If None, reads from GPU_TAG env var or defaults.
        """
        self.base_model_dir = model_dir
        if gpu_tag is None:
            gpu_tag = os.getenv("GPU_TAG", "default_gpu") # Match default in training
            logger.info(f"No GPU tag provided, using tag from environment or default: '{gpu_tag}'")
        else:
            logger.info(f"Using provided GPU tag: '{gpu_tag}'")
        
        self.gpu_specific_model_dir = self.base_model_dir / gpu_tag
        
        if not self.gpu_specific_model_dir.is_dir():
             raise FileNotFoundError(f"GPU-specific model directory not found: {self.gpu_specific_model_dir}")
             
        self.model, self.scaler = self._load(self.gpu_specific_model_dir)
        # TODO: Load feature/target column names if they were saved during training
        # Example: 
        # try:
        #     with open(model_dir / "feature_info.json", 'r') as f:
        #         info = json.load(f)
        #         self.feature_columns = info['feature_columns']
        #         self.target_columns = info['target_columns']
        # except FileNotFoundError:
        #      logger.warning("Feature/target info file not found. Assuming targets == features for inverse scaling.")
        #      pass # Fallback handled in _postprocess

    @staticmethod
    def _load(gpu_specific_model_dir: Path) -> Tuple[LSTMRegressor, joblib.numpy_pickle.NumpyPickler]:
        """Loads the best checkpoint and scaler from the GPU-specific directory."""
        logger.info(f"Loading model artifacts from: {gpu_specific_model_dir}")
        try:
            checkpoints = list(gpu_specific_model_dir.glob('best-surrogate-*.ckpt'))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint files (*.ckpt) found in {gpu_specific_model_dir}")

            def get_val_loss(ckpt_path):
                try:
                    # Extract loss from filename like 'best-surrogate-epoch=XX-val_loss=Y.YY.ckpt'
                    parts = ckpt_path.stem.split('-')
                    for part in reversed(parts):
                        if 'val_loss=' in part:
                             return float(part.split('=')[1])
                    return float('inf') # Fallback if format unexpected
                except (IndexError, ValueError):
                    return float('inf')

            # Find checkpoint with the minimum validation loss
            best_ckpt_path = min(checkpoints, key=get_val_loss)
            if get_val_loss(best_ckpt_path) == float('inf'):
                logger.warning("Could not parse validation loss from checkpoint names. Loading the most recently modified checkpoint.")
                best_ckpt_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            
            logger.info(f"Loading checkpoint: {best_ckpt_path}")
            # Ensure map_location handles CPU-only inference if needed
            model = LSTMRegressor.load_from_checkpoint(str(best_ckpt_path), map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
            model.eval() # Set to evaluation mode
            model.freeze() # Freeze weights

            scaler_path = gpu_specific_model_dir / 'scaler.joblib'
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler file (scaler.joblib) not found at {scaler_path}")
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from: {scaler_path}")

            logger.info("Model and scaler loaded successfully.")
            return model, scaler
        except Exception as e:
            logger.error(f"Error loading model or scaler: {e}", exc_info=True)
            raise

    def _preprocess(self, features: NDArray) -> torch.Tensor:
        """Scales input features and converts to tensor."""
        if not isinstance(features, np.ndarray):
            try:
                features = np.array(features, dtype=np.float32)
            except Exception as e:
                raise TypeError(f"Input features could not be converted to a numpy array: {e}")

        if self.scaler is None:
            raise RuntimeError("Scaler is not loaded.")

        expected_features = self.scaler.n_features_in_
        if features.shape[-1] != expected_features:
            raise ValueError(f"Input feature dim ({features.shape[-1]}) != scaler expected ({expected_features})")

        original_shape = features.shape
        if len(original_shape) == 2: # Single sequence [seq_len, n_features]
            scaled_features = self.scaler.transform(features)
            scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
        elif len(original_shape) == 3: # Batch of sequences [batch_size, seq_len, n_features]
            batch_size, seq_len, n_features_in = original_shape
            features_2d = features.reshape(-1, n_features_in)
            scaled_features_2d = self.scaler.transform(features_2d)
            scaled_features = scaled_features_2d.reshape(batch_size, seq_len, n_features_in)
            scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}. Expected [seq_len, n_features] or [batch_size, seq_len, n_features].")

        # Move tensor to the same device as the model
        if self.model:
             scaled_features_tensor = scaled_features_tensor.to(self.model.device)
        
        return scaled_features_tensor

    def _postprocess(self, predictions_scaled: NDArray) -> NDArray:
        """Inverse transforms scaled predictions back to the original data scale."""
        if self.scaler is None:
            raise RuntimeError("Scaler is not loaded.")

        n_scaler_features = self.scaler.n_features_in_
        n_prediction_features = predictions_scaled.shape[-1]

        # Simple case: prediction has same number of features as scaler
        if n_prediction_features == n_scaler_features:
            try:
                # Inverse transform directly
                return self.scaler.inverse_transform(predictions_scaled)
            except ValueError as e:
                 logger.error(f"Error during inverse_transform: {e}. Shape mismatch? Scaled prediction shape: {predictions_scaled.shape}, Scaler features: {n_scaler_features}", exc_info=True)
                 raise

        # Complex case: Target subset (TODO: requires feature/target name mapping)
        # Assuming for now that train_surrogate used all numeric cols for both,
        # so n_prediction_features should equal n_scaler_features.
        # If this warning appears, the assumption was wrong or something changed.
        logger.warning(f"Prediction dim ({n_prediction_features}) != scaler features ({n_scaler_features}). Cannot accurately inverse scale without feature mapping. Returning SCALED predictions.")
        return predictions_scaled

    def predict(self, features: NDArray) -> NDArray:
        """Runs preprocessing, inference, and postprocessing."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first or initialize the class.")
            
        scaled_features_tensor = self._preprocess(features)
        
        with torch.no_grad():
            # Model expects input shape like [batch, seq_len, features]
            prediction_tensor = self.model(scaled_features_tensor)
        
        predictions_scaled = prediction_tensor.cpu().numpy()
        predictions_original_scale = self._postprocess(predictions_scaled)
        
        return predictions_original_scale

    # Alias __call__ to predict for convenience
    def __call__(self, features: NDArray) -> NDArray:
        return self.predict(features)

# --- Example Usage --- (Requires model artifacts from training)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing SurrogatePredictor...")

    # Point to the BASE directory where GPU-specific subdirs are expected
    base_model_dir = Path("./test_model_output") # CHANGE THIS to your actual BASE model output dir
    gpu_tag_to_test = os.getenv("GPU_TAG", "default_gpu") # Test loading based on env var or default
    model_output_dir_to_test = base_model_dir / gpu_tag_to_test

    if not model_output_dir_to_test.exists() or not any(model_output_dir_to_test.glob('*.ckpt')):
        logger.error(f"GPU-specific model directory {model_output_dir_to_test} does not exist or contains no checkpoints.")
        logger.error("Please run the training script first (ensure GPU_TAG is set if needed) and update the path.")
    else:
        try:
            # Pass the BASE directory to the constructor
            predictor = SurrogatePredictor(base_model_dir)
            logger.info("Predictor initialized successfully.")
            
            n_features = predictor.scaler.n_features_in_
            seq_len = 24
            logger.info(f"Creating dummy input: sequence length={seq_len}, features={n_features}")

            dummy_input_single = np.random.rand(seq_len, n_features).astype(np.float32)
            logger.info(f"Input shape (single): {dummy_input_single.shape}")
            prediction_single = predictor(dummy_input_single)
            logger.info(f"Prediction shape (single): {prediction_single.shape}")
            logger.info(f"Sample prediction (single): {prediction_single[0]}")

            batch_size = 5
            dummy_input_batch = np.random.rand(batch_size, seq_len, n_features).astype(np.float32)
            logger.info(f"Input shape (batch): {dummy_input_batch.shape}")
            prediction_batch = predictor(dummy_input_batch)
            logger.info(f"Prediction shape (batch): {prediction_batch.shape}")
            logger.info(f"Sample prediction (batch[0]): {prediction_batch[0]}")

        except Exception as e:
            logger.error(f"An error occurred during testing: {e}", exc_info=True)

    logger.info("Predictor test finished.") 