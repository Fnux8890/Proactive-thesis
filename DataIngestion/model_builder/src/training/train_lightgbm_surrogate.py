"""LightGBM Surrogate Model Training Pipeline with CUDA Support.

This module implements a clean architecture for training LightGBM models
as surrogate models for MOEA optimization. Key features:

- CUDA acceleration via LightGBM GPU support
- Integration with feature extraction pipeline
- Phenotype data integration for plant growth modeling
- Multiple objective support (energy, plant growth, etc.)
- MLflow experiment tracking
- Model serialization for MOEA deployment

Usage:
    python train_lightgbm_surrogate.py --target energy_consumption
    python train_lightgbm_surrogate.py --target plant_growth --use-phenotypes
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# GPU availability check
def check_gpu_availability():
    """Check and log GPU availability for both PyTorch and LightGBM."""
    logger.info("=" * 60)
    logger.info("GPU AVAILABILITY CHECK")
    logger.info("=" * 60)

    # Check PyTorch CUDA
    try:
        import torch
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"PyTorch CUDA device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.warning(f"PyTorch CUDA check failed: {e}")

    # Check NVIDIA environment variables
    logger.info(f"NVIDIA_VISIBLE_DEVICES: {os.getenv('NVIDIA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'Not set')}")

    # Check if nvidia-smi is available
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info(f"nvidia-smi output: {result.stdout.strip()}")
        else:
            logger.warning("nvidia-smi command failed")
    except Exception as e:
        logger.warning(f"nvidia-smi not available: {e}")

    logger.info("=" * 60)


# MLflow wrapper functions to handle when MLflow is not available
def safe_mlflow_log_params(params):
    try:
        mlflow.log_params(params)
    except Exception as e:
        logger.debug(f"MLflow log_params failed: {e}")


def safe_mlflow_log_metrics(metrics):
    try:
        mlflow.log_metrics(metrics)
    except Exception as e:
        logger.debug(f"MLflow log_metrics failed: {e}")


def safe_mlflow_log_metric(key, value):
    try:
        mlflow.log_metric(key, value)
    except Exception as e:
        logger.debug(f"MLflow log_metric failed: {e}")


def safe_mlflow_log_artifact(path):
    try:
        mlflow.log_artifact(path)
    except Exception as e:
        logger.debug(f"MLflow log_artifact failed: {e}")


def safe_mlflow_log_model(model, artifact_path):
    try:
        mlflow.lightgbm.log_model(model, artifact_path)
    except Exception as e:
        logger.debug(f"MLflow log_model failed: {e}")


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    features_table: str = "tsfresh_features"
    phenotypes_table: str = "phenotypes"
    target_columns: dict[str, str] = field(
        default_factory=lambda: {
            "energy_consumption": "total_energy_kwh",
            "plant_growth": "growth_rate",
            "water_usage": "water_consumption_l",
            "temperature_stability": "temp_variance",
        }
    )
    use_phenotypes: bool = False
    train_test_split: float = 0.8
    validation_folds: int = 5
    scale_features: bool = True
    scaler_type: str = "robust"  # "standard" or "robust"


@dataclass
class ModelConfig:
    """Configuration for LightGBM model."""

    objective: str = "regression"
    metric: str = "rmse"
    boosting_type: str = "gbdt"
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 1000
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42
    # GPU parameters
    device: str = "gpu"  # "cpu" or "gpu"
    gpu_platform_id: int = 0
    gpu_device_id: int = 0
    gpu_use_dp: bool = False  # Use double precision
    # Early stopping
    early_stopping_rounds: int = 50
    verbose: int = -1


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""

    experiment_name: str = "LightGBM_Surrogate_Models"
    run_name: str | None = None
    tracking_uri: str = ""  # Empty string disables MLflow
    log_models: bool = True
    log_plots: bool = True


# ============================================================================
# Abstract Base Classes
# ============================================================================


class DataLoader(ABC):
    """Abstract base class for data loading."""

    @abstractmethod
    def load_features(self) -> pd.DataFrame:
        """Load feature data."""
        pass

    @abstractmethod
    def load_phenotypes(self) -> pd.DataFrame:
        """Load phenotype data."""
        pass

    @abstractmethod
    def prepare_training_data(self, target: str, use_phenotypes: bool) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        pass


class ModelTrainer(ABC):
    """Abstract base class for model training."""

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> Any:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the model."""
        pass


# ============================================================================
# Concrete Implementations
# ============================================================================


class PostgreSQLDataLoader(DataLoader):
    """Data loader for PostgreSQL/TimescaleDB."""

    def __init__(self, connection_string: str, config: DataConfig):
        self.engine = create_engine(connection_string)
        self.config = config
        self._features_df: pd.DataFrame | None = None
        self._phenotypes_df: pd.DataFrame | None = None

    def load_features(self) -> pd.DataFrame:
        """Load features from tsfresh_features table."""
        if self._features_df is not None:
            return self._features_df

        logger.info(f"Loading features from {self.config.features_table}")

        try:
            # First check which columns exist in the table
            col_check_query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{self.config.features_table}'
            AND column_name IN ('era_id', 'index')
            """
            existing_cols = pd.read_sql(text(col_check_query), self.engine)
            has_era_id = 'era_id' in existing_cols['column_name'].values
            has_index = 'index' in existing_cols['column_name'].values

            if has_era_id:
                # Table has era_id column, use it directly
                query = f"""
                SELECT * FROM {self.config.features_table}
                ORDER BY era_id
                """
            elif has_index:
                # Table has index column but not era_id, rename index to era_id
                logger.info(f"Table {self.config.features_table} has 'index' column instead of 'era_id', renaming it")
                query = f"""
                SELECT
                    "index" AS era_id,
                    *
                FROM {self.config.features_table}
                ORDER BY "index"
                """
            else:
                raise ValueError(f"Table {self.config.features_table} has neither 'era_id' nor 'index' column")

            self._features_df = pd.read_sql(text(query), self.engine)

            # Remove duplicate columns if we renamed index to era_id
            if has_index and not has_era_id and 'index' in self._features_df.columns:
                self._features_df = self._features_df.drop(columns=['index'])

            logger.info(f"Loaded {len(self._features_df)} rows with {len(self._features_df.columns)} columns")

        except Exception as e:
            raise ValueError(f"Could not load features from {self.config.features_table}: {e}") from e

        return self._features_df

    def load_phenotypes(self) -> pd.DataFrame:
        """Load phenotype data."""
        if self._phenotypes_df is not None:
            return self._phenotypes_df

        query = f"""
        SELECT * FROM {self.config.phenotypes_table}
        """

        logger.info(f"Loading phenotypes from {self.config.phenotypes_table}")
        self._phenotypes_df = pd.read_sql(text(query), self.engine)
        logger.info(f"Loaded {len(self._phenotypes_df)} phenotype records")

        return self._phenotypes_df

    def prepare_training_data(self, target: str, use_phenotypes: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        features_df = self.load_features()

        # Check if target column exists
        target_col = self.config.target_columns.get(target)
        if not target_col or target_col not in features_df.columns:
            # Try to infer target from feature names
            possible_targets = [col for col in features_df.columns if target in col.lower()]
            if possible_targets:
                target_col = possible_targets[0]
                logger.warning(f"Target '{target}' not found, using inferred column: {target_col}")
            else:
                raise ValueError(f"Target column for '{target}' not found in features")

        # Separate features and target
        feature_cols = [
            col
            for col in features_df.columns
            if col
            not in [
                "era_id",
                "signal_name",
                "level",
                "stage",
                "start_time",
                "end_time",
                "era_rows",
                target_col,
            ]
        ]

        X = features_df[feature_cols].copy()
        y = features_df[target_col].copy()

        # Add phenotype features if requested
        if use_phenotypes and self.config.use_phenotypes:
            phenotypes_df = self.load_phenotypes()
            # Merge phenotype features (assuming plant_id or similar key exists)
            # This is simplified - actual implementation would need proper key matching
            logger.info("Adding phenotype features to training data")
            # Example: Add average phenotype values as features
            for col in phenotypes_df.select_dtypes(include=[np.number]).columns:
                X[f"phenotype_{col}_mean"] = phenotypes_df[col].mean()
                X[f"phenotype_{col}_std"] = phenotypes_df[col].std()

        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")

        return X, y


class LightGBMTrainer(ModelTrainer):
    """LightGBM model trainer with GPU support."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: lgb.Booster | None = None
        self.feature_importance_: pd.DataFrame | None = None

        # Check CUDA availability
        if config.device == "gpu":
            logger.info("Testing LightGBM GPU support...")
            try:
                # Test GPU availability by creating a small dataset
                test_data = lgb.Dataset(np.random.rand(100, 10), label=np.random.rand(100))
                test_params = {
                    "device": "gpu",
                    "gpu_platform_id": config.gpu_platform_id,
                    "gpu_device_id": config.gpu_device_id,
                    "objective": "regression",
                    "verbose": -1,
                }
                lgb.train(test_params, test_data, num_boost_round=1)
                logger.info("✓ GPU support confirmed for LightGBM")
            except Exception as e:
                logger.warning(f"✗ GPU not available for LightGBM: {e}")
                logger.warning("Falling back to CPU mode.")
                self.config.device = "cpu"

    def _get_params(self) -> dict[str, Any]:
        """Get LightGBM parameters."""
        params = {
            "objective": self.config.objective,
            "metric": self.config.metric,
            "boosting_type": self.config.boosting_type,
            "num_leaves": self.config.num_leaves,
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "min_child_samples": self.config.min_child_samples,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "random_state": self.config.random_state,
            "verbose": self.config.verbose,
            "n_jobs": -1,
        }

        # Add GPU parameters if using GPU
        if self.config.device == "gpu":
            params.update(
                {
                    "device": "gpu",
                    "gpu_platform_id": self.config.gpu_platform_id,
                    "gpu_device_id": self.config.gpu_device_id,
                    "gpu_use_dp": self.config.gpu_use_dp,
                }
            )

        return params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> lgb.Booster:
        """Train LightGBM model."""
        logger.info(f"Training LightGBM model on {self.config.device.upper()}")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = None
        callbacks = []

        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            callbacks = [
                lgb.early_stopping(self.config.early_stopping_rounds),
                lgb.log_evaluation(50),
            ]

        # Train model
        logger.info(f"Starting LightGBM training with device: {self._get_params().get('device', 'cpu')}")
        start_time = time.time()
        self.model = lgb.train(
            self._get_params(),
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=[valid_data] if valid_data else None,
            callbacks=callbacks,
        )

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Model trained with {self.model.num_trees()} trees")

        # Store feature importance
        self.feature_importance_ = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": self.model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)

        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model = lgb.Booster(model_file=str(path))
        logger.info(f"Model loaded from {path}")


class SurrogateModelPipeline:
    """Main pipeline for training surrogate models."""

    def __init__(
        self,
        data_loader: DataLoader,
        model_trainer: ModelTrainer,
        data_config: DataConfig,
        experiment_config: ExperimentConfig,
    ):
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        self.data_config = data_config
        self.experiment_config = experiment_config
        self.scaler: StandardScaler | RobustScaler | None = None

    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features if configured."""
        if not self.data_config.scale_features:
            return X_train, X_test

        if self.data_config.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        return X_train_scaled, X_test_scaled

    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "test") -> dict[str, float]:
        """Evaluate model performance."""
        metrics = {
            f"{prefix}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
            f"{prefix}_r2": r2_score(y_true, y_pred),
            f"{prefix}_mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        }

        # Add correlation coefficient
        if len(y_true) > 1:
            correlation, p_value = stats.pearsonr(y_true, y_pred)
            metrics[f"{prefix}_correlation"] = correlation
            metrics[f"{prefix}_p_value"] = p_value

        return metrics

    def _plot_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_importance: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """Generate and save result plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Actual vs Predicted scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Values")

        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()
        plt.savefig(output_dir / "actual_vs_predicted.png", dpi=300)
        plt.close()

        # 2. Residual plot
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        plt.tight_layout()
        plt.savefig(output_dir / "residuals.png", dpi=300)
        plt.close()

        # 3. Feature importance plot (top 20)
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance.head(20)
        ax.barh(range(len(top_features)), top_features["importance"])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=300)
        plt.close()

        # 4. Prediction error distribution
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.hist(residuals, bins=50, edgecolor="black")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        ax.set_title("Prediction Error Distribution")
        ax.axvline(x=0, color="r", linestyle="--", lw=2)
        plt.tight_layout()
        plt.savefig(output_dir / "error_distribution.png", dpi=300)
        plt.close()

    def train(self, target: str, use_phenotypes: bool = False) -> dict[str, Any]:
        """Train a surrogate model for the specified target."""
        logger.info(f"Starting training pipeline for target: {target}")

        # Set up MLflow
        try:
            mlflow.set_tracking_uri(self.experiment_config.tracking_uri)
            mlflow.set_experiment(self.experiment_config.experiment_name)
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Proceeding without MLflow tracking.")

        run_name = self.experiment_config.run_name or f"{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create a dummy context manager for when MLflow is not available
        class DummyMLflowRun:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        try:
            mlflow_run = mlflow.start_run(run_name=run_name)
        except Exception as e:
            logger.warning(f"MLflow run creation failed: {e}. Proceeding without tracking.")
            mlflow_run = DummyMLflowRun()

        with mlflow_run:
            # Log parameters
            safe_mlflow_log_params(
                {
                    "target": target,
                    "use_phenotypes": use_phenotypes,
                    **{f"data_{k}": v for k, v in self.data_config.__dict__.items() if not isinstance(v, dict)},
                    **{f"model_{k}": v for k, v in self.model_trainer.config.__dict__.items()},
                }
            )

            # Load and prepare data
            X, y = self.data_loader.prepare_training_data(target, use_phenotypes)

            # Train-test split
            split_idx = int(len(X) * self.data_config.train_test_split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Scale features
            X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)

            # Further split training data for validation
            val_split_idx = int(len(X_train_scaled) * 0.8)
            X_train_final = X_train_scaled[:val_split_idx]
            X_val = X_train_scaled[val_split_idx:]
            y_train_final = y_train[:val_split_idx]
            y_val = y_train[val_split_idx:]

            # Train model
            start_time = time.time()
            model = self.model_trainer.train(X_train_final, y_train_final, X_val, y_val)
            training_time = time.time() - start_time
            safe_mlflow_log_metric("training_time", training_time)

            # Make predictions
            y_pred_train = self.model_trainer.predict(X_train_scaled)
            y_pred_test = self.model_trainer.predict(X_test_scaled)

            # Evaluate model
            train_metrics = self._evaluate_model(y_train, y_pred_train, "train")
            test_metrics = self._evaluate_model(y_test, y_pred_test, "test")

            # Log metrics
            safe_mlflow_log_metrics({**train_metrics, **test_metrics})

            # Create output directory
            output_dir = Path(f"models/{target}/{run_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_path = output_dir / "model.txt"
            self.model_trainer.save(model_path)

            # Save scaler
            if self.scaler:
                scaler_path = output_dir / "scaler.joblib"
                joblib.dump(self.scaler, scaler_path)
                mlflow.log_artifact(str(scaler_path))

            # Generate plots
            self._plot_results(y_test, y_pred_test, self.model_trainer.feature_importance_, output_dir)

            # Log artifacts
            if self.experiment_config.log_models:
                mlflow.lightgbm.log_model(model, "model")

            if self.experiment_config.log_plots:
                for plot_file in output_dir.glob("*.png"):
                    mlflow.log_artifact(str(plot_file))

            # Save feature importance
            feature_importance_path = output_dir / "feature_importance.csv"
            self.model_trainer.feature_importance_.to_csv(feature_importance_path, index=False)
            mlflow.log_artifact(str(feature_importance_path))

            # Perform cross-validation
            if self.data_config.validation_folds > 1:
                logger.info("Performing time series cross-validation...")
                tscv = TimeSeriesSplit(n_splits=self.data_config.validation_folds)
                cv_scores = []

                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
                    X_fold_train = X_train_scaled.iloc[train_idx]
                    X_fold_val = X_train_scaled.iloc[val_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    y_fold_val = y_train.iloc[val_idx]

                    # Train fold model
                    fold_trainer = LightGBMTrainer(self.model_trainer.config)
                    fold_trainer.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

                    # Evaluate fold
                    y_fold_pred = fold_trainer.predict(X_fold_val)
                    fold_score = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
                    cv_scores.append(fold_score)

                    mlflow.log_metric(f"cv_fold_{fold}_rmse", fold_score)

                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                mlflow.log_metrics({"cv_rmse_mean": cv_mean, "cv_rmse_std": cv_std})
                logger.info(f"Cross-validation RMSE: {cv_mean:.4f} (+/- {cv_std:.4f})")

            results = {
                "model": model,
                "scaler": self.scaler,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_importance": self.model_trainer.feature_importance_,
                "output_dir": output_dir,
                "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else "no_mlflow",
            }

            logger.info(f"Training completed. Test RMSE: {test_metrics['test_rmse']:.4f}")
            return results


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for LightGBM surrogate model training."""
    # Check GPU availability at startup
    check_gpu_availability()

    parser = argparse.ArgumentParser(description="Train LightGBM surrogate models")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target variable to predict (e.g., energy_consumption, plant_growth)",
    )
    parser.add_argument(
        "--use-phenotypes",
        action="store_true",
        help="Include phenotype features in training",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for training (requires LightGBM GPU build)",
    )
    parser.add_argument("--n-estimators", type=int, default=1000, help="Number of boosting iterations")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="LightGBM_Surrogate_Models",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    # Create configurations
    data_config = DataConfig(use_phenotypes=args.use_phenotypes)

    model_config = ModelConfig(
        device="gpu" if args.gpu else "cpu",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )

    experiment_config = ExperimentConfig(experiment_name=args.experiment_name)

    # Create database connection
    db_url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', 'postgres')}@"
        f"{os.getenv('DB_HOST', 'db')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )

    # Create components
    data_loader = PostgreSQLDataLoader(db_url, data_config)
    model_trainer = LightGBMTrainer(model_config)

    # Create and run pipeline
    pipeline = SurrogateModelPipeline(
        data_loader=data_loader,
        model_trainer=model_trainer,
        data_config=data_config,
        experiment_config=experiment_config,
    )

    try:
        results = pipeline.train(args.target, args.use_phenotypes)
        logger.info(f"Model saved to: {results['output_dir']}")
        logger.info(f"MLflow run ID: {results['mlflow_run_id']}")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
