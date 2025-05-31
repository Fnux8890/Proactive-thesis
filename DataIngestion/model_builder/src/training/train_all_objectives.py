#!/usr/bin/env python
"""Train surrogate models for all MOEA objectives.

This script orchestrates the training of multiple LightGBM models,
one for each optimization objective defined in the MOEA configuration.
It ensures all models are trained consistently and ready for use in
multi-objective optimization.

Usage:
    python train_all_objectives.py
    python train_all_objectives.py --objectives energy_consumption plant_growth
    python train_all_objectives.py --composite sustainable_production
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.moea_objectives import COMPOSITE_OBJECTIVES, OBJECTIVES
from src.training.train_lightgbm_surrogate import (
    DataConfig,
    ExperimentConfig,
    LightGBMTrainer,
    ModelConfig,
    PostgreSQLDataLoader,
    SurrogateModelPipeline,
    check_gpu_availability,
)
from src.utils import MultiLevelDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ObjectiveModelOrchestrator:
    """Orchestrates training of models for multiple objectives."""

    def __init__(
        self,
        db_url: str,
        use_gpu: bool = True,
        use_phenotypes: bool = True,
        experiment_name: str = "MOEA_Surrogate_Models",
    ):
        self.db_url = db_url
        self.use_gpu = use_gpu
        self.use_phenotypes = use_phenotypes
        self.experiment_name = experiment_name
        self.results: dict[str, dict] = {}
        
        # Log initialization settings
        logger.info(f"ObjectiveModelOrchestrator initialized:")
        logger.info(f"  - GPU enabled: {self.use_gpu}")
        logger.info(f"  - Use phenotypes: {self.use_phenotypes}")
        logger.info(f"  - Experiment: {self.experiment_name}")

    def train_objective(self, objective_name: str) -> dict:
        """Train a model for a specific objective."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training model for objective: {objective_name}")
        logger.info(f"{'=' * 60}")

        # Get objective configuration
        if objective_name in OBJECTIVES:
            objective = OBJECTIVES[objective_name]
        elif objective_name in COMPOSITE_OBJECTIVES:
            objective = COMPOSITE_OBJECTIVES[objective_name]
        else:
            raise ValueError(f"Unknown objective: {objective_name}")

        # Create configurations
        data_config = DataConfig(use_phenotypes=self.use_phenotypes and bool(objective.related_phenotypes))

        # Check if multi-level features should be used
        use_multi_level = os.getenv("USE_MULTI_LEVEL_FEATURES", "false").lower() == "true"
        feature_tables = os.getenv("FEATURE_TABLES", "").split(",") if os.getenv("FEATURE_TABLES") else None
        
        if use_multi_level:
            logger.info("Using multi-level feature extraction")
            if feature_tables:
                data_config.feature_tables = feature_tables
                logger.info(f"Feature tables: {feature_tables}")

        # Adjust model parameters based on objective
        model_config = ModelConfig(
            device="gpu" if self.use_gpu else "cpu",
            n_estimators=1500 if objective_name == "plant_growth" else 1000,
            learning_rate=0.03 if objective_name == "plant_growth" else 0.05,
            num_leaves=63 if objective_name in ["plant_growth", "crop_quality"] else 31,
            min_child_samples=10 if objective_name == "climate_stability" else 20,
        )
        
        logger.info(f"Model configuration: device={model_config.device}, n_estimators={model_config.n_estimators}")

        experiment_config = ExperimentConfig(
            experiment_name=self.experiment_name,
            run_name=f"{objective_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
        )

        # Create components
        if use_multi_level:
            data_loader = MultiLevelDataLoader(self.db_url, data_config)
        else:
            data_loader = PostgreSQLDataLoader(self.db_url, data_config)
        model_trainer = LightGBMTrainer(model_config)

        # Create and run pipeline
        pipeline = SurrogateModelPipeline(
            data_loader=data_loader,
            model_trainer=model_trainer,
            data_config=data_config,
            experiment_config=experiment_config,
        )

        try:
            results = pipeline.train(objective_name, self.use_phenotypes)
            self.results[objective_name] = results

            # Log objective-specific information
            logger.info(f"Model trained successfully for {objective_name}")
            logger.info(f"Test RMSE: {results['test_metrics']['test_rmse']:.4f}")
            logger.info(f"Test R²: {results['test_metrics']['test_r2']:.4f}")

            # Save objective metadata
            metadata = {
                "objective_name": objective_name,
                "description": objective.description,
                "type": objective.type.value,
                "unit": objective.unit,
                "important_signals": objective.important_signals,
                "related_phenotypes": [p.value for p in objective.related_phenotypes],
                "model_path": str(results["output_dir"] / "model.txt"),
                "mlflow_run_id": results["mlflow_run_id"],
                "test_metrics": results["test_metrics"],
                "training_timestamp": datetime.now().isoformat(),
            }

            metadata_path = results["output_dir"] / "objective_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return results

        except Exception as e:
            logger.error(f"Failed to train model for {objective_name}: {e}")
            raise

    def train_all_objectives(self, objectives: list[str]) -> None:
        """Train models for all specified objectives."""
        logger.info(f"Starting training for {len(objectives)} objectives")

        successful = []
        failed = []

        for objective in objectives:
            try:
                self.train_objective(objective)
                successful.append(objective)
            except Exception as e:
                logger.error(f"Failed to train {objective}: {e}")
                failed.append(objective)

        # Summary report
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Successful: {len(successful)} models")
        for obj in successful:
            if obj in self.results:
                metrics = self.results[obj]["test_metrics"]
                logger.info(f"  - {obj}: RMSE={metrics['test_rmse']:.4f}, R²={metrics['test_r2']:.4f}")

        if failed:
            logger.info(f"\nFailed: {len(failed)} models")
            for obj in failed:
                logger.info(f"  - {obj}")

        # Save summary
        self._save_summary(successful, failed)

    def _save_summary(self, successful: list[str], failed: list[str]) -> None:
        """Save training summary to file."""
        summary_dir = Path("models/training_summaries")
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": self.experiment_name,
            "use_gpu": self.use_gpu,
            "use_phenotypes": self.use_phenotypes,
            "successful_models": successful,
            "failed_models": failed,
            "model_details": {},
        }

        for obj in successful:
            if obj in self.results:
                summary["model_details"][obj] = {
                    "output_dir": str(self.results[obj]["output_dir"]),
                    "mlflow_run_id": self.results[obj]["mlflow_run_id"],
                    "test_metrics": self.results[obj]["test_metrics"],
                }

        summary_path = summary_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {summary_path}")

    def verify_models(self) -> bool:
        """Verify all trained models are ready for MOEA."""
        logger.info("\nVerifying trained models...")

        all_valid = True
        for objective, results in self.results.items():
            model_path = results["output_dir"] / "model.txt"
            scaler_path = results["output_dir"] / "scaler.joblib"
            metadata_path = results["output_dir"] / "objective_metadata.json"

            if not model_path.exists():
                logger.error(f"Model file missing for {objective}: {model_path}")
                all_valid = False

            if results.get("scaler") and not scaler_path.exists():
                logger.error(f"Scaler file missing for {objective}: {scaler_path}")
                all_valid = False

            if not metadata_path.exists():
                logger.error(f"Metadata file missing for {objective}: {metadata_path}")
                all_valid = False

            if all([model_path.exists(), metadata_path.exists()]):
                logger.info(f"✓ {objective} model verified")

        return all_valid


def main():
    """Main entry point."""
    # Check GPU availability at startup
    check_gpu_availability()
    
    parser = argparse.ArgumentParser(description="Train surrogate models for all MOEA objectives")
    parser.add_argument(
        "--objectives",
        nargs="+",
        help="Specific objectives to train (default: all standard objectives)",
    )
    parser.add_argument(
        "--composite",
        type=str,
        help="Train a composite objective (e.g., sustainable_production)",
    )
    parser.add_argument(
        "--skip-phenotypes",
        action="store_true",
        help="Skip phenotype features in training",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="MOEA_Surrogate_Models",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    # Determine objectives to train
    if args.composite:
        if args.composite not in COMPOSITE_OBJECTIVES:
            logger.error(f"Unknown composite objective: {args.composite}")
            logger.info(f"Available: {list(COMPOSITE_OBJECTIVES.keys())}")
            return 1
        objectives_to_train = [args.composite]
    elif args.objectives:
        # Validate objectives
        all_objectives = list(OBJECTIVES.keys()) + list(COMPOSITE_OBJECTIVES.keys())
        invalid = [obj for obj in args.objectives if obj not in all_objectives]
        if invalid:
            logger.error(f"Unknown objectives: {invalid}")
            logger.info(f"Available: {all_objectives}")
            return 1
        objectives_to_train = args.objectives
    else:
        # Default: train all standard objectives
        objectives_to_train = list(OBJECTIVES.keys())

    # Create database connection string
    db_url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', 'postgres')}@"
        f"{os.getenv('DB_HOST', 'db')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )

    # Create orchestrator - default to GPU unless --cpu flag is used
    use_gpu = not args.cpu
    if use_gpu:
        logger.info("GPU mode enabled (default). Use --cpu flag to disable GPU.")
    else:
        logger.info("CPU mode enabled (--cpu flag specified).")
    
    orchestrator = ObjectiveModelOrchestrator(
        db_url=db_url,
        use_gpu=use_gpu,
        use_phenotypes=not args.skip_phenotypes,
        experiment_name=args.experiment_name,
    )

    try:
        # Train all objectives
        orchestrator.train_all_objectives(objectives_to_train)

        # Verify models
        if orchestrator.verify_models():
            logger.info("\nAll models verified and ready for MOEA optimization!")
            return 0
        else:
            logger.error("\nSome models failed verification")
            return 1

    except Exception as e:
        logger.error(f"Training orchestration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
