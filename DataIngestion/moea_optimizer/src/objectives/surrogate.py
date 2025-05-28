"""Surrogate model-based objective functions."""

import logging
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from .base import ObjectiveFunction, ObjectiveType

logger = logging.getLogger(__name__)


class SurrogateObjective(ObjectiveFunction):
    """Objective function based on a trained surrogate model."""

    def __init__(
        self,
        name: str,
        description: str,
        obj_type: ObjectiveType,
        unit: str,
        model_path: str | Path,
        scaler_path: str | Path | None = None,
        feature_columns: list[str] | None = None,
        weight_range: tuple = (0.0, 1.0)
    ):
        super().__init__(name, description, obj_type, unit, weight_range)

        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.feature_columns = feature_columns

        # Load model and scaler
        self.model = self._load_model()
        self.scaler = self._load_scaler() if self.scaler_path else None

        # Model metadata
        self._feature_importance = None
        self._model_type = None

    def _load_model(self) -> Any:
        """Load the surrogate model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info(f"Loading surrogate model for '{self.name}' from {self.model_path}")

        # Determine model type and load accordingly
        if self.model_path.suffix == '.txt':
            # LightGBM model
            model = lgb.Booster(model_file=str(self.model_path))
            self._model_type = 'lightgbm'
            self._feature_importance = model.feature_importance(importance_type='gain')
            if not self.feature_columns:
                self.feature_columns = model.feature_name()
        elif self.model_path.suffix in ['.pkl', '.joblib']:
            # Scikit-learn or other pickled model
            model = joblib.load(self.model_path)
            self._model_type = 'sklearn'
        else:
            raise ValueError(f"Unsupported model file type: {self.model_path.suffix}")

        return model

    def _load_scaler(self) -> Any:
        """Load the feature scaler."""
        if not self.scaler_path.exists():
            logger.warning(f"Scaler file not found: {self.scaler_path}")
            return None

        logger.info(f"Loading scaler from {self.scaler_path}")
        return joblib.load(self.scaler_path)

    def _prepare_features(self, decision_variables: dict[str, float], context: dict[str, Any] | None = None) -> np.ndarray:
        """Prepare feature vector from decision variables and context."""
        # Start with decision variables
        features = decision_variables.copy()

        # Add context features if provided
        if context:
            # Extract relevant context features
            if 'weather' in context:
                weather = context['weather']
                features.update({
                    'outside_temp_c': weather.get('temperature', 20.0),
                    'outside_light_w_m2': weather.get('solar_radiation', 0.0),
                    'wind_speed_m_s': weather.get('wind_speed', 0.0)
                })

            if 'time_info' in context:
                time_info = context['time_info']
                features.update({
                    'hour_of_day': time_info.get('hour', 12),
                    'day_of_year': time_info.get('day_of_year', 180)
                })

            if 'plant_state' in context:
                plant_state = context['plant_state']
                features.update({
                    'plant_age_days': plant_state.get('age_days', 30),
                    'growth_stage': plant_state.get('growth_stage', 0.5)
                })

        # Add derived features
        if 'temperature_setpoint' in features and 'humidity_setpoint' in features:
            # Calculate VPD (Vapor Pressure Deficit)
            temp = features['temperature_setpoint']
            rh = features['humidity_setpoint']
            svp = 0.611 * np.exp(17.27 * temp / (temp + 237.3))  # Saturation vapor pressure (kPa)
            vpd = svp * (1 - rh / 100)
            features['vpd_calculated'] = vpd

        if 'light_intensity' in features and 'light_hours' in features:
            # Calculate Daily Light Integral (DLI)
            # DLI (mol/m²/day) = light intensity (μmol/m²/s) x hours x 3600 / 1,000,000
            dli = features['light_intensity'] * features['light_hours'] * 3.6 / 1000
            features['dli_calculated'] = dli

        # Convert to feature array in correct order
        if self.feature_columns:
            # Use specified feature order
            feature_array = []
            for col in self.feature_columns:
                if col in features:
                    feature_array.append(features[col])
                else:
                    # Use default value or raise error
                    logger.warning(f"Feature '{col}' not found in input, using 0.0")
                    feature_array.append(0.0)
            feature_array = np.array(feature_array).reshape(1, -1)
        else:
            # Use all features in sorted order
            feature_array = np.array([features[k] for k in sorted(features.keys())]).reshape(1, -1)

        return feature_array

    def evaluate(self, decision_variables: dict[str, float], context: dict[str, Any] | None = None) -> float:
        """Evaluate the objective using the surrogate model."""
        # Prepare features
        features = self._prepare_features(decision_variables, context)

        # Scale features if scaler is available
        if self.scaler:
            features = self.scaler.transform(features)

        # Make prediction
        if self._model_type == 'lightgbm':
            prediction = self.model.predict(features)[0]
        else:
            prediction = self.model.predict(features)[0]

        # Ensure prediction is a scalar
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()

        return float(prediction)

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance from the model."""
        if self._feature_importance is None:
            return None

        if self.feature_columns:
            return dict(zip(self.feature_columns, self._feature_importance, strict=False))
        else:
            return {"feature_" + str(i): imp for i, imp in enumerate(self._feature_importance)}

    def validate_prediction_range(self, n_samples: int = 1000) -> dict[str, float]:
        """Validate the prediction range of the surrogate model."""
        predictions = []

        for _ in range(n_samples):
            # Generate random decision variables within bounds
            random_vars = {}
            if hasattr(self, 'bounds'):
                for var, (low, high) in self.bounds.items():
                    random_vars[var] = np.random.uniform(low, high)
            else:
                # Use some default bounds for testing
                for var in ['temperature_setpoint', 'humidity_setpoint', 'co2_setpoint', 'light_intensity']:
                    if var == 'temperature_setpoint':
                        random_vars[var] = np.random.uniform(18, 28)
                    elif var == 'humidity_setpoint':
                        random_vars[var] = np.random.uniform(60, 85)
                    elif var == 'co2_setpoint':
                        random_vars[var] = np.random.uniform(400, 1000)
                    elif var == 'light_intensity':
                        random_vars[var] = np.random.uniform(0, 600)

            pred = self.evaluate(random_vars)
            predictions.append(pred)

        predictions = np.array(predictions)

        return {
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "q25": float(np.percentile(predictions, 25)),
            "q50": float(np.percentile(predictions, 50)),
            "q75": float(np.percentile(predictions, 75))
        }
