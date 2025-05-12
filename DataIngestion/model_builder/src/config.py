# Configuration dataclasses (or Hydra config setup)
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ModelConfig:
    hidden_units: int = 50
    num_layers: int = 2
    # Add other model-specific params like dropout here

@dataclass
class DataConfig:
    sequence_length: int = 24
    batch_size: int = 64
    num_workers: int = 4 # Default, adjust based on hardware
    pin_memory: bool = True
    persistent_workers: bool = True
    # feature_columns: List[str] = field(default_factory=lambda: ['Sensor_A', 'Sensor_B']) # Example
    # target_columns: List[str] = field(default_factory=lambda: ['Sensor_A', 'Sensor_B']) # Example

@dataclass
class TrainerConfig:
    # Removed default max_epochs here, will be set based on GPU tag or CLI arg
    learning_rate: float = 1e-3
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "16-mixed" # Use "32" or "bf16-mixed" as needed
    deterministic: bool = True
    # Add gradient clipping, scheduler options etc. here
    # Add max_epochs here but allow it to be None initially
    max_epochs: int | None = None

@dataclass
class GlobalConfig:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    # Add paths etc.
    data_dir: str = "/data" # Default input dir inside container
    model_dir: str = "/models" # Default output dir inside container
    # Add default epochs per GPU tag
    default_epochs_by_gpu: Dict[str, int] = field(default_factory=lambda: {
        "gtx1660super": 50,   # Default for your 1660 SUPER
        "rtx4070": 150,      # Example for a 4070
        "h100": 500,         # Example for an H100
        "default_gpu": 20,   # Fallback if tag is missing or not listed
        "cpu": 10            # Default if running on CPU
    })

# You could potentially load/override this config from a YAML file using OmegaConf/Hydra 