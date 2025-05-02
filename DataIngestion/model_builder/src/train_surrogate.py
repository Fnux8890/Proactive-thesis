import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting model_builder sanity check (train_surrogate.py)...")

    # --- GPU Check ---
    try:
        import cupy as cp
        import cudf
        logger.info("Successfully imported cupy and cudf.")
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            logger.info(f"Found {device_count} CUDA devices.")
            if device_count > 0:
                 logger.info(f"Device 0: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
            else:
                 logger.warning("No CUDA devices found by cupy!")
            logger.info(f"cuDF version: {cudf.__version__}")
        except cp.cuda.runtime.CUDARuntimeError as e:
            logger.error(f"CUDA runtime error: {e}")
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
    except ImportError as e:
        logger.error(f"Failed to import cupy or cudf: {e}. Check RAPIDS installation and GPU driver.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during GPU check: {e}")
        return

    # --- Volume Checks ---
    data_dir = Path("/data")
    models_dir = Path("/models")

    # Check /data volume (read-only access expected)
    logger.info(f"Checking data directory: {data_dir}")
    if data_dir.exists() and data_dir.is_dir():
        logger.info(f"Contents of {data_dir}:")
        try:
            for item in data_dir.iterdir():
                logger.info(f"- {item.name}")
        except Exception as e:
            logger.error(f"Could not list contents of {data_dir}: {e}")
    else:
        logger.warning(f"{data_dir} does not exist or is not a directory.")

    # Check /models volume (write access expected)
    logger.info(f"Checking models directory: {models_dir}")
    if not models_dir.exists():
        logger.info(f"{models_dir} does not exist, attempting to create.")
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Successfully created {models_dir}.")
        except Exception as e:
            logger.error(f"Failed to create {models_dir}: {e}")
            # Don't proceed with writing if directory creation failed
            logger.info("Model builder sanity check finished with errors.")
            return

    if models_dir.is_dir():
        test_file_path = models_dir / "sanity_check_output.txt"
        logger.info(f"Attempting to write test file to {test_file_path}")
        try:
            with open(test_file_path, "w") as f:
                f.write("Sanity check successful.")
            logger.info(f"Successfully wrote test file: {test_file_path}")
            # Clean up the test file
            # os.remove(test_file_path)
            # logger.info(f"Successfully removed test file: {test_file_path}")
            # Leaving the file for now so you can verify it exists after the run
        except Exception as e:
            logger.error(f"Failed to write or remove test file in {models_dir}: {e}")
    else:
        logger.warning(f"{models_dir} is not a directory, cannot write test file.")


    logger.info("Model builder sanity check finished.")

if __name__ == "__main__":
    main() 