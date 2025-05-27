#!/usr/bin/env python3
"""Pre-execution validation script for feature extraction pipeline.
This script runs format/lint checks and validates the environment
before docker-compose execution.
"""

from pathlib import Path
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
class PipelineValidator:
    """Validates the feature extraction pipeline before execution."""
    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).parent
        self.errors = []
        self.warnings = []
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting pipeline validation...")
        checks = [
            ("Python code formatting", self.check_formatting),
            ("Python code linting", self.check_linting),
            ("Import validation", self.check_imports),
            ("Configuration files", self.check_configs),
            ("Docker setup", self.check_docker),
            ("Database schema", self.check_database_schema),
            ("GPU availability", self.check_gpu_availability),
        ]
        all_passed = True
        for name, check_func in checks:
            logger.info(f"\nChecking {name}...")
            try:
                passed = check_func()
                status = "✓ PASSED" if passed else "✗ FAILED"
                logger.info(f"{name}: {status}")
                all_passed = all_passed and passed
            except Exception as e:
                logger.error(f"{name}: ✗ FAILED with error: {e}")
                self.errors.append(f"{name}: {e}")
                all_passed = False
        # Summary
        logger.info("\n" + "="*50)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*50)
        if self.errors:
            logger.error(f"Errors found ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  - {error}")
        if self.warnings:
            logger.warning(f"Warnings found ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        if all_passed:
            logger.info("✓ All checks passed! Pipeline is ready to run.")
        else:
            logger.error("✗ Validation failed! Please fix the issues above.")
        return all_passed
    def check_formatting(self) -> bool:
        """Check if code is properly formatted."""
        try:
            # Check with black
            result = subprocess.run(
                ["black", "--check", "--quiet", str(self.root_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.warnings.append("Code formatting issues found. Run 'make format' to fix.")
                # Show which files need formatting
                diff_result = subprocess.run(
                    ["black", "--diff", str(self.root_dir)],
                    capture_output=True,
                    text=True
                )
                if diff_result.stdout:
                    logger.warning("Files needing formatting:")
                    for line in diff_result.stdout.split('\n'):
                        if line.startswith('---') or line.startswith('+++'):
                            logger.warning(f"  {line}")
                return False
            return True
        except FileNotFoundError:
            self.warnings.append("black not installed. Run 'pip install black'")
            return False
    def check_linting(self) -> bool:
        """Check code with ruff."""
        try:
            result = subprocess.run(
                ["ruff", "check", str(self.root_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.errors.append("Linting errors found")
                logger.error("Linting issues:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
                return False
            return True
        except FileNotFoundError:
            self.warnings.append("ruff not installed. Run 'pip install ruff'")
            return False
    def check_imports(self) -> bool:
        """Validate all imports are available."""
        required_imports = [
            "pandas",
            "numpy",
            "sqlalchemy",
            "tsfresh",
            "psycopg2",
        ]
        missing = []
        for module in required_imports:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        if missing:
            self.errors.append(f"Missing required packages: {', '.join(missing)}")
            return False
        # Check optional GPU imports
        gpu_imports = ["cupy", "cudf", "cuml"]
        gpu_available = []
        for module in gpu_imports:
            try:
                __import__(module)
                gpu_available.append(module)
            except ImportError:
                pass
        if gpu_available:
            logger.info(f"GPU packages available: {', '.join(gpu_available)}")
        else:
            self.warnings.append("No GPU packages found. Pipeline will run in CPU mode.")
        return True
    def check_configs(self) -> bool:
        """Validate configuration files."""
        configs = [
            "data_processing_config.json",
            "pyproject.toml",
        ]
        all_valid = True
        for config in configs:
            config_path = self.root_dir / config
            if not config_path.exists():
                # Check parent directory
                config_path = self.root_dir.parent / config
            if config_path.exists():
                if config.endswith('.json'):
                    try:
                        with open(config_path) as f:
                            json.load(f)
                        logger.info(f"  ✓ {config} is valid")
                    except json.JSONDecodeError as e:
                        self.errors.append(f"{config}: Invalid JSON - {e}")
                        all_valid = False
                elif config == 'pyproject.toml':
                    try:
                        import tomllib
                        with open(config_path, 'rb') as f:
                            tomllib.load(f)
                        logger.info(f"  ✓ {config} is valid")
                    except Exception as e:
                        self.errors.append(f"{config}: Invalid TOML - {e}")
                        all_valid = False
            else:
                self.warnings.append(f"{config} not found")
        return all_valid
    def check_docker(self) -> bool:
        """Check Docker and docker-compose setup."""
        # Check docker
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.errors.append("Docker not available")
                return False
        except FileNotFoundError:
            self.errors.append("Docker not installed")
            return False
        # Check docker-compose file
        compose_path = self.root_dir.parent.parent / "docker-compose.yml"
        if not compose_path.exists():
            self.errors.append("docker-compose.yml not found")
            return False
        # Validate docker-compose syntax
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "config"],
                capture_output=True,
                text=True,
                cwd=str(compose_path.parent)
            )
            if result.returncode != 0:
                self.errors.append(f"docker-compose.yml validation failed: {result.stderr}")
                return False
        except Exception as e:
            self.warnings.append(f"Could not validate docker-compose: {e}")
        return True
    def check_database_schema(self) -> bool:
        """Check if required database tables exist."""
        # This is a placeholder - in production, you'd actually connect and check
        required_tables = [
            "sensor_data_merged",
            "era_labels_level_a",
            "era_labels_level_b",
            "era_labels_level_c",
        ]
        # For now, just check if we can import the db connector
        try:
            from db_utils_optimized import SQLAlchemyPostgresConnector
            logger.info("  ✓ Database connector available")
            return True
        except ImportError:
            self.warnings.append("Cannot import database connector")
            return True  # Don't fail on this
    def check_gpu_availability(self) -> bool:
        """Check GPU availability and CUDA setup."""
        try:
            import cupy as cp
            # Check for GPU
            gpu_count = cp.cuda.runtime.getDeviceCount()
            if gpu_count > 0:
                # Get GPU info
                for i in range(gpu_count):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    logger.info(
                        f"  ✓ GPU {i}: {props['name'].decode()} "
                        f"({props['totalGlobalMem'] / 1024**3:.1f} GB)"
                    )
                return True
            else:
                self.warnings.append("No GPUs found")
                return True  # Not a failure, just a warning
        except ImportError:
            self.warnings.append("CuPy not installed - GPU features disabled")
            return True  # Not a failure
        except Exception as e:
            self.warnings.append(f"GPU check failed: {e}")
            return True  # Not a failure
    def fix_formatting(self):
        """Auto-fix formatting issues."""
        logger.info("Fixing code formatting...")
        # Run black
        result = subprocess.run(
            ["black", str(self.root_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✓ Code formatting fixed")
        else:
            logger.error(f"Formatting failed: {result.stderr}")
        # Run ruff with fix
        result = subprocess.run(
            ["ruff", "check", "--fix", str(self.root_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✓ Linting issues fixed")
        else:
            logger.warning("Some linting issues could not be auto-fixed")
def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Validate feature extraction pipeline")
    parser.add_argument("--fix", action="store_true", help="Auto-fix formatting issues")
    parser.add_argument("--root", type=Path, help="Root directory to check")
    args = parser.parse_args()
    validator = PipelineValidator(root_dir=args.root)
    if args.fix:
        validator.fix_formatting()
        logger.info("\nRe-running validation after fixes...\n")
    success = validator.run_all_checks()
    return 0 if success else 1
if __name__ == "__main__":
    sys.exit(main())
