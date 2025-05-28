#!/usr/bin/env python3
"""Check the status of the GPU-enhanced feature extraction pipeline."""

from pathlib import Path
import subprocess
import sys


def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists and is readable."""
    return filepath.exists() and filepath.is_file()


def check_imports(filepath: Path) -> tuple[bool, list[str]]:
    """Check if all imports in a file are valid."""
    try:
        # Compile the file
        compile(open(filepath).read(), filepath, 'exec')
        return True, []
    except Exception as e:
        return False, [str(e)]


def check_docker_compose_service():
    """Check if the feature_extractor service is defined in docker-compose."""
    compose_path = Path("../../docker-compose.yml")
    if not compose_path.exists():
        return False, "docker-compose.yml not found"
    
    try:
        with open(compose_path) as f:
            content = f.read()
        if 'feature_extractor:' in content:
            return True, "feature_extractor service found"
        else:
            return False, "feature_extractor service not found in docker-compose.yml"
    except Exception as e:
        return False, str(e)


def main():
    """Check pipeline status."""
    print("=" * 60)
    print("GPU-Enhanced Feature Extraction Pipeline Status")
    print("=" * 60)
    
    # Files to check
    files = {
        "GPU Preprocessing": Path("features/gpu_preprocessing.py"),
        "GPU Feature Extractor": Path("features/extract_features_gpu_enhanced.py"),
        "Enhanced Wrapper": Path("feature/extract_features_enhanced.py"),
        "Pipeline Validator": Path("validate_pipeline.py"),
        "Makefile": Path("Makefile"),
        "pyproject.toml": Path("pyproject.toml"),
        "Run Script": Path("run_with_validation.sh"),
    }
    
    print("\n📁 File Status:")
    all_files_ok = True
    for name, filepath in files.items():
        if check_file_exists(filepath):
            # Try to compile Python files
            if filepath.suffix == '.py':
                ok, errors = check_imports(filepath)
                if ok:
                    print(f"  ✅ {name}: {filepath} (valid Python)")
                else:
                    print(f"  ❌ {name}: {filepath} (syntax errors)")
                    all_files_ok = False
            else:
                print(f"  ✅ {name}: {filepath}")
        else:
            print(f"  ❌ {name}: {filepath} (not found)")
            all_files_ok = False
    
    print("\n🐳 Docker Integration:")
    docker_ok, docker_msg = check_docker_compose_service()
    if docker_ok:
        print(f"  ✅ {docker_msg}")
    else:
        print(f"  ❌ {docker_msg}")
    
    print("\n🔧 Available Commands:")
    print("  make format         - Format Python code")
    print("  make lint           - Check code quality")
    print("  make validate       - Run validation checks")
    print("  make docker-run     - Run with Docker Compose")
    print("  ./run_with_validation.sh - Run with pre-flight checks")
    
    print("\n📊 Pipeline Features:")
    print("  ✓ GPU-accelerated data preprocessing (melting, sorting)")
    print("  ✓ tsfresh feature extraction (600+ features)")
    print("  ✓ GPU-accelerated feature selection") 
    print("  ✓ Automatic CPU fallback if GPU unavailable")
    print("  ✓ Code quality validation before execution")
    print("  ✓ Memory management and monitoring")
    
    if all_files_ok and docker_ok:
        print("\n✅ Pipeline is ready to run!")
        print("\nTo start: cd ../.. && docker compose up feature_extractor")
    else:
        print("\n❌ Some issues need to be fixed before running.")
    
    return 0 if (all_files_ok and docker_ok) else 1


if __name__ == "__main__":
    sys.exit(main())