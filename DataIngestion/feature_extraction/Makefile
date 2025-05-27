# Makefile for Feature Extraction Pipeline
.PHONY: help format lint validate clean install docker-build docker-run docker-validate \
        test test-all test-backend test-thread-safety test-observability \
        test-performance test-sql-safety test-integration test-isolated test-gpu \
        test-soak status summary logs build

# Default target
help:
	@echo "Feature Extraction Pipeline - Available commands:"
	@echo ""
	@echo "Code Quality & Validation:"
	@echo "  make format          - Format code with black and ruff"
	@echo "  make lint            - Check code with ruff"
	@echo "  make validate        - Run all validation checks"
	@echo "  make validate-fix    - Validate and auto-fix issues"
	@echo "  make clean           - Clean temporary files"
	@echo "  make install         - Install dependencies"
	@echo "  make install-gpu     - Install GPU dependencies"
	@echo ""
	@echo "Docker Operations:"
	@echo "  make docker-build    - Build docker image"
	@echo "  make docker-run      - Run feature extraction in docker"
	@echo "  make docker-validate - Validate inside docker container"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test              - Run all tests"
	@echo "  make test-backend      - Test backend adapter"
	@echo "  make test-thread-safety - Test connection pool thread safety"
	@echo "  make test-observability - Test metrics and observability"
	@echo "  make test-performance  - Test performance optimizations"
	@echo "  make test-sql-safety   - Test SQL injection prevention"
	@echo "  make test-integration  - Run integration tests"
	@echo "  make test-isolated     - Run tests without dependencies"
	@echo "  make test-gpu          - Run GPU backend tests"
	@echo "  make test-soak         - Run long-running stability tests"
	@echo "  make status            - Show test container status"
	@echo "  make summary           - Generate test summary report"
	@echo "  make logs SERVICE=name - Show logs for a service"
	@echo "  make build             - Build test containers"

# Format code
format:
	@echo "Formatting Python code..."
	black .
	ruff check --fix .
	@echo "✓ Code formatted"

# Lint code
lint:
	@echo "Linting Python code..."
	ruff check .
	@echo "✓ Linting complete"

# Run validation
validate:
	@echo "Running pipeline validation..."
	python validate_pipeline.py

# Validate and fix
validate-fix:
	@echo "Running validation with auto-fix..."
	python validate_pipeline.py --fix

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
	@echo "✓ Cleaned"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install ruff black mypy
	@echo "✓ Dependencies installed"

# Install GPU dependencies
install-gpu:
	@echo "Installing GPU dependencies..."
	pip install -r requirements-gpu.txt
	@echo "✓ GPU dependencies installed"

# Docker operations
docker-build:
	@echo "Building Docker image..."
	cd ../.. && docker compose build feature_extractor
	@echo "✓ Docker image built"

# Run with docker-compose
docker-run:
	@echo "Running feature extraction with Docker..."
	cd ../.. && docker compose run --rm feature_extractor make validate && \
	docker compose up feature_extractor
	@echo "✓ Feature extraction complete"

# Validate inside docker
docker-validate:
	@echo "Running validation inside Docker..."
	cd ../.. && docker compose run --rm feature_extractor make validate
	@echo "✓ Docker validation complete"

# Quick check before commit
pre-commit: format lint validate
	@echo "✓ Pre-commit checks passed"

# Watch for changes and auto-format
watch:
	@echo "Watching for changes..."
	@while true; do \
		inotifywait -r -e modify,create,delete --exclude '__pycache__|.git' . 2>/dev/null && \
		make format; \
	done

# Run all tests
test: test-all

test-all:
	@echo "🧪 Running all tests..."
	@./run_tests.sh all

# Individual test targets
test-backend:
	@echo "🔧 Testing backend adapter..."
	@./run_tests.sh backend

test-thread-safety:
	@echo "🔒 Testing thread safety..."
	@./run_tests.sh thread-safety

test-observability:
	@echo "📊 Testing observability..."
	@./run_tests.sh observability

test-performance:
	@echo "⚡ Testing performance..."
	@./run_tests.sh performance

test-sql-safety:
	@echo "🛡️ Testing SQL safety..."
	@./run_tests.sh sql-safety

test-integration:
	@echo "🔄 Running integration tests..."
	@./run_tests.sh integration

test-isolated:
	@echo "🏝️ Running isolated tests..."
	@./run_tests.sh isolated

test-gpu:
	@echo "🎮 Running GPU tests..."
	@./run_tests.sh gpu

test-soak:
	@echo "⏱️ Running soak tests..."
	@./run_tests.sh soak

# Utility targets
clean:
	@echo "🧹 Cleaning up test environment..."
	@./run_tests.sh clean

status:
	@echo "📋 Test container status:"
	@docker compose -f docker-compose.test.yaml ps

summary:
	@echo "📊 Generating test summary..."
	@python3 test_summary.py

logs:
	@if [ -z "$(SERVICE)" ]; then \
		echo "❌ Please specify SERVICE=<name>"; \
		echo "Available services:"; \
		docker compose -f docker-compose.test.yaml ps --services; \
	else \
		docker compose -f docker-compose.test.yaml logs $(SERVICE); \
	fi

build:
	@echo "🔨 Building test containers..."
	@docker compose -f docker-compose.test.yaml build

# Quick test targets for CI/CD
ci-quick: test-isolated test-backend test-performance
	@echo "✅ Quick CI tests complete"

ci-full: build test-all summary
	@echo "✅ Full CI tests complete"

# Development helpers
dev-watch:
	@echo "👀 Watching for file changes..."
	@while true; do \
		inotifywait -e modify -r backend/ db/ features/ tests/ 2>/dev/null && \
		make test-isolated; \
	done

# Performance profiling
profile:
	@echo "📈 Running performance profiling..."
	@docker compose -f docker-compose.test.yaml run --rm test-performance \
		python3 -m cProfile -o profile.stats tests/test_sentinel_performance.py

# Coverage report
coverage:
	@echo "📊 Generating coverage report..."
	@docker compose -f docker-compose.test.yaml run --rm test-all \
		python3 -m pytest tests/ --cov=backend --cov=db --cov=features --cov-report=html

# Documentation
docs:
	@echo "📚 Test documentation:"
	@echo "  - Testing Guide: TESTING_GUIDE.md"
	@echo "  - Test README: tests/README.md"
	@echo "  - Observability Guide: OBSERVABILITY_GUIDE.md"