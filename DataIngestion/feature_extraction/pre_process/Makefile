.PHONY: help install install-dev format lint typecheck test test-unit test-integration test-watch ci clean time

# Default target
.DEFAULT_GOAL := help

# Colors for output
COLOR_RESET   = \033[0m
COLOR_GREEN   = \033[32m
COLOR_YELLOW  = \033[33m
COLOR_BLUE    = \033[34m

## Help
help:
	@echo "$(COLOR_BLUE)Available targets:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)install$(COLOR_RESET)       - Install production dependencies"
	@echo "  $(COLOR_GREEN)install-dev$(COLOR_RESET)   - Install development dependencies"
	@echo "  $(COLOR_GREEN)format$(COLOR_RESET)        - Format code with ruff"
	@echo "  $(COLOR_GREEN)lint$(COLOR_RESET)          - Run linting checks"
	@echo "  $(COLOR_GREEN)typecheck$(COLOR_RESET)     - Run type checking with mypy"
	@echo "  $(COLOR_GREEN)test$(COLOR_RESET)          - Run all tests"
	@echo "  $(COLOR_GREEN)test-unit$(COLOR_RESET)     - Run unit tests only"
	@echo "  $(COLOR_GREEN)test-watch$(COLOR_RESET)    - Run tests in watch mode"
	@echo "  $(COLOR_GREEN)ci$(COLOR_RESET)            - Run all CI checks"
	@echo "  $(COLOR_GREEN)clean$(COLOR_RESET)         - Clean cache and build files"

## Install production dependencies
install:
	@echo "$(COLOR_YELLOW)Installing production dependencies...$(COLOR_RESET)"
	uv pip install -e .

## Install development dependencies
install-dev:
	@echo "$(COLOR_YELLOW)Installing development dependencies...$(COLOR_RESET)"
	uv pip install -e ".[dev]"

## Format code
format:
	@echo "$(COLOR_YELLOW)Formatting code...$(COLOR_RESET)"
	uv run ruff format .
	uv run ruff check --fix .

## Run linting
lint:
	@echo "$(COLOR_YELLOW)Running linter...$(COLOR_RESET)"
	uv run ruff check .

## Run type checking
typecheck:
	@echo "$(COLOR_YELLOW)Running type checker...$(COLOR_RESET)"
	uv run mypy . --config-file pyproject.toml

## Run all tests
test:
	@echo "$(COLOR_YELLOW)Running all tests...$(COLOR_RESET)"
	uv run pytest -v

## Run unit tests only
test-unit:
	@echo "$(COLOR_YELLOW)Running unit tests...$(COLOR_RESET)"
	uv run pytest -v -m unit

## Run integration tests only
test-integration:
	@echo "$(COLOR_YELLOW)Running integration tests...$(COLOR_RESET)"
	uv run pytest -v -m integration

## Run tests in watch mode
test-watch:
	@echo "$(COLOR_YELLOW)Running tests in watch mode...$(COLOR_RESET)"
	uv run ptw -- -vx

## Run all CI checks
ci: lint typecheck test
	@echo "$(COLOR_GREEN)All CI checks passed!$(COLOR_RESET)"

## Clean cache and build files
clean:
	@echo "$(COLOR_YELLOW)Cleaning cache files...$(COLOR_RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

## Time a command
time:
	@echo "Usage: make time <target>"
	@echo "Example: make time test"