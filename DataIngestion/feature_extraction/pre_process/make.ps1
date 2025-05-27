# PowerShell equivalent of Makefile commands
param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

# Colors
$ColorReset = "`e[0m"
$ColorGreen = "`e[32m"
$ColorYellow = "`e[33m"
$ColorBlue = "`e[34m"

function Show-Help {
    Write-Host "${ColorBlue}Available targets:${ColorReset}"
    Write-Host "  ${ColorGreen}install${ColorReset}       - Install production dependencies"
    Write-Host "  ${ColorGreen}install-dev${ColorReset}   - Install development dependencies"
    Write-Host "  ${ColorGreen}format${ColorReset}        - Format code with ruff"
    Write-Host "  ${ColorGreen}lint${ColorReset}          - Run linting checks"
    Write-Host "  ${ColorGreen}typecheck${ColorReset}     - Run type checking with mypy"
    Write-Host "  ${ColorGreen}test${ColorReset}          - Run all tests"
    Write-Host "  ${ColorGreen}test-unit${ColorReset}     - Run unit tests only"
    Write-Host "  ${ColorGreen}test-watch${ColorReset}    - Run tests in watch mode"
    Write-Host "  ${ColorGreen}ci${ColorReset}            - Run all CI checks"
    Write-Host "  ${ColorGreen}clean${ColorReset}         - Clean cache and build files"
}

function Install {
    Write-Host "${ColorYellow}Installing production dependencies...${ColorReset}"
    uv pip install -r requirements.txt
}

function Install-Dev {
    Write-Host "${ColorYellow}Installing development dependencies...${ColorReset}"
    uv pip install -r requirements-dev.txt
}

function Format-Code {
    Write-Host "${ColorYellow}Formatting code...${ColorReset}"
    python -m ruff format .
    python -m ruff check --fix .
}

function Run-Lint {
    Write-Host "${ColorYellow}Running linter...${ColorReset}"
    python -m ruff check .
}

function Run-TypeCheck {
    Write-Host "${ColorYellow}Running type checker...${ColorReset}"
    python -m mypy . --config-file pyproject.toml
}

function Run-Test {
    Write-Host "${ColorYellow}Running all tests...${ColorReset}"
    python -m pytest -v
}

function Run-TestUnit {
    Write-Host "${ColorYellow}Running unit tests...${ColorReset}"
    python -m pytest -v -m unit
}

function Run-TestWatch {
    Write-Host "${ColorYellow}Running tests in watch mode...${ColorReset}"
    python -m pytest_watch -- -vx
}

function Run-CI {
    Run-Lint
    Run-TypeCheck
    Run-Test
    Write-Host "${ColorGreen}All CI checks passed!${ColorReset}"
}

function Clean-Cache {
    Write-Host "${ColorYellow}Cleaning cache files...${ColorReset}"
    
    # Remove __pycache__ directories
    Get-ChildItem -Path . -Directory -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove .pytest_cache
    Get-ChildItem -Path . -Directory -Recurse -Filter ".pytest_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove .mypy_cache
    Get-ChildItem -Path . -Directory -Recurse -Filter ".mypy_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove .ruff_cache
    Get-ChildItem -Path . -Directory -Recurse -Filter ".ruff_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove .pyc files
    Get-ChildItem -Path . -File -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    
    Write-Host "${ColorGreen}Cache cleaned!${ColorReset}"
}

# Execute the target
switch ($Target) {
    "help" { Show-Help }
    "install" { Install }
    "install-dev" { Install-Dev }
    "format" { Format-Code }
    "lint" { Run-Lint }
    "typecheck" { Run-TypeCheck }
    "test" { Run-Test }
    "test-unit" { Run-TestUnit }
    "test-watch" { Run-TestWatch }
    "ci" { Run-CI }
    "clean" { Clean-Cache }
    default { 
        Write-Host "${ColorYellow}Unknown target: $Target${ColorReset}"
        Show-Help
    }
}