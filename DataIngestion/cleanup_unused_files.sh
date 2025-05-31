#!/bin/bash
# Script to clean up unused docker-compose files and old scripts

echo "======================================="
echo "🧹 CLEANUP UNUSED FILES"
echo "======================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Files to remove
UNUSED_COMPOSE_FILES=(
    "docker-compose.override.yml"
    "docker-compose.cloud.yml"
    "docker-compose.parallel-feature.yml"
    "docker-compose.production.yml"
)

UNUSED_SCRIPTS=(
    "docker-flow-deploy.sh"
    "preflight_check.sh"
    "run_flow.sh"
    "run_model_builder_gpu_test.sh"
    "run_moea_comparison.sh"
    "run_orchestrated.sh"
    "run_orchestration.ps1"
    "run_parallel_feature_extraction.sh"
    "run_pipeline_with_validation.ps1"
    "run_pipeline_with_validation.sh"
)

# Function to check if file exists and show size
show_file_info() {
    local file=$1
    if [ -f "$file" ]; then
        local size=$(du -h "$file" | cut -f1)
        echo -e "  ${YELLOW}$file${NC} ($size)"
        return 0
    else
        echo -e "  ${GREEN}$file${NC} (already removed)"
        return 1
    fi
}

# Show what will be removed
echo "The following UNUSED files will be removed:"
echo ""
echo "Docker Compose files:"
for file in "${UNUSED_COMPOSE_FILES[@]}"; do
    show_file_info "$file"
done

echo ""
echo "Old/Unused scripts:"
for file in "${UNUSED_SCRIPTS[@]}"; do
    show_file_info "$file"
done

echo ""
echo -e "${GREEN}Files that will be KEPT:${NC}"
echo "  - docker-compose.yml (base configuration)"
echo "  - docker-compose.prod.yml (cloud production overrides)"
echo "  - run_production_pipeline.sh (used by terraform)"
echo "  - test_*.sh (testing scripts)"
echo "  - validate_services.sh (validation)"
echo "  - run_tests.sh (test runner)"
echo "  - All scripts in terraform/parallel-feature/"
echo ""

# Ask for confirmation
read -p "Do you want to remove these unused files? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Removing unused files..."
    
    # Remove unused docker-compose files
    for file in "${UNUSED_COMPOSE_FILES[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file"
            echo -e "${GREEN}✅ Removed: $file${NC}"
        fi
    done
    
    # Remove unused scripts
    for file in "${UNUSED_SCRIPTS[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file"
            echo -e "${GREEN}✅ Removed: $file${NC}"
        fi
    done
    
    echo ""
    echo -e "${GREEN}✅ Cleanup complete!${NC}"
    echo ""
    echo "Remaining docker-compose files:"
    ls -la docker-compose*.yml 2>/dev/null || echo "None found in current directory"
    
    echo ""
    echo "Remaining scripts:"
    ls -la *.sh *.ps1 2>/dev/null | grep -v cleanup_unused_files.sh || echo "None found"
    
else
    echo ""
    echo -e "${YELLOW}Cleanup cancelled. No files were removed.${NC}"
fi

echo ""
echo "To see the current structure:"
echo "  ls -la docker-compose*.yml"
echo "  ls -la *.sh *.ps1"