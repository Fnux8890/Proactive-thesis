#!/bin/bash
# Quick validation script to check for common issues before deployment

echo "======================================="
echo "🔍 VALIDATING SERVICE CONFIGURATIONS"
echo "======================================="
echo ""

ISSUES=()
WARNINGS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if required files exist
echo "Checking required files..."

REQUIRED_FILES=(
    "docker-compose.yml"
    "docker-compose.prod.yml"
    "run_production_pipeline.sh"
    ".env.example"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ Found: $file${NC}"
    else
        echo -e "${RED}❌ Missing: $file${NC}"
        ISSUES+=("Missing required file: $file")
    fi
done

# Check Dockerfiles
echo ""
echo "Checking Dockerfiles..."

DOCKERFILES=(
    "rust_pipeline/Dockerfile"
    "feature_extraction/pre_process/preprocess.dockerfile"
    "feature_extraction/era_detection_rust/dockerfile"
    "feature_extraction/feature/feature.dockerfile"
    "model_builder/dockerfile"
    "moea_optimizer/Dockerfile"
    "moea_optimizer/Dockerfile.gpu"
)

for dockerfile in "${DOCKERFILES[@]}"; do
    if [ -f "$dockerfile" ]; then
        echo -e "${GREEN}✅ Found: $dockerfile${NC}"
        
        # Check for common issues
        if grep -q "COPY.*requirements.*txt" "$dockerfile" 2>/dev/null; then
            # Check if requirements file exists
            dir=$(dirname "$dockerfile")
            if ! ls "$dir"/requirements*.txt >/dev/null 2>&1; then
                echo -e "${RED}   ❌ Missing requirements.txt in $dir${NC}"
                ISSUES+=("Missing requirements.txt for $dockerfile")
            fi
        fi
        
        # Check for proper base images
        if grep -q "FROM.*cuda" "$dockerfile" 2>/dev/null; then
            echo -e "${YELLOW}   ⚠️  Uses CUDA - requires GPU${NC}"
            WARNINGS+=("$dockerfile requires GPU support")
        fi
    else
        echo -e "${RED}❌ Missing: $dockerfile${NC}"
        ISSUES+=("Missing Dockerfile: $dockerfile")
    fi
done

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."

find . -name "requirements*.txt" -type f | while read -r req_file; do
    echo -e "${GREEN}✅ Found: $req_file${NC}"
    
    # Check for problematic dependencies
    if grep -E "^(torch|tensorflow|jax)" "$req_file" >/dev/null 2>&1; then
        echo -e "${YELLOW}   ⚠️  Contains ML frameworks - may need specific versions${NC}"
    fi
    
    # Check for missing files referenced in requirements
    if grep -E "^-e |^\./" "$req_file" >/dev/null 2>&1; then
        echo -e "${YELLOW}   ⚠️  Contains local package references${NC}"
    fi
done

# Check Rust projects
echo ""
echo "Checking Rust projects..."

find . -name "Cargo.toml" -type f | while read -r cargo_file; do
    dir=$(dirname "$cargo_file")
    echo -e "${GREEN}✅ Found: $cargo_file${NC}"
    
    # Check if Cargo.lock exists
    if [ ! -f "$dir/Cargo.lock" ]; then
        echo -e "${YELLOW}   ⚠️  Missing Cargo.lock - dependencies not locked${NC}"
        WARNINGS+=("Missing Cargo.lock in $dir")
    fi
done

# Check environment variables
echo ""
echo "Checking environment configuration..."

if [ -f ".env.example" ]; then
    echo -e "${GREEN}✅ Found .env.example${NC}"
    
    # Extract required variables
    REQUIRED_VARS=$(grep -E "^[A-Z_]+=" .env.example | cut -d= -f1)
    echo "   Required variables:"
    for var in $REQUIRED_VARS; do
        echo "   - $var"
    done
else
    echo -e "${YELLOW}⚠️  No .env.example found${NC}"
fi

# Check docker-compose syntax
echo ""
echo "Validating docker-compose files..."

for compose_file in docker-compose*.yml; do
    if [ -f "$compose_file" ]; then
        if docker compose -f "$compose_file" config > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Valid: $compose_file${NC}"
        else
            echo -e "${RED}❌ Invalid: $compose_file${NC}"
            ISSUES+=("Invalid docker-compose syntax in $compose_file")
            docker compose -f "$compose_file" config 2>&1 | grep -E "ERROR|error" | head -5
        fi
    fi
done

# Check for large files
echo ""
echo "Checking for large files that might cause issues..."

find . -type f -size +100M 2>/dev/null | grep -v -E "\.git/|node_modules/|__pycache__/|target/" | while read -r large_file; do
    size=$(du -h "$large_file" | cut -f1)
    echo -e "${YELLOW}⚠️  Large file: $large_file ($size)${NC}"
    WARNINGS+=("Large file found: $large_file ($size)")
done

# Check for missing GPU configurations
echo ""
echo "Checking GPU configurations..."

if grep -r "USE_GPU" docker-compose*.yml >/dev/null 2>&1; then
    echo -e "${GREEN}✅ GPU configuration found${NC}"
else
    echo -e "${YELLOW}⚠️  No GPU configuration found${NC}"
fi

# Summary
echo ""
echo "======================================="
echo "📊 VALIDATION SUMMARY"
echo "======================================="
echo ""

if [ ${#ISSUES[@]} -eq 0 ] && [ ${#WARNINGS[@]} -eq 0 ]; then
    echo -e "${GREEN}🎉 No issues found! Ready for testing.${NC}"
else
    if [ ${#ISSUES[@]} -gt 0 ]; then
        echo -e "${RED}❌ Issues (${#ISSUES[@]}) - Fix these before deployment:${NC}"
        for issue in "${ISSUES[@]}"; do
            echo "   - $issue"
        done
        echo ""
    fi
    
    if [ ${#WARNINGS[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Warnings (${#WARNINGS[@]}) - Review these:${NC}"
        for warning in "${WARNINGS[@]}"; do
            echo "   - $warning"
        done
    fi
fi

echo ""
echo "Next steps:"
echo "1. Fix any issues listed above"
echo "2. Run: ./test_all_services.sh"
echo "3. Deploy to cloud: cd terraform/parallel-feature && terraform apply"

# Exit with error if critical issues found
if [ ${#ISSUES[@]} -gt 0 ]; then
    exit 1
fi