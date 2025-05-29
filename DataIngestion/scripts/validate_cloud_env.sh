#!/bin/bash
# Validate cloud environment variables are properly set

echo "=== Cloud Environment Validation ==="
echo ""

# Required environment variables
REQUIRED_VARS=(
    "DB_HOST"
    "DB_PASSWORD"
    "DATABASE_URL"
)

# Check each required variable
MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
        echo "❌ $var is not set"
    else
        # Mask sensitive information
        if [[ "$var" == *"PASSWORD"* ]] || [[ "$var" == *"URL"* ]]; then
            echo "✅ $var is set (***masked***)"
        else
            echo "✅ $var = ${!var}"
        fi
    fi
done

echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    echo "✅ .env file exists"
    # Check if it contains required variables
    for var in "${REQUIRED_VARS[@]}"; do
        if grep -q "^$var=" .env; then
            echo "  ✅ $var found in .env"
        else
            echo "  ❌ $var missing from .env"
        fi
    done
else
    echo "❌ .env file not found"
fi

echo ""

# Test database connectivity
if [ -n "$DB_HOST" ] && [ -n "$DB_PASSWORD" ]; then
    echo "Testing database connectivity..."
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U postgres -d postgres -c "SELECT 1" >/dev/null 2>&1; then
        echo "✅ Database connection successful"
        
        # Check if greenhouse database exists
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U postgres -lqt | cut -d \| -f 1 | grep -qw greenhouse; then
            echo "✅ Greenhouse database exists"
            
            # Check TimescaleDB
            if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U postgres -d greenhouse -c "SELECT extname FROM pg_extension WHERE extname = 'timescaledb';" | grep -q timescaledb; then
                echo "✅ TimescaleDB extension is installed"
            else
                echo "⚠️  TimescaleDB extension not found"
            fi
        else
            echo "⚠️  Greenhouse database does not exist"
        fi
    else
        echo "❌ Database connection failed"
    fi
else
    echo "⚠️  Skipping database connectivity test (missing credentials)"
fi

echo ""

# Check Docker
if command -v docker >/dev/null 2>&1; then
    echo "✅ Docker is installed: $(docker --version)"
    
    # Check Docker Compose
    if command -v docker-compose >/dev/null 2>&1; then
        echo "✅ Docker Compose is installed: $(docker-compose --version)"
    else
        echo "❌ Docker Compose not found"
    fi
    
    # Check GPU support
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
        echo "✅ GPU support is working"
        # Count GPUs
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        echo "  ✅ Found $GPU_COUNT GPU(s)"
    else
        echo "⚠️  GPU support not available or not working"
    fi
else
    echo "❌ Docker not found"
fi

echo ""

# Summary
if [ ${#MISSING_VARS[@]} -eq 0 ]; then
    echo "✅ All environment variables are set"
    echo "✅ Ready for cloud deployment!"
else
    echo "❌ Missing ${#MISSING_VARS[@]} required environment variable(s)"
    echo "Please set the following variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  export $var=<value>"
    done
fi