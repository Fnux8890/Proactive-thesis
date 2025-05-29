#!/bin/bash
# Terraform validation script for Google Cloud deployment

set -e

echo "=== Terraform Validation for Production Deployment ==="
echo

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform is not installed. Please install terraform first."
    echo "   Visit: https://www.terraform.io/downloads"
    exit 1
fi

# Check current directory
if [ ! -f "main.tf" ]; then
    echo "❌ main.tf not found. Please run this script from the terraform directory."
    exit 1
fi

# 1. Initialize Terraform
echo "1. Initializing Terraform..."
terraform init -backend=false || {
    echo "❌ Terraform initialization failed"
    exit 1
}
echo "✅ Terraform initialized"

# 2. Format check
echo -e "\n2. Checking Terraform formatting..."
if terraform fmt -check -recursive; then
    echo "✅ Terraform files are properly formatted"
else
    echo "⚠️  Some files need formatting. Run 'terraform fmt -recursive' to fix."
fi

# 3. Validate configuration
echo -e "\n3. Validating Terraform configuration..."
if terraform validate; then
    echo "✅ Terraform configuration is valid"
else
    echo "❌ Terraform validation failed"
    exit 1
fi

# 4. Check for required variables
echo -e "\n4. Checking required variables..."
REQUIRED_VARS=(
    "project_id"
    "region"
    "zone"
)

for var in "${REQUIRED_VARS[@]}"; do
    if grep -q "variable \"$var\"" *.tf; then
        echo "✅ Variable '$var' is defined"
    else
        echo "❌ Required variable '$var' is missing"
        exit 1
    fi
done

# 5. Validate Google Cloud configuration
echo -e "\n5. Checking Google Cloud configuration..."

# Check if gcloud is installed
if command -v gcloud &> /dev/null; then
    echo "✅ gcloud CLI is installed"
    
    # Check if authenticated
    if gcloud auth list --filter=status:ACTIVE --format="get(account)" | grep -q "@"; then
        echo "✅ Authenticated with Google Cloud"
        
        # Check if project is set
        if [ -n "$TF_VAR_project_id" ]; then
            echo "✅ Project ID is set: $TF_VAR_project_id"
        else
            echo "⚠️  TF_VAR_project_id is not set. Will need to provide during apply."
        fi
    else
        echo "⚠️  Not authenticated with Google Cloud. Run 'gcloud auth login'"
    fi
else
    echo "⚠️  gcloud CLI not installed. Cannot validate Google Cloud settings."
fi

# 6. Check Docker Compose references
echo -e "\n6. Checking Docker Compose file references..."

# Check startup.sh for old references
if [ -f "startup.sh" ]; then
    if grep -q "docker-compose.parallel-feature.yml\|docker-compose.cloud.yml" startup.sh; then
        echo "⚠️  startup.sh contains references to old docker-compose files"
        echo "   Should use: docker-compose.production.yml"
    else
        echo "✅ startup.sh uses correct docker-compose files"
    fi
fi

# 7. Security checks (basic)
echo -e "\n7. Running basic security checks..."

# Check for hardcoded credentials
if grep -r "password\s*=\s*\"[^\"]\+\"" *.tf | grep -v "random_password\|var\." ; then
    echo "⚠️  Potential hardcoded passwords found"
else
    echo "✅ No hardcoded passwords found"
fi

# Check for public IP assignments
if grep -q "access_config\s*{" main.tf; then
    echo "⚠️  Instance has public IP configured. Ensure this is intended."
else
    echo "✅ No public IP assignments found"
fi

# 8. Resource validation
echo -e "\n8. Validating resource specifications..."

# Check machine type
if grep -q "machine_type.*=.*\"a2-highgpu-4g\"" main.tf; then
    echo "✅ Correct machine type (a2-highgpu-4g) specified"
else
    echo "❌ Machine type should be 'a2-highgpu-4g' for 4 GPU configuration"
fi

# Check GPU configuration
if grep -q "guest_accelerator.*count.*=.*4" main.tf; then
    echo "✅ Correct GPU count (4) specified"
else
    echo "❌ GPU count should be 4 for A2 instance"
fi

# 9. Generate plan (dry run)
echo -e "\n9. Generating Terraform plan (dry run)..."
if [ -n "$TF_VAR_project_id" ]; then
    echo "Running: terraform plan -var=\"project_id=$TF_VAR_project_id\""
    terraform plan -var="project_id=$TF_VAR_project_id" -out=tfplan.out > /dev/null 2>&1 && {
        echo "✅ Terraform plan generated successfully"
        echo "   Run 'terraform show tfplan.out' to review"
    } || {
        echo "⚠️  Terraform plan failed. Check your configuration."
    }
else
    echo "ℹ️  Skipping plan generation (TF_VAR_project_id not set)"
fi

# 10. Summary
echo -e "\n=== Validation Summary ==="
echo "Most validation checks passed. Review any warnings above."
echo
echo "Next steps:"
echo "1. Fix any issues identified above"
echo "2. Set your project ID: export TF_VAR_project_id=your-project-id"
echo "3. Run: terraform plan"
echo "4. Review the plan carefully"
echo "5. Run: terraform apply"
echo
echo "For production deployment, ensure:"
echo "- You have proper authentication (gcloud auth application-default login)"
echo "- Your project has sufficient quota for A2 instances"
echo "- Billing is enabled for your project"
echo "- You have the necessary IAM permissions"