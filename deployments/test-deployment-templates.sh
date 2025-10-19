#!/bin/bash

# Test script to validate deployment templates for JadeVectorDB multi-cloud deployment

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting validation of deployment templates..."

# Test AWS templates
echo "Validating AWS templates..."
if command -v aws >/dev/null 2>&1; then
    echo "AWS CLI found, validating CloudFormation template syntax..."
    # Validate the CloudFormation template
    if aws cloudformation validate-template --template-body file://aws/jadevectordb-stack.yaml; then
        echo "✓ AWS CloudFormation template is valid"
    else
        echo "✗ AWS CloudFormation template validation failed"
    fi
else
    echo "AWS CLI not found, skipping CloudFormation validation"
fi

# Test Azure templates
echo "Validating Azure templates..."
if command -v az >/dev/null 2>&1; then
    echo "Azure CLI found, validating ARM template syntax..."
    # Validate the ARM template
    if az deployment group validate --template-file azure/jadevectordb-aks-template.json --resource-group /subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test --parameters clusterName=test; then
        echo "✓ Azure ARM template is valid"
    else
        echo "✗ Azure ARM template validation failed"
    fi
else
    echo "Azure CLI not found, skipping ARM template validation"
fi

# Check if Bicep CLI is available and validate the Bicep template
if command -v bicep >/dev/null 2>&1; then
    echo "Bicep CLI found, validating Bicep template syntax..."
    if bicep build azure/jadevectordb-aks-template.bicep -o azure/jadevectordb-aks-template-compiled.json; then
        echo "✓ Azure Bicep template is valid"
        rm azure/jadevectordb-aks-template-compiled.json  # Clean up compiled file
    else
        echo "✗ Azure Bicep template validation failed"
    fi
else
    echo "Bicep CLI not found, skipping Bicep template validation"
fi

# Test GCP templates
echo "Validating GCP templates..."
if command -v gcloud >/dev/null 2>&1; then
    echo "Google Cloud SDK found, validating deployment manager template syntax..."
    # For GCP, we'll just check if the Python file is syntactically valid
    if python3 -m py_compile gcp/jadevectordb-cluster.py; then
        echo "✓ GCP deployment template is syntactically valid"
    else
        echo "✗ GCP deployment template syntax validation failed"
    fi
else
    echo "Google Cloud SDK not found, skipping GCP validation"
fi

# Check Terraform template syntax
if command -v terraform >/dev/null 2>&1; then
    echo "Terraform found, validating Terraform configuration syntax..."
    cd gcp
    # Initialize a temporary Terraform working directory for validation only
    mkdir -p temp-tf-test
    cp terra-jadevectordb.tf temp-tf-test/
    cd temp-tf-test
    
    if terraform init -backend=false >/dev/null 2>&1; then
        if terraform validate; then
            echo "✓ GCP Terraform configuration is valid"
        else
            echo "✗ GCP Terraform configuration validation failed"
        fi
    else
        echo "✗ Could not initialize Terraform for validation"
    fi
    
    cd ..
    rm -rf temp-tf-test
    cd ..
else
    echo "Terraform not found, skipping Terraform validation"
fi

# Validate YAML files (config and documentation)
echo "Validating YAML configuration files..."
if command -v yamllint >/dev/null 2>&1; then
    if yamllint multicloud/jadevectordb-config.yaml; then
        echo "✓ Cloud-agnostic configuration YAML is valid"
    else
        echo "✗ Cloud-agnostic configuration YAML validation failed"
    fi
else
    echo "yamllint not found, skipping YAML validation (install yamllint for validation)"
fi

# Check shell script syntax
echo "Validating shell script syntax..."
if bash -n multicloud/deploy-jadevectordb.sh; then
    echo "✓ Deployment script syntax is valid"
else
    echo "✗ Deployment script syntax validation failed"
fi

# Check that all expected files exist
echo "Verifying all required files exist..."
required_files=(
    "aws/jadevectordb-stack.yaml"
    "aws/eks-cluster.yaml"
    "aws/jadevectordb-eks.yaml"
    "azure/jadevectordb-aks-template.json"
    "azure/jadevectordb-aks-template.bicep"
    "gcp/jadevectordb-deployment.yaml"
    "gcp/jadevectordb-cluster.py"
    "gcp/terraform-jadevectordb.tf"
    "multicloud/jadevectordb-config.yaml"
    "multicloud/deploy-jadevectordb.sh"
    "multicloud/README.md"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✓ All required files are present"
else
    echo "✗ Missing files:"
    printf '%s\n' "${missing_files[@]}"
fi

echo "Validation complete."