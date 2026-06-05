#!/bin/bash

# Prerequisites check and setup
echo "Setting up Comprehensive Pulumi JavaScript project..."

# 1. Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js first."
    echo "Visit: https://nodejs.org/"
    exit 1
fi

# 2. Check if Pulumi is installed, if not, install it
if ! command -v pulumi &> /dev/null; then
    echo "Installing Pulumi..."
    curl -fsSL https://get.pulumi.com | sh
    export PATH=$PATH:$HOME/.pulumi/bin
fi

# 3. Create project directory if it doesn't exist
PROJECT_DIR="pulumi-js-project"
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
fi

# 4. Copy your index.js to project directory (if running from parent directory)
if [ -f "index.js" ] && [ "$(pwd)" != "$(pwd)/$PROJECT_DIR" ]; then
    cp index.js "$PROJECT_DIR/"
fi

cd "$PROJECT_DIR" || exit 1

# 5. Initialize npm project if package.json doesn't exist
if [ ! -f "package.json" ]; then
    echo "Initializing npm project..."
    npm init -y
fi

# 6. Install required dependencies
echo "Installing Pulumi core..."
npm install @pulumi/pulumi

echo "Installing cloud provider packages..."
npm install @pulumi/aws @pulumi/azure-native @pulumi/gcp @pulumi/digitalocean

echo "Installing container and orchestration packages..."
npm install @pulumi/kubernetes @pulumi/docker

echo "Installing infrastructure packages..."
npm install @pulumi/cloudflare @pulumi/github @pulumi/random @pulumi/tls @pulumi/vault

echo "Installing database packages..."
npm install @pulumi/postgresql @pulumi/mysql @pulumi/mongodbatlas

echo "Installing monitoring and observability packages..."
npm install @pulumi/datadog @pulumi/newrelic @pulumi/pagerduty

echo "Installing identity and auth packages..."
npm install @pulumi/okta @pulumi/auth0

echo "Installing other service packages..."
npm install @pulumi/stripe

# 7. Create Pulumi.yaml if it doesn't exist
if [ ! -f "Pulumi.yaml" ]; then
    echo "Creating Pulumi.yaml..."
    cat > Pulumi.yaml << EOF
name: pulumi-js-project
runtime: nodejs
description: A comprehensive Pulumi JavaScript program with multiple providers
main: index.js
EOF
fi

# 8. Login to Pulumi (using local backend for simplicity)
echo "Logging in to Pulumi..."
pulumi login --local

# 9. Check if stack exists, if not create it
if ! pulumi stack ls 2>/dev/null | grep -q "dev"; then
    echo "Creating Pulumi stack..."
    pulumi stack init dev
else
    echo "Selecting existing dev stack..."
    pulumi stack select dev
fi

# 10. Configure AWS region (and other basic configs)
echo "Configuring providers..."
pulumi config set aws:region us-east-1  # Change to your preferred region
pulumi config set azure-native:location "East US"
pulumi config set gcp:project "your-gcp-project"  # Update with your GCP project
pulumi config set digitalocean:token "your-do-token" || true
pulumi config set github:token "your-github-token" || true
pulumi config set cloudflare:apiToken "your-cf-token" || true

# 11. Show installed packages
echo ""
echo "📦 Installed Pulumi packages:"
npm list | grep "@pulumi"

# 12. Check AWS credentials (primary provider)
if ! aws sts get-caller-identity &> /dev/null 2>&1; then
    echo ""
    echo "⚠️  AWS credentials not configured. Please configure AWS credentials:"
    echo "   Option 1: Set environment variables"
    echo "      export AWS_ACCESS_KEY_ID=your-key"
    echo "      export AWS_SECRET_ACCESS_KEY=your-secret"
    echo ""
    echo "   Option 2: Use AWS CLI"
    echo "      aws configure"
    echo ""
fi

# 13. Create a summary of what will be tested
echo ""
echo "✅ Setup complete! This project includes the following providers:"
echo "   ☁️  Cloud Providers: AWS, Azure, GCP, DigitalOcean"
echo "   🐳 Container/K8s: Docker, Kubernetes"
echo "   🔧 Infrastructure: Cloudflare, GitHub, Vault"
echo "   💾 Databases: PostgreSQL, MySQL, MongoDB Atlas"
echo "   📊 Monitoring: Datadog, New Relic, PagerDuty"
echo "   🔐 Identity: Okta, Auth0"
echo "   💳 Services: Stripe"
echo "   🎲 Utilities: Random, TLS"

echo ""
echo "📋 Next steps:"
echo "   - To deploy: pulumi up"
echo "   - To destroy: pulumi destroy"
echo "   - To see outputs: pulumi stack output"
echo ""
echo "Note: Many resources are configured with 'ignoreChanges' to prevent"
echo "      errors if credentials are not configured for all providers."