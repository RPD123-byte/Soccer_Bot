#!/bin/bash

# CDK Setup and Run Script for TypeScript
# This script sets up the CDK environment and synthesizes the stack without deploying

set -e  # Exit on error

echo "🚀 AWS CDK TypeScript Setup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 14.x or higher."
    echo "You can install it from: https://nodejs.org/"
    echo "Or using nvm: nvm install --lts"
    exit 1
else
    NODE_VERSION=$(node --version)
    print_status "Node.js found: $NODE_VERSION"
    
    # Check Node version (should be 14.x or higher)
    NODE_MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
    if [ "$NODE_MAJOR_VERSION" -lt 14 ]; then
        print_error "Node.js version $NODE_VERSION is too old. Please install Node.js 14.x or higher."
        exit 1
    fi
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. It should come with Node.js."
    exit 1
else
    print_status "npm found: $(npm --version)"
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    print_warning "AWS CLI is not configured. Please run 'aws configure' first."
    echo "You can still synthesize the stack, but deployment will fail."
else
    print_status "AWS CLI configured for account: $(aws sts get-caller-identity --query Account --output text)"
fi

# Check if required files exist
echo ""
echo "Checking required files..."
REQUIRED_FILES=("bin/app.ts" "lib/my-application-stack.ts" "package.json" "tsconfig.json" "cdk.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "$file not found in current directory"
        echo "Make sure you have the following directory structure:"
        echo "  ├── bin/"
        echo "  │   └── app.ts"
        echo "  ├── lib/"
        echo "  │   └── my-application-stack.ts"
        echo "  ├── package.json"
        echo "  ├── tsconfig.json"
        echo "  └── cdk.json"
        exit 1
    else
        print_status "$file found"
    fi
done

# Install dependencies
echo ""
echo "Installing npm dependencies..."
npm install
print_status "npm dependencies installed"

# Build TypeScript
echo ""
echo "Building TypeScript..."
npm run build
print_status "TypeScript compiled successfully"

# Install AWS CDK CLI globally if not installed
if ! command -v cdk &> /dev/null; then
    echo ""
    echo "Installing AWS CDK CLI globally..."
    npm install -g aws-cdk
    print_status "AWS CDK CLI installed"
else
    print_status "AWS CDK CLI already installed: $(cdk --version)"
fi

# Bootstrap CDK (only needed once per account/region)
echo ""
echo "Checking CDK bootstrap status..."
BOOTSTRAP_REQUIRED=false
if ! aws cloudformation describe-stacks --stack-name CDKToolkit &> /dev/null; then
    print_warning "CDK is not bootstrapped in this account/region"
    echo "To deploy, you'll need to run: cdk bootstrap"
    BOOTSTRAP_REQUIRED=true
else
    print_status "CDK is already bootstrapped"
fi

# Synthesize the CDK app
echo ""
echo "Current directory: $(pwd)"
echo "Project structure:"
find . -name "*.ts" -o -name "*.json" | grep -E -v "node_modules|cdk.out" | sort
echo ""
echo "CDK configuration (cdk.json):"
cat cdk.json | head -5
echo ""
echo "Synthesizing CDK application..."
npm run synth
print_status "CDK synthesis complete"

# List stacks
echo ""
echo "Available stacks:"
npx cdk list

# Generate CloudFormation template
echo ""
echo "CloudFormation template generated in: cdk.out/"
print_status "You can find the synthesized template in the cdk.out directory"

# Show what would be deployed (diff against empty stack)
echo ""
echo "Showing what would be created if you deploy:"
echo "============================================"
npx cdk diff --no-color || true

# Print next steps
echo ""
echo "✨ Setup complete! ✨"
echo ""
echo "Next steps:"
echo "1. Review the generated CloudFormation template in cdk.out/"
echo "2. Modify files in lib/ to customize your infrastructure"

if [ "$BOOTSTRAP_REQUIRED" = true ]; then
    echo "3. Bootstrap CDK: npm run cdk bootstrap"
    echo "4. Deploy when ready: npm run deploy"
else
    echo "3. Deploy when ready: npm run deploy"
fi

echo ""
echo "Useful npm scripts:"
echo "  npm run build   - Compile TypeScript"
echo "  npm run watch   - Watch for changes and compile"
echo "  npm run test    - Run tests"
echo "  npm run synth   - Synthesize CloudFormation template"
echo "  npm run diff    - Show differences between stack and deployed"
echo "  npm run deploy  - Deploy stack to AWS"
echo "  npm run destroy - Remove stack from AWS"
echo ""
echo "Direct CDK commands:"
echo "  npx cdk synth   - Synthesize CloudFormation template"
echo "  npx cdk deploy  - Deploy stack to AWS"
echo "  npx cdk destroy - Remove stack from AWS"