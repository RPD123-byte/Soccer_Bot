#!/bin/bash

# CDK Setup and Run Script for Go
# This script sets up the CDK environment and synthesizes the stack without deploying

set -e  # Exit on error

echo "🚀 AWS CDK Go Setup Script"
echo "=========================="

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

# Check if Go is installed
if ! command -v go &> /dev/null; then
    print_error "Go is not installed. Please install Go 1.21 or higher."
    echo "You can install it from: https://go.dev/dl/"
    echo "Or using homebrew: brew install go"
    exit 1
else
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    print_status "Go found: go version $GO_VERSION"
    
    # Check Go version (should be 1.21 or higher)
    REQUIRED_VERSION="1.21"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$GO_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_warning "Go version $GO_VERSION is installed, but 1.21 or higher is recommended"
    fi
fi

# Check if Node.js is installed (required for CDK CLI)
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 14.x or higher."
    exit 1
else
    print_status "Node.js found: $(node --version)"
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
if [ ! -f "main.go" ]; then
    print_error "main.go not found in current directory"
    exit 1
else
    print_status "main.go found"
fi

if [ ! -f "cdk.json" ]; then
    print_error "cdk.json not found in current directory"
    exit 1
else
    print_status "cdk.json found"
fi

if [ ! -f "go.mod" ]; then
    print_error "go.mod not found in current directory"
    exit 1
else
    print_status "go.mod found"
fi

# Initialize Go module if needed
echo ""
echo "Downloading Go dependencies..."
go mod download
print_status "Go dependencies downloaded"

# Tidy up dependencies
echo ""
echo "Tidying Go modules..."
go mod tidy
print_status "Go modules tidied"

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

# Build the Go application
echo ""
echo "Building Go application..."
go build -o cdk-app main.go
print_status "Go application built successfully"

# Synthesize the CDK app
echo ""
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la
echo ""
echo "CDK configuration (cdk.json):"
cat cdk.json | head -5
echo ""
echo "Synthesizing CDK application..."
cdk synth
print_status "CDK synthesis complete"

# List stacks
echo ""
echo "Available stacks:"
cdk list

# Generate CloudFormation template
echo ""
echo "CloudFormation template generated in: cdk.out/"
print_status "You can find the synthesized template in the cdk.out directory"

# Show what would be deployed (diff against empty stack)
echo ""
echo "Showing what would be created if you deploy:"
echo "============================================"
cdk diff --no-color || true

# Print next steps
echo ""
echo "✨ Setup complete! ✨"
echo ""
echo "Next steps:"
echo "1. Review the generated CloudFormation template in cdk.out/"
echo "2. Modify main.go to customize your infrastructure"

if [ "$BOOTSTRAP_REQUIRED" = true ]; then
    echo "3. Bootstrap CDK: cdk bootstrap"
    echo "4. Deploy when ready: cdk deploy"
else
    echo "3. Deploy when ready: cdk deploy"
fi

echo ""
echo "Useful commands:"
echo "  cdk synth       - Synthesize CloudFormation template"
echo "  cdk diff        - Show differences between stack and deployed"
echo "  cdk deploy      - Deploy stack to AWS"
echo "  cdk destroy     - Remove stack from AWS"
echo "  cdk docs        - Open CDK documentation"
echo ""
echo "Go development commands:"
echo "  go mod download - Download dependencies"
echo "  go build        - Build the application"
echo "  go test         - Run tests"