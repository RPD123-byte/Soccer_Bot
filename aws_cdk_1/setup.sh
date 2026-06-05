#!/bin/bash

# CDK Setup and Run Script
# This script sets up the CDK environment and synthesizes the stack without deploying

set -e  # Exit on error

echo "🚀 AWS CDK Python Setup Script"
echo "=============================="

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

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    print_error "Python 3.11 is not installed. Please install Python 3.11.11."
    echo "You can install it using:"
    echo "  brew install python@3.11  # on macOS"
    echo "  or use pyenv: pyenv install 3.11.11"
    exit 1
else
    print_status "Python 3.11 found: $(python3.11 --version)"
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
if [ ! -f "app.py" ]; then
    print_error "app.py not found in current directory"
    exit 1
else
    print_status "app.py found"
fi

if [ ! -f "cdk.json" ]; then
    print_error "cdk.json not found in current directory"
    exit 1
else
    print_status "cdk.json found"
fi

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in current directory"
    exit 1
else
    print_status "requirements.txt found"
fi

# Make app.py executable
chmod +x app.py
print_status "Made app.py executable"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating Python virtual environment..."
    python3.11 -m venv .venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
print_status "pip upgraded"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt --quiet
print_status "Python dependencies installed"

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
echo "2. Modify app.py to customize your infrastructure"

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
echo "To activate the virtual environment in the future:"
echo "  source .venv/bin/activate"