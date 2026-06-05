#!/bin/bash

set -e

echo "🚀 AWS CDK .NET Setup Script"
echo "============================"

# Check if .NET is installed
if ! command -v dotnet &> /dev/null; then
    echo "❌ .NET is not installed. Please install .NET 8.0 or higher."
    echo "Download from: https://dotnet.microsoft.com/download"
    exit 1
fi

echo "✅ .NET found: $(dotnet --version)"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Required for CDK CLI."
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Check AWS CLI
if ! aws sts get-caller-identity &> /dev/null; then
    echo "⚠️  AWS CLI not configured. Run 'aws configure' to deploy."
else
    echo "✅ AWS CLI configured"
fi

# Install CDK CLI if needed
if ! command -v cdk &> /dev/null; then
    echo "Installing AWS CDK CLI..."
    npm install -g aws-cdk
fi

echo "✅ CDK CLI: $(cdk --version)"

# Restore NuGet packages
echo "Restoring NuGet packages..."
dotnet restore

# Build the project
echo "Building project..."
dotnet build

# Synthesize
echo "Synthesizing CDK app..."
cdk synth

echo ""
echo "✨ Setup complete!"
echo ""
echo "Available commands:"
echo "  cdk synth    - Synthesize CloudFormation"
echo "  cdk deploy   - Deploy stack"
echo "  cdk destroy  - Remove stack"
echo "  cdk diff     - Show changes"