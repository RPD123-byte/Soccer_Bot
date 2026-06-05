#!/bin/bash

set -e

echo "🚀 AWS CDK Java Setup Script"
echo "============================"

# Check Java
if ! command -v java &> /dev/null; then
    echo "❌ Java not installed. Install Java 17+."
    exit 1
fi
echo "✅ Java found: $(java -version 2>&1 | head -n1)"

# Check Maven
if ! command -v mvn &> /dev/null; then
    echo "❌ Maven not installed. Install Maven 3.6+."
    exit 1
fi
echo "✅ Maven found: $(mvn -version | head -n1)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not installed."
    exit 1
fi
echo "✅ Node.js found: $(node --version)"

# Check AWS
if ! aws sts get-caller-identity &> /dev/null; then
    echo "⚠️  AWS CLI not configured."
else
    echo "✅ AWS CLI configured"
fi

# Install CDK
if ! command -v cdk &> /dev/null; then
    npm install -g aws-cdk
fi
echo "✅ CDK CLI: $(cdk --version)"

# Create directory structure
mkdir -p src/main/java/com/myorg

# Move Java files if needed
[ -f App.java ] && mv App.java src/main/java/com/myorg/
[ -f MyApplicationStack.java ] && mv MyApplicationStack.java src/main/java/com/myorg/

# Build
echo "Building project..."
mvn compile

# Synthesize
echo "Synthesizing CDK app..."
cdk synth

echo ""
echo "✨ Setup complete!"
echo ""
echo "Commands:"
echo "  cdk synth    - Generate CloudFormation"
echo "  cdk deploy   - Deploy stack"
echo "  cdk destroy  - Remove stack"