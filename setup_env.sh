#!/bin/bash

echo "🔧 Setting up environment for llama-combine"
echo ""

# Check if .env exists
if [ -f .env ]; then
    echo "✅ .env file already exists"
else
    # Copy template to .env
    if [ -f env-template.txt ]; then
        cp env-template.txt .env
        echo "✅ Created .env from template"
    else
        echo "❌ env-template.txt not found"
        exit 1
    fi
fi

echo ""
echo "📝 Next steps:"
echo ""
echo "1. Edit .env file and add your tokens:"
echo "   - HF_TOKEN: Required for Llama models"
echo "   - OPENAI_API_KEY: Required for data generation features"
echo ""
echo "2. For Llama models specifically:"
echo "   a. Create account at https://huggingface.co"
echo "   b. Accept license at https://huggingface.co/meta-llama/Llama-3.2-1B"
echo "   c. Create token at https://huggingface.co/settings/tokens"
echo "   d. Add token to .env file"
echo ""
echo "3. Alternative: Use TinyLlama (no auth required)"
echo "   - Select 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' in the UI"
echo ""

# Make script executable
chmod +x setup_env.sh 