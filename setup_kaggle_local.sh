#!/bin/bash
# Kaggle Local Training Setup Script
# This script helps you set up the environment for Kaggle local training

set -e

echo "🚀 Kaggle Local Training Setup"
echo "================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
else
    echo "✅ Python found"
fi

if ! command_exists pip; then
    echo "❌ pip not found. Please install pip"
    exit 1
else
    echo "✅ pip found"
fi

# Check if in iris-pipeline directory
if [ ! -f "STEP_BY_STEP_GUIDE.md" ]; then
    echo "❌ Please run this script from the iris-pipeline directory"
    exit 1
else
    echo "✅ In iris-pipeline directory"
fi

# Install basic dependencies
echo ""
echo "📦 Installing basic dependencies..."
pip install pandas scikit-learn pytest fastapi uvicorn pydantic httpx

# Create models directory
echo ""
echo "📁 Creating models directory..."
mkdir -p models/iris-gemma-local
echo "✅ Created models/iris-gemma-local/"

# Set up environment variables
echo ""
echo "🔧 Environment Setup"
echo "===================="
echo "Add these to your shell profile (.bashrc, .zshrc, etc.):"
echo ""
echo "# Kaggle Local Training"
echo "export USE_LLM_MODEL=true"
echo "export USE_LOCAL_GEMMA=true" 
echo "export LOCAL_MODEL_PATH=models/iris-gemma-local"
echo "export HF_TOKEN=your_huggingface_token_here"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file for local development..."
    cat > .env << EOF
# Kaggle Local Training Configuration
USE_LLM_MODEL=true
USE_LOCAL_GEMMA=true
LOCAL_MODEL_PATH=models/iris-gemma-local
HF_TOKEN=your_huggingface_token_here
EOF
    echo "✅ Created .env file"
else
    echo "ℹ️  .env file already exists"
fi

# Test basic functionality
echo ""
echo "🧪 Testing basic functionality..."
export PYTHONPATH=.
python -c "
import src.gemma_classifier as gc
mock = gc.MockGemmaIrisClassifier()
assert mock.load_model(), 'Mock model should load'
result = mock.predict(5.1, 3.5, 1.4, 0.2)
assert len(result) == 2, 'Should return (species, confidence)'
print('✅ Basic functionality test passed')
"

echo ""
echo "📖 Next Steps"
echo "============="
echo "1. Get your HuggingFace token from https://huggingface.co/settings/tokens"
echo "2. Update HF_TOKEN in .env file or export it"
echo "3. Upload kaggle_iris_training.ipynb to Kaggle"
echo "4. Enable P100 GPU and run the notebook"
echo "5. Download trained model and extract to models/iris-gemma-local/"
echo "6. Test with: python src/train_enhanced.py --model_type local"
echo ""
echo "📋 For detailed instructions, see:"
echo "   - STEP_BY_STEP_GUIDE.md (Option 4)"
echo "   - Run: python demo_kaggle_local.py"
echo ""
echo "🎉 Setup complete! Ready for Kaggle local training!"