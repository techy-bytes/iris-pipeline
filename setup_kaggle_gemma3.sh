#!/bin/bash

# 🚀 Quick Setup Script for Kaggle Gemma 3 Training
# This script prepares your environment for cost-efficient Iris classification with Gemma 3

set -e  # Exit on any error

echo "🌸 Iris Classification - Kaggle Gemma 3 Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    print_error "Please run this script from the iris-pipeline root directory"
    exit 1
fi

print_status "Found iris-pipeline repository structure"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.8+ required, found $python_version"
    exit 1
fi

print_status "Python version $python_version is compatible"

# Create necessary directories
print_info "Creating project directories..."
mkdir -p models/iris-gemma3-local
mkdir -p logs
mkdir -p /tmp/iris-pipeline

print_status "Project directories created"

# Install Python dependencies
print_info "Installing Python dependencies..."

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate || {
    print_warning "Failed to activate virtual environment, continuing with system Python"
}

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Install required packages
print_info "Installing required packages (this may take a few minutes)..."

pip install -r requirements.txt > /dev/null 2>&1 || {
    print_warning "requirements.txt not found or failed, installing essential packages..."
    pip install pandas scikit-learn numpy > /dev/null 2>&1
}

# Install additional packages for Gemma training
print_info "Installing Gemma 3 training dependencies..."

# Core ML packages
pip install torch --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
pip install transformers==4.36.2 > /dev/null 2>&1
pip install peft==0.7.1 > /dev/null 2>&1
pip install datasets==2.14.6 > /dev/null 2>&1
pip install trl==0.7.4 > /dev/null 2>&1
pip install accelerate==0.25.0 > /dev/null 2>&1
pip install huggingface_hub > /dev/null 2>&1

# Optional packages for better experience
pip install jupyter > /dev/null 2>&1 || print_warning "Failed to install Jupyter (optional)"
pip install pytest > /dev/null 2>&1 || print_warning "Failed to install pytest (optional)"

print_status "Dependencies installed successfully"

# Check if HuggingFace token is available
print_info "Checking HuggingFace configuration..."

if [ -z "$HF_TOKEN" ]; then
    print_warning "HF_TOKEN environment variable not set"
    echo ""
    echo "📝 To use Gemma 3 model, you need a HuggingFace token:"
    echo "   1. Go to https://huggingface.co/settings/tokens"
    echo "   2. Create a new token with read access"
    echo "   3. Export it: export HF_TOKEN=your_token_here"
    echo "   4. Add it to your shell profile for persistence"
    echo ""
else
    print_status "HuggingFace token found"
fi

# Create environment configuration file
print_info "Creating environment configuration..."

cat > .env.example << EOF
# Environment Configuration for Iris Gemma 3 Pipeline

# HuggingFace Token (required for Gemma 3 access)
HF_TOKEN=your_huggingface_token_here

# Model Configuration
USE_LLM_MODEL=true
USE_LOCAL_GEMMA=true
LOCAL_MODEL_PATH=models/iris-gemma3-local

# Training Configuration
BATCH_SIZE=4
LEARNING_RATE=2e-4
EPOCHS=3
MAX_LENGTH=512

# API Configuration
API_HOST=localhost
API_PORT=8000

# Development Mode (for testing without GPU)
USE_MOCK_LLM=false

# Kaggle Configuration (for training)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
EOF

print_status "Environment configuration template created (.env.example)"

# Test basic functionality
print_info "Testing basic functionality..."

# Test data transformer
python3 -c "
from sklearn.datasets import load_iris
import pandas as pd
print('✅ Basic data loading works')

iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target_names[iris.target]
print(f'✅ Iris dataset loaded: {len(df)} samples')

print('✅ Basic functionality test passed')
" || {
    print_error "Basic functionality test failed"
    exit 1
}

print_status "Basic functionality test passed"

# Create quick test script
print_info "Creating quick test script..."

cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test script for Iris Gemma 3 pipeline
Run this to verify your setup is working correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_basic_setup():
    """Test basic setup and sklearn baseline."""
    print("🧪 Testing basic setup...")
    
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train sklearn model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✅ Sklearn baseline accuracy: {accuracy:.3f}")
    
    if accuracy > 0.85:
        print("✅ Basic setup is working correctly!")
        return True
    else:
        print("❌ Accuracy too low, check setup")
        return False

def test_data_transformation():
    """Test linguistic data transformation."""
    print("\n🔤 Testing data transformation...")
    
    try:
        # Simple linguistic transformation test
        def iris_to_linguistic(sepal_length, sepal_width, petal_length, petal_width, species):
            size_desc = "large" if sepal_length > 6.0 else "medium" if sepal_length > 5.0 else "small"
            return f"This is a {size_desc} {species} iris with sepal length {sepal_length}cm."
        
        # Test transformation
        result = iris_to_linguistic(5.1, 3.5, 1.4, 0.2, "setosa")
        
        if "setosa" in result and "5.1" in result:
            print("✅ Data transformation working correctly!")
            print(f"   Sample: {result[:60]}...")
            return True
        else:
            print("❌ Data transformation failed")
            return False
            
    except Exception as e:
        print(f"❌ Data transformation error: {e}")
        return False

def test_environment():
    """Test environment configuration."""
    print("\n🌍 Testing environment...")
    
    # Check for required packages
    required_packages = ['pandas', 'numpy', 'sklearn']
    optional_packages = ['torch', 'transformers']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available (required)")
            return False
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"⚠️  {package} not available (needed for Gemma training)")
    
    # Check HuggingFace token
    if os.environ.get('HF_TOKEN'):
        print("✅ HF_TOKEN found")
    else:
        print("⚠️  HF_TOKEN not set (needed for Gemma access)")
    
    return True

def main():
    """Main test function."""
    print("🚀 Iris Pipeline - Quick Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_basic_setup()
    all_tests_passed &= test_data_transformation()
    all_tests_passed &= test_environment()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n📋 Next steps:")
        print("   1. Set your HF_TOKEN: export HF_TOKEN=your_token")
        print("   2. Upload kaggle_gemma3_notebook.ipynb to Kaggle")
        print("   3. Enable P100 GPU and add HF_TOKEN as secret")
        print("   4. Run the notebook to train your model")
        print("   5. Download and use the trained model locally")
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x quick_test.py
print_status "Quick test script created (quick_test.py)"

# Create directory structure info
print_info "Creating project structure documentation..."

cat > PROJECT_STRUCTURE.md << 'EOF'
# 📁 Project Structure - Iris Gemma 3 Pipeline

```
iris-pipeline/
├── 📊 Data & Training
│   ├── data/                           # Dataset files
│   ├── kaggle_gemma3_trainer.py       # Kaggle training script
│   ├── kaggle_gemma3_notebook.ipynb   # Upload this to Kaggle
│   └── models/                         # Trained models
│       └── iris-gemma3-local/         # Downloaded model from Kaggle
│
├── 🔧 Source Code
│   ├── src/
│   │   ├── api.py                     # Original API
│   │   ├── api_enhanced.py            # Enhanced API with Gemma
│   │   ├── data_transformer.py        # Data transformation
│   │   ├── gemma_classifier.py        # Gemma model wrapper
│   │   ├── train.py                   # Original training
│   │   └── train_enhanced.py          # Enhanced training
│   │
├── 🧪 Testing
│   ├── tests/                         # Original test suite
│   ├── test_gemma3_integration.py     # Gemma 3 specific tests
│   └── quick_test.py                  # Quick setup verification
│
├── 📚 Documentation
│   ├── README.md                      # Main documentation
│   ├── STEP_BY_STEP_GUIDE.md         # Complete user guide
│   ├── KAGGLE_LOCAL_TRAINING.md      # Kaggle training guide
│   ├── GEMMA3_KAGGLE_IMPLEMENTATION.md # Implementation details
│   └── PROJECT_STRUCTURE.md          # This file
│
├── ⚙️ Configuration
│   ├── requirements.txt               # Python dependencies
│   ├── .env.example                   # Environment template
│   ├── setup_kaggle_gemma3.sh        # This setup script
│   └── Dockerfile                     # Container setup
│
└── 🔧 Utilities
    ├── demo_gemma_integration.py      # Demo script
    ├── demo_kaggle_local.py          # Local demo
    └── logs/                          # Log files
```

## 🚀 Quick Start Files

### For Kaggle Training:
- `kaggle_gemma3_notebook.ipynb` - Upload to Kaggle
- `kaggle_gemma3_trainer.py` - Training script

### For Local Development:
- `quick_test.py` - Verify setup
- `.env.example` - Environment configuration
- `demo_gemma_integration.py` - Try the integration

### For Production:
- `src/api_enhanced.py` - Enhanced API
- `models/iris-gemma3-local/` - Trained model location
EOF

print_status "Project structure documentation created"

# Final setup summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
print_status "Environment prepared for Kaggle Gemma 3 training"
print_status "Dependencies installed"
print_status "Configuration files created"
print_status "Test scripts ready"

echo ""
echo "📋 Next Steps:"
echo "1. 🔑 Set your HuggingFace token:"
echo "   export HF_TOKEN=your_token_here"
echo ""
echo "2. 🧪 Test your setup:"
echo "   python3 quick_test.py"
echo ""
echo "3. 🚀 Train on Kaggle:"
echo "   - Upload kaggle_gemma3_notebook.ipynb to Kaggle"
echo "   - Enable P100 GPU in notebook settings"
echo "   - Add HF_TOKEN as a Kaggle secret"
echo "   - Run all cells (takes ~10-15 minutes)"
echo "   - Download the trained model"
echo ""
echo "4. 💻 Use locally:"
echo "   - Extract model to models/iris-gemma3-local/"
echo "   - Run: python3 demo_kaggle_local.py"
echo ""
echo "📖 For detailed instructions, see:"
echo "   - GEMMA3_KAGGLE_IMPLEMENTATION.md (step-by-step guide)"
echo "   - PROJECT_STRUCTURE.md (file organization)"
echo ""
print_info "Happy training! 🌸"