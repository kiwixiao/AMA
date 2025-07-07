#!/bin/bash

# CFD Analysis Pipeline Setup Script
# This script automates the installation of the CFD analysis environment

set -e  # Exit on any error

echo "🚀 CFD Analysis Pipeline Setup"
echo "=============================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found"
    USE_CONDA=true
else
    echo "⚠️  Conda not found, using pip instead"
    USE_CONDA=false
fi

# Setup with conda (recommended)
if [ "$USE_CONDA" = true ]; then
    echo "📦 Setting up conda environment..."
    
    # Check if environment already exists
    if conda env list | grep -q "cfd-analysis"; then
        echo "⚠️  Environment 'cfd-analysis' already exists"
        read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n cfd-analysis
        else
            echo "❌ Setup cancelled"
            exit 1
        fi
    fi
    
    # Create environment
    echo "🔧 Creating conda environment..."
    conda env create -f environment.yml
    
    echo "✅ Conda environment created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate cfd-analysis"
    
else
    # Setup with pip
    echo "📦 Setting up pip environment..."
    
    # Check if we're in a virtual environment
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        echo "⚠️  No virtual environment detected"
        read -p "Do you want to create a virtual environment? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python -m venv cfd-env
            source cfd-env/bin/activate
            echo "✅ Virtual environment created and activated"
        else
            echo "⚠️  Installing packages globally (not recommended)"
        fi
    fi
    
    # Install packages
    echo "🔧 Installing Python packages..."
    pip install -r requirements.txt
    
    echo "✅ Pip packages installed successfully!"
    echo ""
    if [[ -n "${VIRTUAL_ENV}" ]]; then
        echo "Virtual environment is active. To reactivate later, run:"
        echo "  source cfd-env/bin/activate"
    fi
fi

# Test installation
echo ""
echo "🧪 Testing installation..."
if python src/main.py --listsubjects &> /dev/null; then
    echo "✅ Installation test passed!"
else
    echo "⚠️  Installation test failed - some dependencies may be missing"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
if [ "$USE_CONDA" = true ]; then
    echo "   conda activate cfd-analysis"
else
    echo "   source cfd-env/bin/activate  # if using virtual environment"
fi
echo ""
echo "2. Test with your data:"
echo "   python src/main.py --listsubjects"
echo "   python src/main.py --subject YOUR_SUBJECT_NAME"
echo ""
echo "3. For custom raw data directory:"
echo "   python src/main.py --subject YOUR_SUBJECT_NAME --rawdir YOUR_RAW_DIR"
echo ""
echo "📖 See README.md for detailed usage instructions" 