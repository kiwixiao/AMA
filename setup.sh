#!/bin/bash

# AMA (Airway Metrics Analysis) Setup Script
# Handles both conda and pip environments automatically

set -e

# Pass --dev for editable install (developers who modify source code)
# Default is regular install (end users)
DEV_MODE=false
if [ "$1" = "--dev" ]; then
    DEV_MODE=true
fi

echo "AMA Setup"
echo "========="
if [ "$DEV_MODE" = true ]; then
    echo "(developer mode - editable install)"
fi
echo ""

# Detect environment manager
USE_CONDA=false
if command -v conda &> /dev/null; then
    USE_CONDA=true
fi

# --- Conda path ---
if [ "$USE_CONDA" = true ]; then
    echo "Conda detected."

    ENV_NAME="ama"

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Environment '${ENV_NAME}' already exists."
        echo "Updating..."
        conda env update -n "${ENV_NAME}" -f environment.yml --prune
    else
        echo "Creating conda environment '${ENV_NAME}'..."
        conda env create -f environment.yml
    fi

    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    if [ "$DEV_MODE" = true ]; then
        pip install -e .
    else
        pip install .
    fi

    echo ""
    echo "Done. To use:"
    echo "  conda activate ${ENV_NAME}"
    echo "  ama --listsubjects"

# --- Pip path ---
else
    echo "Conda not found, using pip."

    VENV_DIR=".venv"

    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment in ${VENV_DIR}..."
        python -m venv "${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"
    if [ "$DEV_MODE" = true ]; then
        pip install -e ".[gui]"
    else
        pip install ".[gui]"
    fi

    echo ""
    echo "Done. To use:"
    echo "  source ${VENV_DIR}/bin/activate"
    echo "  ama --listsubjects"
fi

# --- Verify ---
echo ""
echo "Verifying installation..."

python -c "
import sys
required = ['numpy', 'pandas', 'scipy', 'matplotlib', 'plotly', 'h5py', 'sklearn']
optional = ['pyvista', 'pyvistaqt']
ok = True
for m in required:
    try:
        __import__(m)
        print(f'  OK  {m}')
    except ImportError:
        print(f'  FAIL {m}')
        ok = False
for m in optional:
    try:
        __import__(m)
        print(f'  OK  {m}')
    except ImportError:
        print(f'  SKIP {m} (optional, needed for --point-picker GUI)')
if not ok:
    print('Some required packages failed to import.')
    sys.exit(1)
"

if ama --listsubjects > /dev/null 2>&1; then
    echo "  OK  ama command"
else
    echo "  WARN ama --listsubjects returned non-zero (may need data files)"
fi

echo ""
echo "Setup complete."
echo ""
echo "3-Phase Workflow:"
echo "  Phase 1: ama --subject SUBJ --prepare --flow-profile FLOWProfile.csv"
echo "  Phase 2: ama --subject SUBJ --point-picker"
echo "  Phase 3: ama --subject SUBJ --plotting"
