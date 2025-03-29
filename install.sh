#!/bin/bash
# LlamaForge Installation Script

set -e

VENV_DIR="venv"
INSTALL_TYPE="basic"
CUDA_AVAILABLE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --with-cuda)
      CUDA_AVAILABLE=true
      shift
      ;;
    --all)
      INSTALL_TYPE="all"
      shift
      ;;
    --server)
      INSTALL_TYPE="server"
      shift
      ;;
    --dev)
      INSTALL_TYPE="dev"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 8 ]; then
  echo "Error: Python 3.8 or higher is required (found $PYTHON_VERSION)"
  exit 1
fi

echo "=== LlamaForge Installation ==="
echo "Using Python $PYTHON_VERSION"
echo "Installing in virtual environment: $VENV_DIR"
echo "Installation type: $INSTALL_TYPE"
if [ "$CUDA_AVAILABLE" = true ]; then
  echo "CUDA support: Enabled"
else
  echo "CUDA support: Disabled"
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  echo "Using existing virtual environment in $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Update pip
echo "Updating pip..."
pip install --upgrade pip

# Install LlamaForge
echo "Installing LlamaForge..."

case $INSTALL_TYPE in
  "basic")
    pip install -e .
    ;;
  "server")
    pip install -e ".[server]"
    ;;
  "dev")
    pip install -e ".[dev]"
    ;;
  "all")
    # Install based on CUDA availability
    if [ "$CUDA_AVAILABLE" = true ]; then
      echo "Installing with CUDA support..."
      pip install -e ".[all]"
    else
      echo "Installing without CUDA support..."
      # For CPU-only installations, we need to be specific about torch
      pip install torch --index-url https://download.pytorch.org/whl/cpu
      pip install -e ".[all]"
    fi
    ;;
esac

# Create configuration directory
CONFIG_DIR="$HOME/.llamaforge"
if [ ! -d "$CONFIG_DIR" ]; then
  echo "Creating configuration directory: $CONFIG_DIR"
  mkdir -p "$CONFIG_DIR"
  mkdir -p "$CONFIG_DIR/models"
fi

# Initialize configuration
echo "Initializing configuration..."
python -c "from llamaforge import LlamaForge; forge = LlamaForge(); forge.config.save()"

echo "=== Installation Complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To get started, try:"
echo "  llamaforge list models"
echo "  llamaforge help"
echo ""
echo "Check the examples directory for usage examples." 