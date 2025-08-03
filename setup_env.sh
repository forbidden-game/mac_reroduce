#!/bin/bash

# MAC Framework Environment Setup Script
# This script sets up the Python environment using uv

set -e

echo "🚀 Setting up MAC Framework environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ uv is available"

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv

# Activate environment (source it manually after script)
echo "📋 Virtual environment created at .venv/"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

echo "🎯 Testing MAC backbone implementation..."
uv run python test_mac_backbone.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate   # On Linux/Mac"
echo "  .venv\\Scripts\\activate      # On Windows"
echo ""
echo "Or use uv run for any command:"
echo "  uv run python Pretraing_MAC.PY --help"
echo "  uv run python test_mac_backbone.py"
echo ""
echo "📖 See README.md for detailed usage instructions"
