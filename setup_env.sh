#!/bin/bash

# MAC Framework Environment Setup Script
# This script sets up the Python environment using uv

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
