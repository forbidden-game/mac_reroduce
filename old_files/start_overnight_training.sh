#!/bin/bash

# Quick launcher for overnight training
# Usage: ./start_overnight_training.sh

echo "🚀 MAC Overnight Training Launcher"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "Pretraing_MAC.PY" ]; then
    echo "❌ Error: Pretraing_MAC.PY not found!"
    echo "   Please run this script from the mac_reroduce directory"
    exit 1
fi

# Check Python environment
echo "🔍 Checking environment..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || {
    echo "❌ Error: PyTorch not available!"
    exit 1
}

# Check CUDA
python -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')"

# Check datasets
echo "🔍 Checking datasets..."
datasets=("RML201610A" "RML201610B" "RML2018")
for dataset in "${datasets[@]}"; do
    if python -c "from util import load_RML2016; import argparse; args = argparse.Namespace(); args.ab_choose='$dataset'; args.snr_tat=6; args.RML2016a_path='data/'; args.RML2016b_path='data/'; args.RML2018_path='data/'; load_RML2016(args)" &>/dev/null; then
        echo "✅ $dataset dataset ready"
    else
        echo "⚠️  $dataset dataset not found or has issues"
    fi
done

echo ""
echo "🌙 Ready to start overnight training!"
echo "📊 Will train 3 datasets × 200 epochs each = 600 total epochs"
echo "⏰ Estimated time: 3-6 hours"
echo ""

# Ask for confirmation
read -p "Start overnight training now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting overnight training..."
    python overnight_training.py --start-immediately
else
    echo "Training cancelled. You can start manually with:"
    echo "   python overnight_training.py --start-immediately"
fi
