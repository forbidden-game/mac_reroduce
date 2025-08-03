# MAC Framework - Multi-representation domain Attentive Contrastive learning

Implementation of the paper "Multi-representation domain attentive contrastive learning based unsupervised automatic modulation recognition" published in Nature Communications.

## 📋 Overview

This framework implements an unsupervised learning approach for automatic modulation recognition (AMR) in wireless communications using multi-domain contrastive learning.

### Key Features

- **Multi-domain signal processing**: I-Q, Amplitude-Phase, Instantaneous Frequency, Wavelet, FFT
- **Contrastive learning**: Inter-domain and intra-domain contrastive mechanisms  
- **Domain attention**: Dynamic selection of representation domains
- **Few-shot learning**: Effective with limited labeled data
- **Multiple datasets**: Support for RML2016.10A/B and RML2018.01A

## 🚀 Quick Start with UV

### Prerequisites

Install [uv](https://github.com/astral-sh/uv) - the fast Python package installer:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environment Setup

1. **Create and activate environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   # Basic installation
   uv pip install -e .
   
   # Or with GPU support
   uv pip install -e ".[gpu]"
   
   # Or with development tools
   uv pip install -e ".[dev]"
   
   # Or everything
   uv pip install -e ".[dev,docs,jupyter,gpu]"
   ```

3. **Quick dependency sync**:
   ```bash
   uv sync  # Installs all dependencies from pyproject.toml
   ```

### Alternative: Direct Script Installation

```bash
# Install specific dependencies for running the framework
uv pip install torch torchvision numpy scikit-learn matplotlib seaborn tensorboard-logger PyWavelets tqdm h5py
```

## 🏃‍♂️ Running the Framework

### Using UV Scripts (Recommended)

After installation, you can use the configured scripts:

```bash
# Test the implementation
uv run mac-test

# Run pretraining  
uv run mac-pretrain --ab_choose RML201610A --view_chose ALL --epochs 240

# Run fine-tuning
uv run mac-finetune --model_path result/ckpt_epoch_240.pth --N_shot 50
```

### Using Python Directly

```bash
# Test implementation
uv run python test_mac_backbone.py

# Pretraining (240 epochs)
uv run python Pretraing_MAC.PY --ab_choose RML201610A --view_chose ALL --mod_l AN --epochs 240

# Fine-tuning with few-shot learning
uv run python Fine_tuning_Times.py --model_path result/ckpt_epoch_240.pth --N_shot 50 --epochs 120
```

## 📊 Dataset Support

The framework supports the following datasets:

- **RML2016.10A**: 11 modulation types, 220,000 samples
- **RML2016.10B**: 10 modulation types, 1,200,000 samples  
- **RML2018.01A**: 24 modulation types, 2,555,904 samples

Download datasets from [DeepSig](https://www.deepsig.ai/datasets/).

## 🏗️ Architecture

### MAC Backbone Components

1. **Multi-domain Transformations**:
   - **AN**: Amplitude-Phase representation
   - **AF**: Instantaneous Frequency
   - **WT**: Wavelet Transform
   - **FFT**: Frequency spectrum

2. **Shared CNN Encoder**: 4-layer CNN with batch normalization

3. **Contrastive Learning**:
   - Inter-domain: Between I-Q and transformed domains
   - Intra-domain: Augmented I-Q samples

4. **Domain Attention**: Dynamic weighting of domain features

### Training Pipeline

1. **Unsupervised Pretraining** (240 epochs):
   - Multi-domain contrastive learning
   - "I-Q single centralization" strategy
   - Memory bank with momentum updates

2. **Supervised Fine-tuning** (120 epochs):
   - Linear evaluation or full fine-tuning
   - Few-shot learning support
   - Domain attention integration

## 🔧 Configuration

### Key Hyperparameters

```python
# Contrastive learning
nce_k = 16384      # Number of negative samples
nce_t = 0.07       # Temperature parameter  
nce_m = 0.9        # Momentum coefficient

# Training
batch_size = 64    # Pretraining batch size
feat_dim = 128     # Feature dimension
learning_rate = 0.03  # Pretraining learning rate
```

### Environment Variables

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0

# PyTorch settings  
export TORCH_HOME=./torch_cache
```

## 📈 Expected Performance

Based on the paper results:

| Dataset | Few-shot Samples | Accuracy |
|---------|------------------|----------|
| RML2016.10A | 100 (10%) | ~78.93% |
| RML2016.10B | 100 (1.67%) | ~74.15% |
| RML2018.01A | 100 (2.44%) | ~79.24% |

## 🛠️ Development

### Code Quality Tools

```bash
# Format code
uv run black .
uv run isort .

# Linting
uv run flake8 .
uv run ruff check .

# Type checking
uv run mypy models/

# Testing
uv run pytest tests/
```

### Jupyter Development

```bash
# Install jupyter extras
uv pip install -e ".[jupyter]"

# Start JupyterLab
uv run jupyter lab
```

## 📂 Project Structure

```
mac-framework/
├── models/
│   ├── __init__.py
│   ├── backbone.py          # MAC_backbone implementation
│   └── LinearModel.py       # Linear classifiers with domain attention
├── NCE/
│   ├── NCEAverage.py        # Memory bank management
│   └── NCECriterion.py      # Contrastive loss functions
├── data/                    # Dataset storage
├── Pretraing_MAC.PY         # Unsupervised pretraining script
├── Fine_tuning_Times.py     # Supervised fine-tuning script
├── test_mac_backbone.py     # Test implementation
├── util.py                  # Utility functions
├── buildv1.py               # Data augmentation functions
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## 🎯 Citation

```bibtex
@article{li2025mac,
  title={Multi-representation domain attentive contrastive learning based unsupervised automatic modulation recognition},
  author={Li, Yu and Shi, Xiaoran and Tan, Haoyue and Zhang, Zhenxi and Yang, Xinyao and Zhou, Feng},
  journal={Nature Communications},
  volume={16},
  pages={5951},
  year={2025},
  doi={10.1038/s41467-025-60921-z}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📞 Support

- Create an issue on GitHub for bug reports
- Check the paper for theoretical background
- See test_mac_backbone.py for usage examples

---

**Note**: This implementation requires PyTorch and is optimized for CUDA GPUs. CPU training is supported but significantly slower.
