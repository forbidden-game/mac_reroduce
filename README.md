# Multi-Representation Attentive Contrastive Learning (MAC) for AMR

This repository contains the official PyTorch implementation for the paper: **"Multi-representation domain attentive contrastive learning based unsupervised automatic modulation recognition"**.

The project reproduces the MAC model, which leverages unsupervised, multi-domain contrastive pre-training and an attention-based fine-tuning stage to achieve state-of-the-art performance in Automatic Modulation Recognition (AMR).

## Model Architecture

The implementation correctly mirrors the architecture described in the paper:

1.  **Shared Backbone (`models/backbone.py`)**: A CNN-based feature extractor is shared across all signal representations to learn common underlying features.
2.  **Multi-Domain Representation (`models/backbone.py`)**: The model transforms raw I-Q signals into four additional domains:
    *   Amplitude-Phase (AN)
    *   Instantaneous Frequency (AF)
    *   Wavelet (WT)
    *   FFT
3.  **Contrastive Pre-Training (`Pretraing_MAC.PY`)**: The model is first trained on unlabeled data using a contrastive loss (NCE). It learns to distinguish between different signals by maximizing the similarity between different representations of the same signal (inter-domain) and augmented versions of the source signal (intra-domain).
4.  **Domain Attention (DA) Fine-Tuning (`models/LinearModel.py`, `Fine_tuning_Times.py`)**: After pre-training, a classifier with a Domain Attention module is attached. This module learns to dynamically weigh the importance of each representation domain's features, which are then used for the final classification.

## Setup

### 1. Dependencies

First, create and activate a Python virtual environment using `uv`:

```bash
uv venv
source .venv/bin/activate
```

Then, install the required dependencies from `pyproject.toml` using `uv`:

```bash
uv pip install -e .
```

This command will install all the necessary packages in editable mode.

### 2. Dataset Preparation

The code expects the datasets to be pre-processed and saved as pickled PyTorch `TensorDataset` objects. As requested, the following instructions assume all data is placed within a root `data/` folder.

Create the following directory structure:

```
.
├── data/
│   ├── RML2016.10a/
│   │   ├── 0_train_MV_dataset
│   │   ├── 0_test_MV_dataset
│   │   ├── 2_train_MV_dataset
│   │   ├── 2_test_MV_dataset
│   │   └── ... (and so on for each SNR)
│   └── RML2016.10b/
│       ├── 0_MT4_train_dataset
│       ├── 0_MT4_test_dataset
│       └── ...
├── models/
├── NCE/
├── Pretraing_MAC.PY
├── Fine_tuning_Times.py
└── ... (other project files)
```

**Note**: The data loading script (`util.py`) constructs file paths based on the dataset name and SNR. Ensure your pickled files match the naming convention shown above (e.g., `{snr}_train_MV_dataset`).

## How to Run

The model reproduction is a two-step process: unsupervised pre-training followed by supervised fine-tuning.

### Step 1: Unsupervised Pre-training

Run the `Pretraing_MAC.PY` script to train the `MAC_backbone` on unlabeled data. This will save a model checkpoint that will be used in the next step.

**Example Command:**

```bash
python Pretraing_MAC.PY \
    --ab_choose RML201610A \
    --RML2016a_path ./data/RML2016.10a/ \
    --snr_tat 18 \
    --view_chose ALL \
    --batch_size 64 \
    --epochs 240 \
    --learning_rate 0.01 \
    --nce_k 16384 \
    --model_path ./saved_models/ \
    --tb_path ./tensorboard_logs/
```

**Key Arguments:**
*   `--ab_choose`: The dataset to use (`RML201610A`, `RML201610B`, `RML2018`).
*   `--RML2016a_path`: Path to the dataset directory. **Update this for your chosen dataset.**
*   `--snr_tat`: The Signal-to-Noise Ratio (SNR) of the data to use for training.
*   `--view_chose`: Use `ALL` for the full multi-domain model as described in the paper.
*   `--epochs`: Number of pre-training epochs. The paper uses 240.
*   `--model_path`: Directory to save the trained model checkpoints.
*   `--tb_path`: Directory to save TensorBoard logs.

A checkpoint file (e.g., `ckpt_epoch_240.pth`) will be saved in the specified model path.

### Step 2: Fine-tuning with Domain Attention

Run the `Fine_tuning_Times.py` script to load the pre-trained backbone, attach the Domain Attention classifier, and fine-tune on a small number of labeled samples.

**Example Command:**

```bash
python Fine_tuning_Times.py \
    --ab_choose RML201610A \
    --RML2016a_path ./data/RML2016.10a/ \
    --model_path ./saved_models/MAC_backbone/ckpt_epoch_240.pth \
    --snr_tat 18 \
    --view_chose ALL \
    --N_shot 50 \
    --epochs 120 \
    --learning_rate 0.0001
```

**Key Arguments:**
*   `--model_path`: **Crucially**, this must be the full path to the checkpoint file saved during pre-training.
*   `--N_shot`: The number of labeled samples per class to use for fine-tuning (e.g., 10, 20, 50, 100).
*   `--epochs`: Number of fine-tuning epochs. The paper uses up to 120.
*   Other arguments like `--ab_choose`, `--RML2016a_path`, `--snr_tat`, and `--view_chose` should match the ones used in pre-training.

This script will run the fine-tuning process and print the final classification accuracy, which should reproduce the results reported in the paper for the given dataset and `N_shot` value.
