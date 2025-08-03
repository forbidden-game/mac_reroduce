# MAC Training Analysis Tools

This document provides comprehensive usage instructions for the three analysis tools created for your MAC (Modulation Aware Contrastive) learning project.

## 📊 Overview of Tools

### 1. **`analyze_datasets.py`** - Dataset Analysis & Visualization
- Comprehensive analysis of RML2016.10a, RML2016.10b, and RML2018 datasets
- Signal characteristics visualization (constellations, time series, PSD)
- Dataset statistics and class distribution analysis

### 2. **`visualize_features.py`** - Feature Visualization Tools
- Extract and visualize learned features from trained MAC models
- t-SNE, UMAP, and PCA embeddings for feature analysis
- Class separation analysis and confusion matrices
- Feature evolution tracking across training epochs

### 3. **`training_dashboard.py`** - Training Monitoring Dashboard
- Real-time training progress monitoring
- System resource tracking (GPU, CPU, RAM)
- Loss curves and learning rate visualization
- Training efficiency analysis and recommendations

---

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Install required packages
pip install matplotlib seaborn pandas scikit-learn umap-learn psutil GPUtil tqdm
```

### 1. Dataset Analysis

**Basic Usage:**
```bash
# Analyze all datasets
python analyze_datasets.py --datasets all

# Analyze specific dataset
python analyze_datasets.py --datasets rml2016a

# Custom output directory
python analyze_datasets.py --datasets all --output-dir my_analysis_results
```

**What it generates:**
- Class distribution plots
- SNR level analysis
- Constellation diagrams
- Time series examples
- Power spectral density plots
- Comprehensive comparison report

### 2. Feature Visualization

**Basic Usage:**
```bash
# Visualize features from a trained model
python visualize_features.py \
    --checkpoint path/to/your/model_checkpoint.pth \
    --dataset RML201610A \
    --snr 6 \
    --output-dir feature_results

# Use different embedding method
python visualize_features.py \
    --checkpoint path/to/checkpoint.pth \
    --dataset RML201610A \
    --embedding-method umap \
    --max-samples 2000

# Analyze feature evolution across epochs
python visualize_features.py \
    --checkpoint path/to/checkpoint.pth \
    --dataset RML201610A \
    --evolution-dir path/to/checkpoint_directory/
```

**What it generates:**
- Feature embeddings (t-SNE/UMAP/PCA) for all domains
- Class separation analysis
- Feature statistics across domains
- Evolution plots (if multiple checkpoints provided)

### 3. Training Dashboard

**Demo Mode (Test the dashboard):**
```bash
# Run demo to see dashboard capabilities
python training_dashboard.py --mode demo

# Analyze existing tensorboard logs
python training_dashboard.py --mode analyze --log-dir path/to/logs

# Get integration instructions
python training_dashboard.py --mode integrate
```

**Integration with Training Script:**
Add this to your `Pretraing_MAC.PY`:

```python
# At the top of your script
from training_dashboard import EnhancedTrainingMonitor

# Initialize monitor (in main function)
monitor = EnhancedTrainingMonitor(dashboard_dir="training_dashboard")
monitor.start_monitoring(args.epochs)

# At start of each epoch
monitor.log_epoch_start(epoch)

# In training loop (for each batch)
batch_metrics = {
    'train_loss': loss.item(),
    'l_loss': l_loss.item(),
    'ab_loss': ab_loss.item(),
    'td_loss': T_loss_meter.val,
    'sd_loss': l_loss_meter.val,
    'batch_time': batch_time.val,
    'data_time': data_time.val
}
monitor.log_batch_metrics(idx, len(train_loader), batch_metrics)

# At end of each epoch
epoch_metrics = {'learning_rate': optimizer.param_groups[0]['lr']}
monitor.log_epoch_end(epoch, epoch_metrics)

# At end of training
monitor.stop_monitoring()
```

---

## 📋 Detailed Usage Examples

### Complete Workflow Example

```bash
# Step 1: Analyze your datasets first
python analyze_datasets.py --datasets all --output-dir dataset_analysis
# Results: dataset_analysis/

# Step 2: Train your model with monitoring
# (Integrate training_dashboard.py into Pretraing_MAC.PY as shown above)
python Pretraing_MAC.PY --ab_choose RML201610A --epochs 240
# Results: training_dashboard/

# Step 3: Analyze learned features
python visualize_features.py \
    --checkpoint training_dashboard/ckpt_epoch_120.pth \
    --dataset RML201610A \
    --snr 6 \
    --output-dir feature_analysis_epoch120
# Results: feature_analysis_epoch120/

# Step 4: Compare features across epochs
python visualize_features.py \
    --checkpoint training_dashboard/ckpt_epoch_240.pth \
    --dataset RML201610A \
    --evolution-dir training_dashboard/ \
    --output-dir feature_evolution_analysis
```

### Advanced Options

**Dataset Analysis Options:**
```bash
python analyze_datasets.py \
    --datasets rml2016a rml2016b \
    --output-dir comparative_analysis
```

**Feature Visualization Options:**
```bash
python visualize_features.py \
    --checkpoint model.pth \
    --dataset RML201610A \
    --snr 6 \
    --max-samples 10000 \
    --embedding-method umap \
    --output-dir detailed_features
```

**Training Dashboard Options:**
```bash
# Analyze existing training logs
python training_dashboard.py \
    --mode analyze \
    --log-dir 2018pretrain_logs_ \
    --dashboard-dir log_analysis_results
```

---

## 📁 Output Structure

### Dataset Analysis Output
```
analysis_results/
├── RML2016.10a_class_distribution.png
├── RML2016.10a_snr_distribution.png
├── RML2016.10a_heatmap.png
├── RML2016.10a_constellations.png
├── RML2016.10a_time_series.png
├── RML2016.10a_psd.png
├── dataset_comparison.png
└── analysis_report.txt
```

### Feature Visualization Output
```
feature_analysis/
├── RML201610A_tsne_embeddings.png
├── RML201610A_separation_analysis.png
├── RML201610A_comprehensive_analysis.png
├── RML201610A_feature_analysis_report.txt
└── RML201610A_confusion_matrix.png (if applicable)
```

### Training Dashboard Output
```
training_dashboard/
├── comprehensive_dashboard.png      # Main dashboard
├── loss_evolution.png             # Loss curves
├── system_resources.png           # GPU/CPU/RAM usage
├── training_progress.png          # Progress and timing
├── learning_rate.png              # LR schedule
├── training_metrics.json          # Raw metrics data
└── training_report.txt            # Summary report
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors:**
```bash
# Install missing packages
pip install umap-learn GPUtil psutil seaborn
```

**2. CUDA/GPU Issues:**
```python
# If GPU not available, models will automatically use CPU
# Check GPU availability: torch.cuda.is_available()
```

**3. Memory Issues:**
```bash
# Reduce max_samples for feature visualization
python visualize_features.py --max-samples 1000 ...

# Reduce batch_size for training
python Pretraing_MAC.PY --batch_size 32 ...
```

**4. Matplotlib Issues:**
```python
# If running on server without display
import matplotlib
matplotlib.use('Agg')
```

### Dataset Path Issues

Make sure your data structure matches:
```
data/
├── RML2016.10a_dict.pkl           # For RML2016.10a analysis
├── processed/
│   ├── RML2016.10b/               # For RML2016.10b analysis
│   └── RML2018/                   # For RML2018 analysis
└── 6_train_MV_dataset             # For training
```

---

## 🎯 Key Features Explained

### Dataset Analysis (`analyze_datasets.py`)

1. **Train/Test Split Analysis:**
   - Shows exactly how data is split (80/20)
   - Per-SNR level analysis
   - Class balance verification

2. **Modulation Classes:**
   - **RML2016.10a:** 11 classes (includes AM-SSB)
   - **RML2016.10b:** 10 classes (excludes AM-SSB)
   - **RML2018:** 24 classes

3. **Signal Visualizations:**
   - Constellation diagrams for different modulations
   - Time-domain I/Q signals
   - Frequency-domain power spectral density

### Feature Visualization (`visualize_features.py`)

1. **Multi-Domain Analysis:**
   - L-domain (time-domain)
   - TD-domain (frequency-domain)
   - SD-domain (spatial-domain)
   - All sub-domains (TD1, TD2, TD3)

2. **Embedding Methods:**
   - **t-SNE:** Best for local structure preservation
   - **UMAP:** Good balance of local/global structure
   - **PCA:** Linear dimensionality reduction

3. **Class Separation Metrics:**
   - Silhouette scores for each domain
   - Inter-class vs intra-class distances

### Training Dashboard (`training_dashboard.py`)

1. **Real-time Monitoring:**
   - Live loss curves during training
   - System resource utilization
   - ETA calculations

2. **Comprehensive Analysis:**
   - Domain-specific loss tracking
   - Learning rate schedule visualization
   - Training efficiency metrics

3. **Automated Recommendations:**
   - Performance optimization suggestions
   - Resource usage warnings
   - Training convergence analysis

---

## 📊 Interpreting Results

### Good Signs to Look For:

1. **Dataset Analysis:**
   - Balanced class distributions
   - Clear constellation patterns
   - Appropriate SNR coverage

2. **Feature Visualization:**
   - Well-separated clusters in embeddings
   - High silhouette scores (>0.3)
   - Progressive improvement across epochs

3. **Training Dashboard:**
   - Consistently decreasing loss
   - Stable GPU memory usage (<90%)
   - Efficient data loading (data_time < 30% of batch_time)

### Warning Signs:

1. **Dataset Issues:**
   - Severely imbalanced classes
   - Missing SNR levels
   - Corrupted signal patterns

2. **Feature Problems:**
   - Overlapping clusters
   - Low silhouette scores (<0.1)
   - No improvement across epochs

3. **Training Issues:**
   - Plateaued loss curves
   - High GPU memory usage (>95%)
   - Slow data loading (data_time > 50% of batch_time)

---

## 💡 Tips for Best Results

1. **Start with dataset analysis** to understand your data quality
2. **Monitor training in real-time** to catch issues early
3. **Analyze features at multiple epochs** to track learning progress
4. **Compare different domains** to understand which features work best
5. **Use appropriate embedding methods** based on your analysis goals

---

## 🤝 Support

If you encounter issues:

1. Check the output directories for detailed logs
2. Verify your data paths and file formats
3. Ensure all dependencies are installed
4. Review the integration code for training dashboard
5. Check GPU memory availability for large analyses

The tools are designed to be robust and provide meaningful analysis for your MAC model development and evaluation.
