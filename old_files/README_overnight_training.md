# 🌙 Overnight MAC Training

Keep your GPU busy while you sleep! This script trains MAC models on all 3 datasets sequentially.

## 🚀 Quick Start

### Option 1: Interactive Launcher
```bash
./start_overnight_training.sh
```

### Option 2: Direct Launch
```bash
python overnight_training.py --start-immediately
```

## 📊 What It Does

- **Sequential Training**: RML2016.10A → RML2016.10B → RML2018
- **200 epochs per dataset** (600 total epochs)
- **Batch size 128** for optimal GPU utilization
- **Automatic organization** of results and logs
- **Comprehensive monitoring** with dashboards

## 📁 Output Structure

```
overnight_training_results/
├── WAKE_UP_SUMMARY.txt          # 🌅 Your morning coffee summary!
├── training_summary.log          # Complete training log
├── training_results.json         # Machine-readable results
├── RML2016.10A/
│   ├── model_checkpoints/         # Best model saved here
│   ├── training_dashboard/        # Beautiful plots & analysis
│   ├── logs/                      # Tensorboard logs
│   └── training_output.log        # Raw training output
├── RML2016.10B/
│   └── [same structure]
└── RML2018/
    └── [same structure]
```

## ⏰ Timing

- **Per dataset**: ~1-2 hours (200 epochs)
- **Total time**: ~3-6 hours
- **Perfect for overnight!** 😴

## 🎯 Features

### 🛡️ **Error Resilience**
- If one dataset fails, continues with the next
- Detailed error logging and recovery

### 📊 **Comprehensive Monitoring**
- Real-time progress updates
- GPU/CPU/RAM monitoring
- Training dashboards with beautiful plots
- ETA calculations

### 🌅 **Morning Report**
The script generates `WAKE_UP_SUMMARY.txt` with:
- Training duration and success rate
- Per-dataset results
- Quick status overview
- Next steps recommendations

## 📋 Command Options

```bash
python overnight_training.py [options]

Options:
  --start-immediately    Skip confirmation prompt
  --output-dir DIR      Custom output directory (default: overnight_training_results)
  --epochs N            Epochs per dataset (default: 200)
  --batch-size N        Batch size (default: 128)
```

## 🔍 Monitoring Progress

### Check Status While Running
```bash
tail -f overnight_training_results/training_summary.log
```

### Check GPU Usage
```bash
nvidia-smi
```

### Check Individual Dataset Progress
```bash
tail -f overnight_training_results/RML2016.10A/training_output.log
```

## 🎉 Success Examples

### Perfect Night (All datasets trained)
```
🌅 GOOD MORNING! YOUR OVERNIGHT TRAINING RESULTS
==================================================

⏰ Training Duration: 4.2 hours
✅ Successful: 3/3 datasets
❌ Failed: 0/3 datasets

📊 DATASET RESULTS:
--------------------
✅ RML2016.10A: completed
   └── 200 epochs in 1.3h
✅ RML2016.10B: completed
   └── 200 epochs in 1.4h
✅ RML2018: completed
   └── 200 epochs in 1.5h

🎉 PERFECT NIGHT! All models trained successfully!
Time to analyze those beautiful loss curves! ☕
```

## 🛠️ Troubleshooting

### GPU Not Being Used
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify GPU drivers are installed
- Check if other processes are using GPU: `nvidia-smi`

### Dataset Not Found
- Ensure datasets are in `data/` directory
- Run preprocessing scripts first
- Check data paths in `util.py`

### Out of Memory
- Reduce batch size: `--batch-size 64`
- Close other GPU-intensive applications

### Training Interrupted
- Results are saved after each dataset
- Restart training manually on failed datasets
- Check logs for specific error details

## 🎯 Pro Tips

1. **Before Starting**:
   ```bash
   # Check everything is ready
   ./start_overnight_training.sh
   ```

2. **Remote Monitoring** (if using SSH):
   ```bash
   # Use screen or tmux to keep training running
   screen -S overnight_training
   python overnight_training.py --start-immediately
   # Ctrl+A, D to detach
   
   # Reconnect later with:
   screen -r overnight_training
   ```

3. **Free Up Memory** before starting:
   ```bash
   # Clear any existing training processes
   pkill -f "python.*Pretraing_MAC"
   
   # Clear GPU memory
   python -c "import torch; torch.cuda.empty_cache()"
   ```

## 🔥 Expected Results

After a successful overnight run, you'll have:
- **3 trained MAC models** ready for fine-tuning
- **Comprehensive training dashboards** with loss curves, system monitoring
- **Detailed analysis** of training efficiency and performance
- **Baseline models** for your research experiments

Sweet dreams and happy training! 😴🚀
