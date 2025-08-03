# MAC Comprehensive Experiments Automation System

A complete automation system for running comprehensive MAC (Multi-representation domain Attentive Contrastive learning) experiments across all datasets and SNR levels with detailed analysis and reporting.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 16GB RAM
- Sufficient disk space (~50GB for all experiments)

### Setup
```bash
# Install dependencies
pip install -r requirements_automation.txt

# Ensure datasets are preprocessed
python verify_corrected_splits.py

# Run all experiments (this will take ~26 hours)
python run_comprehensive_experiments.py
```

## 📋 System Overview

### **Total Experiments**: 66 training runs
- **RML201610A**: 20 SNR levels (-20dB to +18dB, 240 epochs each)
- **RML201610B**: 20 SNR levels (-20dB to +18dB, 240 epochs each)  
- **RML2018**: 26 SNR levels (-20dB to +30dB, 120 epochs each)

### **Estimated Time**: ~26 hours total
- RML201610A: ~8 hours
- RML201610B: ~8 hours
- RML2018: ~10 hours

## 🔧 Configuration

Edit `automation_config.yaml` to customize:
- Training parameters (epochs, batch size, learning rate)
- Dataset selection (enable/disable specific datasets)
- SNR levels to test
- Analysis options

## 🎯 Usage Examples

### Run All Experiments
```bash
python run_comprehensive_experiments.py
```

### Run Specific Datasets
```bash
python run_comprehensive_experiments.py --datasets RML201610A RML2018
```

### Resume Interrupted Experiments
```bash
python run_comprehensive_experiments.py --resume
```

### Custom Configuration
```bash
python run_comprehensive_experiments.py --config custom_config.yaml
```

## 📊 Analysis and Reporting

### Generate Analysis
```bash
python analyze_results.py
```

### Create Final Report
```bash
python generate_final_report.py
```

### Monitor Progress (Real-time)
```bash
python monitor_progress.py
```

## 📁 Output Structure

```
comprehensive_mac_experiments/
├── RML201610A/
│   ├── snr_-20/          # Individual SNR experiments
│   │   ├── logs/
│   │   ├── model_checkpoints/
│   │   └── visualizations/
│   └── ... (20 SNR folders)
├── RML201610B/
│   └── ... (20 SNR folders)
├── RML2018/
│   └── ... (26 SNR folders)
├── cross_dataset_analysis/
│   ├── snr_sensitivity_analysis.png
│   ├── convergence_analysis.png
│   ├── performance_summary.csv
│   └── best_models_summary.json
├── final_report/
│   ├── comprehensive_report.html  # 📄 Main interactive report
│   ├── executive_summary.txt
│   └── raw_analysis_data.json
└── automation_logs/
    ├── progress.json
    ├── error_log.txt
    └── timing_analysis.json
```

## 🔍 Key Features

### **Smart Automation**
- ✅ Sequential execution for GPU memory safety
- ✅ Automatic error recovery and retry
- ✅ Progress persistence across interruptions
- ✅ GPU memory cleanup between experiments

### **Comprehensive Analysis**
- ✅ SNR sensitivity analysis across datasets
- ✅ Training convergence patterns
- ✅ Statistical significance testing
- ✅ Best model identification
- ✅ Cross-dataset performance comparison

### **Rich Reporting**
- ✅ Interactive HTML dashboard
- ✅ Executive summary
- ✅ Performance visualizations
- ✅ Model recommendations

## 📈 Monitoring and Progress

### Real-time Progress Display
```
=== MAC COMPREHENSIVE EXPERIMENTS PROGRESS ===
Dataset: RML201610A [████████████████████] 20/20 SNRs (100%)
Dataset: RML201610B [██████████░░░░░░░░░░] 10/20 SNRs (50%)
Dataset: RML2018    [░░░░░░░░░░░░░░░░░░░░] 0/26 SNRs (0%)

Overall Progress: [████████░░░░░░░░░░░░] 30/66 experiments (45.5%)
Estimated Time Remaining: 12.3 hours
Current: Training RML201610B SNR=-2dB Epoch 45/240
```

### Progress Files
- `automation_logs/progress.json` - Detailed progress tracking
- `automation_logs/timing_analysis.json` - Performance metrics
- `automation_logs/error_log.txt` - Error logs and debugging

## 🏆 Results and Best Models

The system automatically identifies:
- **Overall best model** across all datasets/SNRs
- **Best model per dataset**
- **Best models by SNR range** (low/medium/high)
- **Performance trends** and recommendations

## ⚡ Performance Optimizations

### Memory Management
- Automatic GPU memory cleanup between experiments
- Progressive loading for large datasets
- Checkpoint compression to save disk space

### Error Handling
- Up to 3 automatic retries on failure
- Graceful degradation on OOM errors
- Detailed error logging with diagnostics

### Resume Capability
- Checkpoint-based resume from any point
- Progress persistence across system restarts
- Smart skipping of completed experiments

## 🛠️ Troubleshooting

### Common Issues

**Out of GPU Memory**
- Reduce batch size in config
- Ensure cleanup_between_runs is enabled
- Monitor GPU memory usage

**Slow Performance**
- Check system resources
- Reduce number of workers
- Use SSD storage for datasets

**Experiment Failures**
- Check error_log.txt for details
- Verify dataset preprocessing
- Ensure sufficient disk space

### Debug Mode
```bash
# Run with verbose logging
python run_comprehensive_experiments.py --verbose

# Check system requirements
python check_system_requirements.py

# Verify datasets
python verify_corrected_splits.py
```

## 📞 Support

For issues or questions:
1. Check `automation_logs/error_log.txt`
2. Review system requirements
3. Verify dataset preprocessing
4. Check available disk space and GPU memory

## 🎉 Next Steps After Completion

1. **Review Results**: Open `final_report/comprehensive_report.html`
2. **Analyze Performance**: Check cross-dataset comparisons
3. **Select Best Model**: Use recommendations from analysis
4. **Deploy Model**: Use best model for your application
5. **Fine-tune**: Consider fine-tuning on your specific data

---

**Happy Experimenting! 🚀**

*This automation system provides comprehensive MAC experiments with minimal manual intervention, delivering publication-ready results and analysis.*
