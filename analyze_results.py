#!/usr/bin/env python3
"""
MAC Comprehensive Experiments Results Analyzer

This script analyzes results from comprehensive MAC experiments,
generating statistical analysis and comparative visualizations.
"""

import os
import sys
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import glob
import re
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MACResultsAnalyzer:
    """Comprehensive analysis of MAC experiment results"""
    
    def __init__(self, config_path: str = "automation_config.yaml"):
        self.config = self.load_config(config_path)
        self.experiment_dir = Path(self.config['experiment_settings']['output_dir'])
        self.analysis_dir = self.experiment_dir / "cross_dataset_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.results_summary = {}
        self.training_curves = {}
        self.performance_metrics = {}
        self.best_models = {}
        
        print("🔍 MAC Results Analyzer initialized")
        print(f"📁 Experiment directory: {self.experiment_dir}")
        print(f"📊 Analysis output: {self.analysis_dir}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    
    def scan_experiment_results(self) -> Dict[str, Dict[str, Any]]:
        """Scan and collect all experiment results"""
        print("🔎 Scanning experiment results...")
        
        results = {}
        
        for dataset in ['RML201610A', 'RML201610B', 'RML2018']:
            dataset_dir = self.experiment_dir / dataset
            if not dataset_dir.exists():
                print(f"⚠️  Dataset directory not found: {dataset}")
                continue
            
            results[dataset] = {}
            
            # Scan SNR directories
            snr_dirs = list(dataset_dir.glob("snr_*"))
            print(f"📂 Found {len(snr_dirs)} SNR experiments for {dataset}")
            
            for snr_dir in snr_dirs:
                # Extract SNR value from directory name
                snr_match = re.search(r'snr_(-?\d+)', snr_dir.name)
                if not snr_match:
                    continue
                
                snr = int(snr_match.group(1))
                
                # Look for training logs and results
                log_dir = snr_dir / "logs"
                if not log_dir.exists():
                    continue
                
                # Extract results from training logs
                training_log = log_dir / "training_output.log"
                if training_log.exists():
                    experiment_results = self.parse_training_log(training_log, dataset, snr)
                    if experiment_results:
                        results[dataset][snr] = experiment_results
                        
                        # Look for best models
                        model_dir = snr_dir / "model_checkpoints"
                        if model_dir.exists():
                            best_models = list(model_dir.glob("best_model_epoch_*.pth"))
                            if best_models:
                                # Find the best model (lowest validation loss)
                                best_model = min(best_models, 
                                               key=lambda x: float(re.search(r'val_loss_([\d\.]+)', x.name).group(1)))
                                results[dataset][snr]['best_model_path'] = str(best_model)
        
        print(f"✅ Collected results from {sum(len(v) for v in results.values())} experiments")
        return results
    
    def parse_training_log(self, log_file: Path, dataset: str, snr: int) -> Optional[Dict[str, Any]]:
        """Parse training log to extract metrics"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract training curves
            train_losses = []
            val_losses = []
            epochs = []
            learning_rates = []
            best_val_loss = float('inf')
            total_epochs = 0
            
            # Parse epoch summaries
            epoch_pattern = r"--- Epoch (\d+) Summary ---.*?train_loss: ([\d\.]+).*?val_loss: ([\d\.]+).*?learning_rate: ([\d\.]+)"
            matches = re.findall(epoch_pattern, content, re.DOTALL)
            
            for match in matches:
                epoch, train_loss, val_loss, lr = match
                epochs.append(int(epoch))
                train_losses.append(float(train_loss))
                val_losses.append(float(val_loss))
                learning_rates.append(float(lr))
                
                if float(val_loss) < best_val_loss:
                    best_val_loss = float(val_loss)
                
                total_epochs = max(total_epochs, int(epoch))
            
            # Extract final performance metrics
            final_train_loss = train_losses[-1] if train_losses else None
            final_val_loss = val_losses[-1] if val_losses else None
            
            # Calculate convergence metrics
            convergence_epoch = None
            if len(val_losses) > 10:
                # Find when validation loss stabilized (moving average stops decreasing significantly)
                window_size = 10
                val_moving_avg = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
                
                for i in range(len(val_moving_avg) - window_size):
                    recent_avg = np.mean(val_moving_avg[i:i+window_size])
                    future_avg = np.mean(val_moving_avg[i+window_size:i+2*window_size]) if i+2*window_size < len(val_moving_avg) else recent_avg
                    
                    if abs(future_avg - recent_avg) / recent_avg < 0.01:  # Less than 1% change
                        convergence_epoch = i + window_size
                        break
            
            # Extract training time
            time_pattern = r"Training completed in ([\d\.]+) hours"
            time_match = re.search(time_pattern, content)
            training_time = float(time_match.group(1)) if time_match else None
            
            return {
                'dataset': dataset,
                'snr': snr,
                'epochs': epochs,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'best_val_loss': best_val_loss,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'total_epochs': total_epochs,
                'convergence_epoch': convergence_epoch,
                'training_time_hours': training_time,
                'completed': total_epochs > 0
            }
            
        except Exception as e:
            print(f"⚠️  Error parsing {log_file}: {e}")
            return None
    
    def analyze_snr_sensitivity(self, results: Dict[str, Dict[str, Any]]):
        """Analyze SNR sensitivity across datasets"""
        print("📈 Analyzing SNR sensitivity...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MAC Performance vs SNR Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Best validation loss vs SNR
        ax1 = axes[0, 0]
        for dataset in results:
            if results[dataset]:
                snrs = sorted(results[dataset].keys())
                best_losses = [results[dataset][snr]['best_val_loss'] for snr in snrs]
                ax1.plot(snrs, best_losses, marker='o', linewidth=2, label=dataset)
        
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Best Validation Loss')
        ax1.set_title('Validation Loss vs SNR')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training time vs SNR
        ax2 = axes[0, 1]
        for dataset in results:
            if results[dataset]:
                snrs = sorted(results[dataset].keys())
                times = [results[dataset][snr].get('training_time_hours', 0) for snr in snrs]
                ax2.plot(snrs, times, marker='s', linewidth=2, label=dataset)
        
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Training Time (hours)')
        ax2.set_title('Training Time vs SNR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convergence epoch vs SNR
        ax3 = axes[1, 0]
        for dataset in results:
            if results[dataset]:
                snrs = sorted(results[dataset].keys())
                conv_epochs = [results[dataset][snr].get('convergence_epoch', 0) for snr in snrs]
                # Filter out None values
                valid_data = [(s, c) for s, c in zip(snrs, conv_epochs) if c is not None and c > 0]
                if valid_data:
                    snrs_valid, conv_epochs_valid = zip(*valid_data)
                    ax3.plot(snrs_valid, conv_epochs_valid, marker='^', linewidth=2, label=dataset)
        
        ax3.set_xlabel('SNR (dB)')
        ax3.set_ylabel('Convergence Epoch')
        ax3.set_title('Convergence Speed vs SNR')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance improvement ratio
        ax4 = axes[1, 1]
        for dataset in results:
            if results[dataset]:
                snrs = sorted(results[dataset].keys())
                if len(snrs) > 1:
                    # Calculate relative improvement from lowest to highest SNR
                    losses = [results[dataset][snr]['best_val_loss'] for snr in snrs]
                    baseline_loss = max(losses)  # Worst performance
                    improvements = [(baseline_loss - loss) / baseline_loss * 100 for loss in losses]
                    ax4.plot(snrs, improvements, marker='d', linewidth=2, label=dataset)
        
        ax4.set_xlabel('SNR (dB)')
        ax4.set_ylabel('Performance Improvement (%)')
        ax4.set_title('Relative Performance Improvement')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'snr_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save numerical data
        snr_analysis = {}
        for dataset in results:
            if results[dataset]:
                snr_analysis[dataset] = {
                    'snr_range': [min(results[dataset].keys()), max(results[dataset].keys())],
                    'best_snr': min(results[dataset].keys(), key=lambda s: results[dataset][s]['best_val_loss']),
                    'worst_snr': max(results[dataset].keys(), key=lambda s: results[dataset][s]['best_val_loss']),
                    'performance_range': [
                        min(results[dataset][snr]['best_val_loss'] for snr in results[dataset]),
                        max(results[dataset][snr]['best_val_loss'] for snr in results[dataset])
                    ]
                }
        
        with open(self.analysis_dir / 'snr_sensitivity_data.json', 'w') as f:
            json.dump(snr_analysis, f, indent=2)
        
        print(f"✅ SNR sensitivity analysis saved to {self.analysis_dir}")
    
    def analyze_convergence_patterns(self, results: Dict[str, Dict[str, Any]]):
        """Analyze training convergence patterns"""
        print("📊 Analyzing convergence patterns...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Plot training curves for each dataset
        for idx, dataset in enumerate(['RML201610A', 'RML201610B', 'RML2018']):
            if dataset not in results or not results[dataset]:
                continue
            
            ax_train = axes[0, idx]
            ax_val = axes[1, idx]
            
            # Select representative SNR levels
            available_snrs = sorted(results[dataset].keys())
            if len(available_snrs) >= 3:
                selected_snrs = [available_snrs[0], available_snrs[len(available_snrs)//2], available_snrs[-1]]
            else:
                selected_snrs = available_snrs
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(selected_snrs)))
            
            for snr, color in zip(selected_snrs, colors):
                if snr in results[dataset]:
                    exp_data = results[dataset][snr]
                    epochs = exp_data.get('epochs', [])
                    train_losses = exp_data.get('train_losses', [])
                    val_losses = exp_data.get('val_losses', [])
                    
                    if epochs and train_losses:
                        ax_train.plot(epochs, train_losses, color=color, linewidth=2, 
                                    label=f'SNR {snr}dB', alpha=0.8)
                    
                    if epochs and val_losses:
                        ax_val.plot(epochs, val_losses, color=color, linewidth=2,
                                  label=f'SNR {snr}dB', alpha=0.8)
            
            ax_train.set_title(f'{dataset} - Training Loss')
            ax_train.set_xlabel('Epoch')
            ax_train.set_ylabel('Training Loss')
            ax_train.legend()
            ax_train.grid(True, alpha=0.3)
            
            ax_val.set_title(f'{dataset} - Validation Loss')
            ax_val.set_xlabel('Epoch')
            ax_val.set_ylabel('Validation Loss')
            ax_val.legend()
            ax_val.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Convergence analysis saved to {self.analysis_dir}")
    
    def generate_performance_summary(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Generate comprehensive performance summary table"""
        print("📋 Generating performance summary...")
        
        summary_data = []
        
        for dataset in results:
            for snr in results[dataset]:
                exp_data = results[dataset][snr]
                
                summary_data.append({
                    'Dataset': dataset,
                    'SNR': snr,
                    'Best_Val_Loss': exp_data.get('best_val_loss', np.nan),
                    'Final_Train_Loss': exp_data.get('final_train_loss', np.nan),
                    'Final_Val_Loss': exp_data.get('final_val_loss', np.nan),
                    'Total_Epochs': exp_data.get('total_epochs', 0),
                    'Convergence_Epoch': exp_data.get('convergence_epoch', np.nan),
                    'Training_Time_Hours': exp_data.get('training_time_hours', np.nan),
                    'Completed': exp_data.get('completed', False)
                })
        
        df = pd.DataFrame(summary_data)
        
        # Calculate additional metrics
        df['Training_Efficiency'] = df['Best_Val_Loss'] / df['Training_Time_Hours']  # Lower is better
        df['Convergence_Efficiency'] = df['Best_Val_Loss'] / df['Convergence_Epoch']  # Lower is better
        
        # Save summary table
        df.to_csv(self.analysis_dir / 'performance_summary.csv', index=False)
        
        # Generate summary statistics
        summary_stats = {}
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            summary_stats[dataset] = {
                'num_experiments': len(dataset_df),
                'completed_experiments': dataset_df['Completed'].sum(),
                'avg_best_val_loss': dataset_df['Best_Val_Loss'].mean(),
                'std_best_val_loss': dataset_df['Best_Val_Loss'].std(),
                'avg_training_time': dataset_df['Training_Time_Hours'].mean(),
                'total_training_time': dataset_df['Training_Time_Hours'].sum(),
                'best_snr_performance': {
                    'snr': dataset_df.loc[dataset_df['Best_Val_Loss'].idxmin(), 'SNR'],
                    'loss': dataset_df['Best_Val_Loss'].min()
                }
            }
        
        with open(self.analysis_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"✅ Performance summary saved to {self.analysis_dir}")
        return df
    
    def statistical_analysis(self, results: Dict[str, Dict[str, Any]]):
        """Perform statistical analysis of results"""
        print("🧮 Performing statistical analysis...")
        
        # Collect data for statistical tests
        dataset_performances = {}
        
        for dataset in results:
            if results[dataset]:
                performances = [results[dataset][snr]['best_val_loss'] for snr in results[dataset]]
                dataset_performances[dataset] = performances
        
        # Perform ANOVA test
        if len(dataset_performances) >= 2:
            datasets = list(dataset_performances.keys())
            performance_lists = [dataset_performances[d] for d in datasets]
            
            try:
                f_stat, p_value = stats.f_oneway(*performance_lists)
                
                statistical_results = {
                    'anova_test': {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'interpretation': 'Significant differences between datasets' if p_value < 0.05 else 'No significant differences'
                    }
                }
                
                # Post-hoc pairwise tests
                pairwise_tests = {}
                for i, dataset1 in enumerate(datasets):
                    for j, dataset2 in enumerate(datasets):
                        if i < j:
                            try:
                                t_stat, t_p_value = stats.ttest_ind(dataset_performances[dataset1], 
                                                                  dataset_performances[dataset2])
                                pairwise_tests[f"{dataset1}_vs_{dataset2}"] = {
                                    't_statistic': t_stat,
                                    'p_value': t_p_value,
                                    'significant': t_p_value < 0.05
                                }
                            except Exception as e:
                                print(f"⚠️  Error in t-test for {dataset1} vs {dataset2}: {e}")
                
                statistical_results['pairwise_tests'] = pairwise_tests
                
            except Exception as e:
                print(f"⚠️  Error in statistical analysis: {e}")
                statistical_results = {'error': str(e)}
        else:
            statistical_results = {'error': 'Insufficient data for statistical analysis'}
        
        # Save statistical results
        with open(self.analysis_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(statistical_results, f, indent=2, default=str)
        
        print(f"✅ Statistical analysis saved to {self.analysis_dir}")
    
    def identify_best_models(self, results: Dict[str, Dict[str, Any]]):
        """Identify and summarize best models"""
        print("🏆 Identifying best models...")
        
        best_models = {}
        
        # Overall best model
        all_results = []
        for dataset in results:
            for snr in results[dataset]:
                exp_data = results[dataset][snr]
                all_results.append({
                    'dataset': dataset,
                    'snr': snr,
                    'loss': exp_data['best_val_loss'],
                    'path': exp_data.get('best_model_path', '')
                })
        
        if all_results:
            overall_best = min(all_results, key=lambda x: x['loss'])
            best_models['overall_best'] = overall_best
            
            # Best model per dataset
            best_models['per_dataset'] = {}
            for dataset in results:
                if results[dataset]:
                    dataset_best = min(
                        [{'snr': snr, 'loss': results[dataset][snr]['best_val_loss'], 
                          'path': results[dataset][snr].get('best_model_path', '')} 
                         for snr in results[dataset]],
                        key=lambda x: x['loss']
                    )
                    best_models['per_dataset'][dataset] = dataset_best
            
            # Best model per SNR range
            snr_ranges = {
                'low_snr': (-20, -5),
                'medium_snr': (-4, 5),
                'high_snr': (6, 30)
            }
            
            best_models['per_snr_range'] = {}
            for range_name, (min_snr, max_snr) in snr_ranges.items():
                range_results = [r for r in all_results if min_snr <= r['snr'] <= max_snr]
                if range_results:
                    range_best = min(range_results, key=lambda x: x['loss'])
                    best_models['per_snr_range'][range_name] = range_best
        
        # Save best models summary
        with open(self.analysis_dir / 'best_models_summary.json', 'w') as f:
            json.dump(best_models, f, indent=2, default=str)
        
        print(f"✅ Best models summary saved to {self.analysis_dir}")
        return best_models
    
    def run_comprehensive_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting comprehensive MAC results analysis...")
        
        # Scan and collect results
        results = self.scan_experiment_results()
        
        if not any(results.values()):
            print("❌ No experiment results found. Please run experiments first.")
            return
        
        # Save raw results
        with open(self.analysis_dir / 'raw_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Run all analyses
        self.analyze_snr_sensitivity(results)
        self.analyze_convergence_patterns(results)
        df_summary = self.generate_performance_summary(results)
        self.statistical_analysis(results)
        best_models = self.identify_best_models(results)
        
        # Generate final summary report
        self.generate_analysis_report(results, df_summary, best_models)
        
        print("🎉 Comprehensive analysis completed!")
        print(f"📁 Results saved to: {self.analysis_dir}")
    
    def generate_analysis_report(self, results: Dict, df_summary: pd.DataFrame, best_models: Dict):
        """Generate comprehensive analysis report"""
        report_path = self.analysis_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("MAC COMPREHENSIVE EXPERIMENTS ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Experiment Overview
            f.write("EXPERIMENT OVERVIEW\n")
            f.write("-" * 30 + "\n")
            total_experiments = sum(len(v) for v in results.values())
            completed_experiments = df_summary['Completed'].sum()
            f.write(f"Total experiments: {total_experiments}\n")
            f.write(f"Completed experiments: {completed_experiments}\n")
            f.write(f"Success rate: {completed_experiments/total_experiments*100:.1f}%\n\n")
            
            # Dataset Summary
            f.write("DATASET SUMMARY\n")
            f.write("-" * 30 + "\n")
            for dataset in results:
                if results[dataset]:
                    dataset_df = df_summary[df_summary['Dataset'] == dataset]
                    f.write(f"\n{dataset}:\n")
                    f.write(f"  Experiments: {len(dataset_df)}\n")
                    f.write(f"  SNR range: {dataset_df['SNR'].min()} to {dataset_df['SNR'].max()} dB\n")
                    f.write(f"  Avg validation loss: {dataset_df['Best_Val_Loss'].mean():.4f} ± {dataset_df['Best_Val_Loss'].std():.4f}\n")
                    f.write(f"  Total training time: {dataset_df['Training_Time_Hours'].sum():.2f} hours\n")
            
            # Best Models
            f.write("\n\nBEST MODELS\n")
            f.write("-" * 30 + "\n")
            if 'overall_best' in best_models:
                best = best_models['overall_best']
                f.write(f"Overall best: {best['dataset']} SNR {best['snr']}dB (loss: {best['loss']:.4f})\n")
            
            if 'per_dataset' in best_models:
                f.write("\nBest per dataset:\n")
                for dataset, best in best_models['per_dataset'].items():
                    f.write(f"  {dataset}: SNR {best['snr']}dB (loss: {best['loss']:.4f})\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDations\n")
            f.write("-" * 30 + "\n")
            f.write("1. Model Selection:\n")
            if 'overall_best' in best_models:
                best = best_models['overall_best']
                f.write(f"   - Use {best['dataset']} model trained at SNR {best['snr']}dB for best performance\n")
            
            f.write("2. SNR Considerations:\n")
            f.write("   - Check SNR sensitivity plots for deployment SNR range\n")
            f.write("   - Consider ensemble of models for robust performance\n")
            
            f.write("3. Training Efficiency:\n")
            avg_time = df_summary['Training_Time_Hours'].mean()
            f.write(f"   - Average training time: {avg_time:.2f} hours per experiment\n")
            f.write("   - Consider early stopping based on convergence analysis\n")
        
        print(f"✅ Analysis report saved to {report_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze MAC experiment results")
    parser.add_argument("--config", default="automation_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--output-dir", 
                       help="Override output directory")
    
    args = parser.parse_args()
    
    analyzer = MACResultsAnalyzer(args.config)
    
    if args.output_dir:
        analyzer.experiment_dir = Path(args.output_dir)
        analyzer.analysis_dir = analyzer.experiment_dir / "cross_dataset_analysis"
        analyzer.analysis_dir.mkdir(exist_ok=True)
    
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()
