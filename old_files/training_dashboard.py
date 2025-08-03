#!/usr/bin/env python3
"""
Training Monitoring Dashboard for MAC Models
Real-time training progress monitoring and visualization
"""

import os
import sys
import time
import json
import threading
import argparse
from datetime import datetime
import numpy as np

# FIX: Set matplotlib backend BEFORE importing pyplot to prevent segfaults
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, deque
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style with error handling
try:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except:
    # Fallback if seaborn style not available
    plt.style.use('default')
    
# Global lock for thread-safe plotting
import threading
_plot_lock = threading.Lock()

class TrainingMonitor:
    def __init__(self, log_dir="training_logs", dashboard_dir="dashboard_output"):
        self.log_dir = log_dir
        self.dashboard_dir = dashboard_dir
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Training metrics storage
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'l_loss': [],
            'ab_loss': [],
            'td_loss': [],
            'sd_loss': [],
            'learning_rate': [],
            'batch_time': [],
            'data_time': [],
            'gpu_memory': [],
            'gpu_utilization': [],
            'cpu_percent': [],
            'ram_percent': [],
            'timestamp': []
        }
        
        # Real-time metrics (for current epoch)
        self.current_epoch_metrics = defaultdict(list)
        
        # Moving averages
        self.moving_window = 100
        self.moving_averages = defaultdict(lambda: deque(maxlen=self.moving_window))
        
        # Training state
        self.training_start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.is_training = False
        
        # Auto-refresh settings (DISABLED to prevent segfaults)
        self.refresh_interval = 30  # seconds
        self.auto_refresh = False  # Disabled to prevent threading issues
        
    def start_monitoring(self, total_epochs):
        """Start monitoring training process"""
        self.training_start_time = time.time()
        self.total_epochs = total_epochs
        self.is_training = True
        self.current_epoch = 0
        
        print(f"Started training monitoring for {total_epochs} epochs")
        print(f"Dashboard output directory: {self.dashboard_dir}")
        
        # Start auto-refresh thread
        if self.auto_refresh:
            self.refresh_thread = threading.Thread(target=self._auto_refresh_plots)
            self.refresh_thread.daemon = True
            self.refresh_thread.start()
    
    def log_epoch_start(self, epoch):
        """Log the start of a new epoch"""
        self.current_epoch = epoch
        self.current_epoch_metrics = defaultdict(list)
        print(f"\n--- Epoch {epoch}/{self.total_epochs} Started ---")
    
    def log_batch_metrics(self, batch_idx, total_batches, metrics_dict):
        """Log metrics for current batch"""
        # Store current batch metrics
        for key, value in metrics_dict.items():
            self.current_epoch_metrics[key].append(value)
            self.moving_averages[key].append(value)
        
        # Calculate progress
        progress = (batch_idx + 1) / total_batches * 100
        
        # Print progress update
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            self._print_batch_progress(batch_idx + 1, total_batches, progress, metrics_dict)
    
    def log_epoch_end(self, epoch, epoch_metrics):
        """Log the end of an epoch and store epoch-level metrics"""
        # Calculate epoch averages
        epoch_avg_metrics = {}
        for key, values in self.current_epoch_metrics.items():
            if values:
                epoch_avg_metrics[key] = np.mean(values)
        
        # Add additional epoch metrics
        epoch_avg_metrics.update(epoch_metrics)
        
        # Store epoch metrics
        self.metrics['epoch'].append(epoch)
        self.metrics['timestamp'].append(datetime.now())
        
        for key, value in epoch_avg_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Log system metrics (with error handling)
        try:
            self._log_system_metrics()
        except Exception as e:
            print(f"Warning: Error logging system metrics: {e}")
        
        # Print epoch summary
        self._print_epoch_summary(epoch, epoch_avg_metrics)
        
        # Update plots (with thread safety and error handling)
        try:
            with _plot_lock:  # Thread-safe plotting
                self._update_plots()
        except Exception as e:
            print(f"Warning: Error updating plots: {e}")
        
        # Save metrics to file
        try:
            self._save_metrics()
        except Exception as e:
            print(f"Warning: Error saving metrics: {e}")
    
    def _log_system_metrics(self):
        """Log system resource usage"""
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.metrics['gpu_memory'].append(gpu.memoryUtil * 100)
                self.metrics['gpu_utilization'].append(gpu.load * 100)
            else:
                self.metrics['gpu_memory'].append(0)
                self.metrics['gpu_utilization'].append(0)
        except:
            self.metrics['gpu_memory'].append(0)
            self.metrics['gpu_utilization'].append(0)
        
        # CPU and RAM metrics
        self.metrics['cpu_percent'].append(psutil.cpu_percent())
        self.metrics['ram_percent'].append(psutil.virtual_memory().percent)
    
    def _print_batch_progress(self, batch_num, total_batches, progress, metrics):
        """Print formatted batch progress"""
        # Calculate ETA
        if self.training_start_time:
            elapsed = time.time() - self.training_start_time
            if progress > 0:
                eta_total = elapsed / (progress / 100)
                eta_remaining = eta_total - elapsed
                eta_str = f"{eta_remaining/3600:.1f}h" if eta_remaining > 3600 else f"{eta_remaining/60:.1f}m"
            else:
                eta_str = "N/A"
        else:
            eta_str = "N/A"
        
        # Format metrics string
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        
        print(f"Epoch {self.current_epoch} [{batch_num:4d}/{total_batches}] "
              f"({progress:5.1f}%) | {metrics_str} | ETA: {eta_str}")
    
    def _print_epoch_summary(self, epoch, metrics):
        """Print epoch summary"""
        print(f"\n--- Epoch {epoch} Summary ---")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.6f}")
        
        # Calculate time statistics
        if self.training_start_time:
            elapsed = time.time() - self.training_start_time
            avg_time_per_epoch = elapsed / epoch if epoch > 0 else 0
            remaining_epochs = self.total_epochs - epoch
            eta = avg_time_per_epoch * remaining_epochs
            
            print(f"Elapsed time: {elapsed/3600:.2f}h")
            print(f"Avg time per epoch: {avg_time_per_epoch/60:.1f}m")
            print(f"Estimated time remaining: {eta/3600:.2f}h")
        print("-" * 40)
    
    def _update_plots(self):
        """Update all monitoring plots"""
        if len(self.metrics['epoch']) < 2:
            return
        
        # Create comprehensive dashboard
        self._create_loss_plots()
        self._create_system_plots()
        self._create_training_progress_plot()
        self._create_learning_rate_plot()
        self._create_comprehensive_dashboard()
    
    def _create_loss_plots(self):
        """Create loss evolution plots"""
        if not self.metrics['epoch']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Main training loss
        ax = axes[0, 0]
        if self.metrics['train_loss']:
            ax.plot(self.metrics['epoch'], self.metrics['train_loss'], 'b-', linewidth=2, label='Training Loss')
            ax.set_title('Training Loss Evolution')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Domain-specific losses
        ax = axes[0, 1]
        loss_types = ['l_loss', 'ab_loss', 'td_loss', 'sd_loss']
        colors = ['red', 'blue', 'green', 'orange']
        
        for loss_type, color in zip(loss_types, colors):
            if self.metrics[loss_type]:
                ax.plot(self.metrics['epoch'], self.metrics[loss_type], 
                       color=color, linewidth=2, label=loss_type.replace('_', ' ').title())
        
        ax.set_title('Domain-specific Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Loss moving averages
        ax = axes[1, 0]
        if len(self.metrics['train_loss']) > 10:
            # Calculate moving average
            window = min(10, len(self.metrics['train_loss'])//2)
            if window > 1:
                ma_loss = pd.Series(self.metrics['train_loss']).rolling(window=window).mean()
                ax.plot(self.metrics['epoch'], ma_loss, 'r-', linewidth=2, label=f'MA({window})')
                ax.plot(self.metrics['epoch'], self.metrics['train_loss'], 'b-', alpha=0.3, label='Original')
                ax.set_title(f'Training Loss Moving Average (Window={window})')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Loss distribution (histogram)
        ax = axes[1, 1]
        if self.metrics['train_loss']:
            ax.hist(self.metrics['train_loss'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Training Loss Distribution')
            ax.set_xlabel('Loss Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dashboard_dir}/loss_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_system_plots(self):
        """Create system resource monitoring plots"""
        if not self.metrics['epoch']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # GPU Memory Usage
        ax = axes[0, 0]
        if self.metrics['gpu_memory']:
            ax.plot(self.metrics['epoch'], self.metrics['gpu_memory'], 'g-', linewidth=2)
            ax.set_title('GPU Memory Usage')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Memory Usage (%)')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='Warning (90%)')
            ax.legend()
        
        # GPU Utilization
        ax = axes[0, 1]
        if self.metrics['gpu_utilization']:
            ax.plot(self.metrics['epoch'], self.metrics['gpu_utilization'], 'b-', linewidth=2)
            ax.set_title('GPU Utilization')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Utilization (%)')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
        
        # CPU Usage
        ax = axes[1, 0]
        if self.metrics['cpu_percent']:
            ax.plot(self.metrics['epoch'], self.metrics['cpu_percent'], 'r-', linewidth=2)
            ax.set_title('CPU Usage')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('CPU Usage (%)')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
        
        # RAM Usage
        ax = axes[1, 1]
        if self.metrics['ram_percent']:
            ax.plot(self.metrics['epoch'], self.metrics['ram_percent'], 'orange', linewidth=2)
            ax.set_title('RAM Usage')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RAM Usage (%)')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='Warning (90%)')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.dashboard_dir}/system_resources.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_training_progress_plot(self):
        """Create training progress and timing plots"""
        if not self.metrics['epoch']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training Progress
        ax = axes[0, 0]
        progress = [e / self.total_epochs * 100 for e in self.metrics['epoch']]
        ax.plot(self.metrics['epoch'], progress, 'g-', linewidth=3)
        ax.set_title('Training Progress')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Progress (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add current progress annotation
        if progress:
            current_progress = progress[-1]
            ax.annotate(f'{current_progress:.1f}%', 
                       xy=(self.metrics['epoch'][-1], current_progress),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Batch timing
        ax = axes[0, 1]
        if self.metrics['batch_time']:
            ax.plot(self.metrics['epoch'], self.metrics['batch_time'], 'b-', linewidth=2, label='Batch Time')
        if self.metrics['data_time']:
            ax.plot(self.metrics['epoch'], self.metrics['data_time'], 'r-', linewidth=2, label='Data Loading Time')
        ax.set_title('Training Timing')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Learning Rate Schedule
        ax = axes[1, 0]
        if self.metrics['learning_rate']:
            ax.semilogy(self.metrics['epoch'], self.metrics['learning_rate'], 'purple', linewidth=2)
            ax.set_title('Learning Rate Schedule')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate (log scale)')
            ax.grid(True, alpha=0.3)
        
        # Training Speed (samples per second)
        ax = axes[1, 1]
        if self.metrics['batch_time']:
            # Estimate samples per second (assuming batch size of 64)
            batch_size = 64  # Default assumption
            samples_per_sec = [batch_size / bt if bt > 0 else 0 for bt in self.metrics['batch_time']]
            ax.plot(self.metrics['epoch'], samples_per_sec, 'green', linewidth=2)
            ax.set_title('Training Speed')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Samples/Second')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dashboard_dir}/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_learning_rate_plot(self):
        """Create detailed learning rate analysis"""
        if not self.metrics['learning_rate']:
            return
            
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['epoch'], self.metrics['learning_rate'], 'purple', linewidth=2)
        plt.title('Learning Rate Schedule (Linear)')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(self.metrics['epoch'], self.metrics['learning_rate'], 'purple', linewidth=2)
        plt.title('Learning Rate Schedule (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate (log scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dashboard_dir}/learning_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all metrics"""
        if len(self.metrics['epoch']) < 2:
            return
            
        fig = plt.figure(figsize=(20, 15))
        
        # Create a 3x3 grid of subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Training Loss
        ax1 = fig.add_subplot(gs[0, 0])
        if self.metrics['train_loss']:
            ax1.plot(self.metrics['epoch'], self.metrics['train_loss'], 'b-', linewidth=2)
            ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
        
        # Domain Losses
        ax2 = fig.add_subplot(gs[0, 1])
        loss_types = ['l_loss', 'ab_loss', 'td_loss', 'sd_loss']
        colors = ['red', 'blue', 'green', 'orange']
        for loss_type, color in zip(loss_types, colors):
            if self.metrics[loss_type]:
                ax2.plot(self.metrics['epoch'], self.metrics[loss_type], 
                        color=color, linewidth=2, label=loss_type.replace('_', ' ').title())
        ax2.set_title('Domain Losses', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # Learning Rate
        ax3 = fig.add_subplot(gs[0, 2])
        if self.metrics['learning_rate']:
            ax3.semilogy(self.metrics['epoch'], self.metrics['learning_rate'], 'purple', linewidth=2)
            ax3.set_title('Learning Rate', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('LR (log scale)')
            ax3.grid(True, alpha=0.3)
        
        # GPU Memory
        ax4 = fig.add_subplot(gs[1, 0])
        if self.metrics['gpu_memory']:
            ax4.plot(self.metrics['epoch'], self.metrics['gpu_memory'], 'g-', linewidth=2)
            ax4.set_title('GPU Memory', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Memory (%)')
            ax4.set_ylim(0, 100)
            ax4.grid(True, alpha=0.3)
        
        # GPU Utilization
        ax5 = fig.add_subplot(gs[1, 1])
        if self.metrics['gpu_utilization']:
            ax5.plot(self.metrics['epoch'], self.metrics['gpu_utilization'], 'b-', linewidth=2)
            ax5.set_title('GPU Utilization', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Utilization (%)')
            ax5.set_ylim(0, 100)
            ax5.grid(True, alpha=0.3)
        
        # Training Progress
        ax6 = fig.add_subplot(gs[1, 2])
        progress = [e / self.total_epochs * 100 for e in self.metrics['epoch']]
        ax6.plot(self.metrics['epoch'], progress, 'g-', linewidth=3)
        ax6.set_title('Training Progress', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Progress (%)')
        ax6.set_ylim(0, 100)
        ax6.grid(True, alpha=0.3)
        
        # Timing
        ax7 = fig.add_subplot(gs[2, 0])
        if self.metrics['batch_time']:
            ax7.plot(self.metrics['epoch'], self.metrics['batch_time'], 'b-', linewidth=2, label='Batch Time')
        if self.metrics['data_time']:
            ax7.plot(self.metrics['epoch'], self.metrics['data_time'], 'r-', linewidth=2, label='Data Time')
        ax7.set_title('Training Timing', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Time (s)')
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=8)
        
        # System Resources
        ax8 = fig.add_subplot(gs[2, 1])
        if self.metrics['cpu_percent']:
            ax8.plot(self.metrics['epoch'], self.metrics['cpu_percent'], 'r-', linewidth=2, label='CPU')
        if self.metrics['ram_percent']:
            ax8.plot(self.metrics['epoch'], self.metrics['ram_percent'], 'orange', linewidth=2, label='RAM')
        ax8.set_title('System Resources', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Usage (%)')
        ax8.set_ylim(0, 100)
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=8)
        
        # Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Create summary text
        summary_text = []
        if self.metrics['epoch']:
            current_epoch = self.metrics['epoch'][-1]
            summary_text.append(f"Current Epoch: {current_epoch}/{self.total_epochs}")
            
            if self.training_start_time:
                elapsed = time.time() - self.training_start_time
                summary_text.append(f"Elapsed: {elapsed/3600:.1f}h")
                
                if current_epoch > 0:
                    avg_time = elapsed / current_epoch
                    remaining = avg_time * (self.total_epochs - current_epoch)
                    summary_text.append(f"ETA: {remaining/3600:.1f}h")
            
            if self.metrics['train_loss']:
                current_loss = self.metrics['train_loss'][-1]
                best_loss = min(self.metrics['train_loss'])
                summary_text.append(f"Current Loss: {current_loss:.4f}")
                summary_text.append(f"Best Loss: {best_loss:.4f}")
            
            if self.metrics['learning_rate']:
                current_lr = self.metrics['learning_rate'][-1]
                summary_text.append(f"Current LR: {current_lr:.6f}")
        
        summary_str = '\n'.join(summary_text)
        ax9.text(0.1, 0.9, summary_str, transform=ax9.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax9.set_title('Training Summary', fontsize=12, fontweight='bold')
        
        # Add main title
        fig.suptitle('MAC Training Dashboard', fontsize=16, fontweight='bold')
        
        plt.savefig(f'{self.dashboard_dir}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        # Convert timestamps to strings for JSON serialization
        metrics_to_save = self.metrics.copy()
        metrics_to_save['timestamp'] = [t.isoformat() if hasattr(t, 'isoformat') else str(t) 
                                       for t in metrics_to_save['timestamp']]
        
        with open(f'{self.dashboard_dir}/training_metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
    
    def _auto_refresh_plots(self):
        """Auto-refresh plots in a separate thread"""
        while self.is_training:
            time.sleep(self.refresh_interval)
            if len(self.metrics['epoch']) > 1:
                try:
                    self._update_plots()
                except Exception as e:
                    print(f"Error updating plots: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring and generate final report"""
        self.is_training = False
        
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            print(f"\nTraining completed in {total_time/3600:.2f} hours")
        
        # Generate final plots
        self._update_plots()
        
        # Generate training report
        self._generate_training_report()
        
        print(f"Training dashboard saved to: {self.dashboard_dir}")
    
    def _generate_training_report(self):
        """Generate comprehensive training report"""
        if not self.metrics['epoch']:
            return
            
        report_file = f'{self.dashboard_dir}/training_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("MAC Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Training Overview
            f.write("TRAINING OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Epochs: {self.total_epochs}\n")
            f.write(f"Completed Epochs: {len(self.metrics['epoch'])}\n")
            
            if self.training_start_time:
                total_time = time.time() - self.training_start_time
                f.write(f"Total Training Time: {total_time/3600:.2f} hours\n")
                f.write(f"Average Time per Epoch: {total_time/len(self.metrics['epoch'])/60:.1f} minutes\n")
            
            f.write("\n")
            
            # Loss Analysis
            if self.metrics['train_loss']:
                f.write("LOSS ANALYSIS\n")
                f.write("-" * 15 + "\n")
                f.write(f"Initial Loss: {self.metrics['train_loss'][0]:.6f}\n")
                f.write(f"Final Loss: {self.metrics['train_loss'][-1]:.6f}\n")
                f.write(f"Best Loss: {min(self.metrics['train_loss']):.6f}\n")
                f.write(f"Loss Reduction: {(self.metrics['train_loss'][0] - self.metrics['train_loss'][-1]):.6f}\n")
                f.write(f"Loss Reduction %: {((self.metrics['train_loss'][0] - self.metrics['train_loss'][-1])/self.metrics['train_loss'][0]*100):.2f}%\n")
                f.write("\n")
            
            # System Performance
            if self.metrics['gpu_memory']:
                f.write("SYSTEM PERFORMANCE\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average GPU Memory: {np.mean(self.metrics['gpu_memory']):.1f}%\n")
                f.write(f"Peak GPU Memory: {max(self.metrics['gpu_memory']):.1f}%\n")
                f.write(f"Average GPU Utilization: {np.mean(self.metrics['gpu_utilization']):.1f}%\n")
                f.write(f"Average CPU Usage: {np.mean(self.metrics['cpu_percent']):.1f}%\n")
                f.write(f"Average RAM Usage: {np.mean(self.metrics['ram_percent']):.1f}%\n")
                f.write("\n")
            
            # Training Efficiency
            if self.metrics['batch_time']:
                f.write("TRAINING EFFICIENCY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Batch Time: {np.mean(self.metrics['batch_time']):.3f} seconds\n")
                f.write(f"Average Data Loading Time: {np.mean(self.metrics['data_time']):.3f} seconds\n")
                
                # Estimate throughput
                batch_size = 64  # Assumption
                avg_batch_time = np.mean(self.metrics['batch_time'])
                throughput = batch_size / avg_batch_time if avg_batch_time > 0 else 0
                f.write(f"Estimated Throughput: {throughput:.1f} samples/second\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if self.metrics['gpu_memory'] and max(self.metrics['gpu_memory']) > 90:
                f.write("⚠️  High GPU memory usage detected. Consider reducing batch size.\n")
            
            if self.metrics['data_time'] and self.metrics['batch_time']:
                avg_data_time = np.mean(self.metrics['data_time'])
                avg_batch_time = np.mean(self.metrics['batch_time'])
                data_ratio = avg_data_time / avg_batch_time if avg_batch_time > 0 else 0
                
                if data_ratio > 0.3:
                    f.write("⚠️  Data loading is taking significant time. Consider increasing num_workers.\n")
                
                if avg_batch_time > 1.0:
                    f.write("⚠️  Slow batch processing. Consider optimizing model or reducing complexity.\n")
            
            if self.metrics['learning_rate']:
                if self.metrics['learning_rate'][-1] < 1e-6:
                    f.write("⚠️  Learning rate is very small. Training may have converged or need adjustment.\n")
            
            if self.metrics['train_loss'] and len(self.metrics['train_loss']) > 10:
                recent_losses = self.metrics['train_loss'][-10:]
                loss_variance = np.var(recent_losses)
                if loss_variance < 1e-8:
                    f.write("⚠️  Loss has plateaued. Consider adjusting learning rate or other hyperparameters.\n")
            
            f.write("\n")
            f.write("Report generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")


class EnhancedTrainingMonitor(TrainingMonitor):
    """Enhanced training monitor with integration for MAC training script"""
    
    def __init__(self, log_dir="training_logs", dashboard_dir="dashboard_output"):
        super().__init__(log_dir, dashboard_dir)
        
    def integrate_with_training_script(self):
        """
        Instructions for integrating with the main training script.
        This method provides code snippets and guidance.
        """
        integration_code = '''
# Integration with Pretraing_MAC.PY
# Add this at the beginning of your training script:

from training_dashboard import EnhancedTrainingMonitor

# Initialize monitor
monitor = EnhancedTrainingMonitor(dashboard_dir="training_dashboard")

# At the start of training (in main function):
monitor.start_monitoring(args.epochs)

# At the start of each epoch (before training loop):
monitor.log_epoch_start(epoch)

# In the training loop (for each batch):
batch_metrics = {
    'train_loss': loss.item(),
    'l_loss': l_loss.item(),
    'ab_loss': ab_loss.item(),
    'td_loss': l_prob,  # or appropriate TD loss
    'sd_loss': ab_prob, # or appropriate SD loss
    'batch_time': batch_time.val,
    'data_time': data_time.val
}
monitor.log_batch_metrics(idx, len(train_loader), batch_metrics)

# At the end of each epoch:
epoch_metrics = {
    'learning_rate': optimizer.param_groups[0]['lr']
}
monitor.log_epoch_end(epoch, epoch_metrics)

# At the end of training:
monitor.stop_monitoring()
        '''
        
        print("INTEGRATION INSTRUCTIONS")
        print("=" * 50)
        print(integration_code)
        
        # Save integration instructions to file
        with open(f'{self.dashboard_dir}/integration_instructions.txt', 'w') as f:
            f.write("MAC Training Dashboard Integration\n")
            f.write("=" * 40 + "\n\n")
            f.write(integration_code)
        
        return integration_code


def demo_training_monitor():
    """Demo function showing how to use the training monitor"""
    print("Running Training Monitor Demo...")
    
    # Create monitor
    monitor = EnhancedTrainingMonitor(dashboard_dir="demo_dashboard")
    
    # Simulate training
    total_epochs = 10
    batches_per_epoch = 50
    
    monitor.start_monitoring(total_epochs)
    
    # Simulate training epochs
    for epoch in range(1, total_epochs + 1):
        monitor.log_epoch_start(epoch)
        
        # Simulate training batches
        for batch_idx in range(batches_per_epoch):
            # Simulate decreasing loss with some noise
            base_loss = 2.0 * np.exp(-epoch * 0.1) + 0.1
            noise = np.random.normal(0, 0.05)
            
            batch_metrics = {
                'train_loss': base_loss + noise,
                'l_loss': base_loss * 0.3 + noise * 0.1,
                'ab_loss': base_loss * 0.3 + noise * 0.1,
                'td_loss': base_loss * 0.2 + noise * 0.1,
                'sd_loss': base_loss * 0.2 + noise * 0.1,
                'batch_time': 0.5 + np.random.normal(0, 0.1),
                'data_time': 0.1 + np.random.normal(0, 0.02)
            }
            
            monitor.log_batch_metrics(batch_idx, batches_per_epoch, batch_metrics)
            
            # Small delay to simulate training
            time.sleep(0.01)
        
        # End epoch
        epoch_metrics = {
            'learning_rate': 0.01 * (0.5 ** (epoch // 3))  # Simulate LR decay
        }
        monitor.log_epoch_end(epoch, epoch_metrics)
        
        # Simulate epoch processing time
        time.sleep(0.1)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("Demo completed! Check 'demo_dashboard/' for results.")


def analyze_existing_logs(log_dir):
    """Analyze existing tensorboard logs and generate dashboard"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        print(f"Analyzing logs from: {log_dir}")
        
        # Find event files
        event_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if 'events.out.tfevents' in file:
                    event_files.append(os.path.join(root, file))
        
        if not event_files:
            print("No tensorboard event files found!")
            return
        
        # Create monitor
        monitor = TrainingMonitor(dashboard_dir=f"{log_dir}_analysis")
        
        # Process each event file
        for event_file in event_files:
            print(f"Processing: {event_file}")
            
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            # Extract scalar data
            scalar_tags = ea.Tags()['scalars']
            
            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                
                for event in scalar_events:
                    epoch = event.step
                    value = event.value
                    
                    # Map tensorboard tags to monitor metrics
                    if 'loss' in tag.lower():
                        if 'l_loss' in tag:
                            if len(monitor.metrics['l_loss']) <= epoch:
                                monitor.metrics['l_loss'].extend([0] * (epoch + 1 - len(monitor.metrics['l_loss'])))
                            monitor.metrics['l_loss'][epoch] = value
                        elif 'ab_loss' in tag:
                            if len(monitor.metrics['ab_loss']) <= epoch:
                                monitor.metrics['ab_loss'].extend([0] * (epoch + 1 - len(monitor.metrics['ab_loss'])))
                            monitor.metrics['ab_loss'][epoch] = value
                        else:
                            if len(monitor.metrics['train_loss']) <= epoch:
                                monitor.metrics['train_loss'].extend([0] * (epoch + 1 - len(monitor.metrics['train_loss'])))
                            monitor.metrics['train_loss'][epoch] = value
                    elif 'learning_rate' in tag.lower():
                        if len(monitor.metrics['learning_rate']) <= epoch:
                            monitor.metrics['learning_rate'].extend([0] * (epoch + 1 - len(monitor.metrics['learning_rate'])))
                        monitor.metrics['learning_rate'][epoch] = value
        
        # Update epoch list
        max_epoch = max(len(monitor.metrics['train_loss']), 
                       len(monitor.metrics['l_loss']),
                       len(monitor.metrics['ab_loss']),
                       len(monitor.metrics['learning_rate']))
        
        monitor.metrics['epoch'] = list(range(1, max_epoch + 1))
        monitor.metrics['timestamp'] = [datetime.now() for _ in range(max_epoch)]
        
        # Generate plots
        monitor.total_epochs = max_epoch
        monitor._update_plots()
        monitor._generate_training_report()
        
        print(f"Analysis complete! Dashboard saved to: {monitor.dashboard_dir}")
        
    except ImportError:
        print("tensorboard package required for log analysis. Install with: pip install tensorboard")
    except Exception as e:
        print(f"Error analyzing logs: {e}")


def main():
    parser = argparse.ArgumentParser(description='MAC Training Dashboard')
    parser.add_argument('--mode', choices=['demo', 'analyze', 'integrate'], 
                       default='demo',
                       help='Mode: demo (run demo), analyze (analyze existing logs), integrate (show integration)')
    parser.add_argument('--log-dir', 
                       help='Log directory to analyze (for analyze mode)')
    parser.add_argument('--dashboard-dir', default='dashboard_output',
                       help='Output directory for dashboard')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_training_monitor()
    elif args.mode == 'analyze':
        if not args.log_dir:
            print("--log-dir required for analyze mode")
            return
        analyze_existing_logs(args.log_dir)
    elif args.mode == 'integrate':
        monitor = EnhancedTrainingMonitor(dashboard_dir=args.dashboard_dir)
        monitor.integrate_with_training_script()


if __name__ == "__main__":
    main()
