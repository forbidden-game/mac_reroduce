#!/usr/bin/env python3
"""
MAC Comprehensive Experiments Automation Script

This script automates training across all datasets and SNR levels,
providing comprehensive analysis and reporting.
"""

import os
import sys
import time
import yaml
import json
import logging
import argparse
import traceback
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import torch
import gc
import psutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

class ComprehensiveExperimentRunner:
    """Main class for running comprehensive MAC experiments"""
    
    def __init__(self, config_path: str = "automation_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.output_dir = Path(self.config['experiment_settings']['output_dir'])
        self.start_time = datetime.now()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize tracking
        self.progress_file = self.output_dir / "automation_logs" / "progress.json"
        self.error_log_file = self.output_dir / "automation_logs" / "error_log.txt"
        self.timing_file = self.output_dir / "automation_logs" / "timing_analysis.json"
        
        # Calculate total experiments FIRST (needed by load_progress)
        self.total_experiments = self.calculate_total_experiments()
        
        # Load or initialize progress
        self.progress = self.load_progress()
        self.timing_data = self.load_timing_data()
        
        self.logger.info(f"Initialized Comprehensive Experiment Runner")
        self.logger.info(f"Total experiments planned: {self.total_experiments}")
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "automation_logs").mkdir(exist_ok=True)
        
        # Setup logger
        log_file = self.output_dir / "automation_logs" / "automation.log"
        logging.basicConfig(
            level=getattr(logging, self.config['experiment_settings']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_total_experiments(self) -> int:
        """Calculate total number of experiments"""
        total = 0
        for dataset, config in self.config['datasets'].items():
            if config['enabled']:
                total += len(config['snr_levels'])
        return total
    
    def load_progress(self) -> Dict[str, Any]:
        """Load existing progress or initialize new"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                self.logger.info("Loaded existing progress data")
                return progress
            except Exception as e:
                self.logger.warning(f"Could not load progress: {e}")
        
        # Initialize new progress
        progress = {
            'completed_experiments': [],
            'failed_experiments': [],
            'current_experiment': None,
            'total_experiments': self.total_experiments,
            'start_time': self.start_time.isoformat(),
            'status': 'initialized'
        }
        return progress
    
    def load_timing_data(self) -> Dict[str, Any]:
        """Load timing analysis data"""
        if self.timing_file.exists():
            try:
                with open(self.timing_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load timing data: {e}")
        
        return {
            'experiments': {},
            'total_time': 0,
            'average_time_per_experiment': 0
        }
    
    def save_progress(self):
        """Save current progress"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save progress: {e}")
    
    def save_timing_data(self):
        """Save timing analysis"""
        try:
            with open(self.timing_file, 'w') as f:
                json.dump(self.timing_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save timing data: {e}")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory between experiments"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                # Log memory usage
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                self.logger.info(f"GPU Memory: {memory_used:.2f}GB / {memory_total:.2f}GB")
                
        except Exception as e:
            self.logger.warning(f"GPU cleanup warning: {e}")
    
    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'gpu_memory_percent': torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100 if torch.cuda.is_available() else 0
            }
        except Exception:
            return {'cpu_percent': 0, 'memory_percent': 0, 'gpu_memory_percent': 0}
    
    def create_experiment_directories(self, dataset: str, snr: int):
        """Create directory structure for experiment"""
        exp_dir = self.output_dir / dataset / f"snr_{snr}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        subdirs = ['logs', 'model_checkpoints', 'visualizations', 'training_dashboard']
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(exist_ok=True)
        
        return exp_dir
    
    def run_single_experiment(self, dataset: str, snr: int) -> bool:
        """Run a single training experiment"""
        exp_id = f"{dataset}_SNR_{snr}"
        self.logger.info(f"Starting experiment: {exp_id}")
        
        # Check if already completed
        if exp_id in self.progress['completed_experiments']:
            self.logger.info(f"Experiment {exp_id} already completed, skipping...")
            return True
        
        # Update progress
        self.progress['current_experiment'] = exp_id
        self.progress['status'] = 'running'
        self.save_progress()
        
        # Create directories
        exp_dir = self.create_experiment_directories(dataset, snr)
        
        # Get dataset config
        dataset_config = self.config['datasets'][dataset]
        training_config = self.config['training']
        
        # Cleanup before starting
        self.cleanup_gpu_memory()
        
        # Prepare command
        cmd = [
            'uv', 'run', 'python', 'Pretraing_MAC.PY',
            '--ab_choose', dataset,
            '--snr_tat', str(snr),
            '--epochs', str(dataset_config['epochs']),
            '--batch_size', str(dataset_config['batch_size']),
            '--learning_rate', str(training_config['learning_rate']),
            '--nce_k', str(training_config['nce_k']),
            '--nce_t', str(training_config['nce_t']),
            '--nce_m', str(training_config['nce_m']),
            '--view_chose', training_config['view_chose'],
            '--mod_l', training_config['mod_l'],
            '--feat_dim', str(training_config['feat_dim']),
            '--num_workers', str(training_config['num_workers']),
            '--print_freq', str(training_config['print_freq']),
            '--save_freq', str(training_config['save_freq']),
            '--n_1', str(training_config['n_1']),
            '--n_t', str(training_config['n_t']),
            '--momentum', str(training_config['momentum']),
            '--weight_decay', str(training_config['weight_decay'])
        ]
        
        # Add lr_decay_epochs
        lr_decay_str = ','.join(map(str, training_config['lr_decay_epochs']))
        cmd.extend(['--lr_decay_epochs', lr_decay_str])
        cmd.extend(['--lr_decay_rate', str(training_config['lr_decay_rate'])])
        
        try:
            exp_start_time = time.time()
            
            # Run experiment with output capture
            log_file = exp_dir / 'logs' / 'training_output.log'
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Real-time output logging
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    # Optionally print progress lines
                    if any(keyword in line for keyword in ['Epoch', 'val_loss', 'best_model']):
                        self.logger.info(f"[{exp_id}] {line.strip()}")
                
                process.wait()
                return_code = process.returncode
            
            exp_duration = time.time() - exp_start_time
            
            if return_code == 0:
                self.logger.info(f"✅ Experiment {exp_id} completed successfully in {exp_duration/3600:.2f} hours")
                
                # Record timing
                self.timing_data['experiments'][exp_id] = {
                    'duration': exp_duration,
                    'start_time': exp_start_time,
                    'dataset': dataset,
                    'snr': snr,
                    'epochs': dataset_config['epochs']
                }
                
                # Update progress
                self.progress['completed_experiments'].append(exp_id)
                self.progress['current_experiment'] = None
                self.save_progress()
                self.save_timing_data()
                
                return True
            else:
                raise subprocess.CalledProcessError(return_code, cmd)
                
        except Exception as e:
            self.logger.error(f"❌ Experiment {exp_id} failed: {str(e)}")
            
            # Log detailed error
            with open(self.error_log_file, 'a') as f:
                f.write(f"\n=== {exp_id} - {datetime.now()} ===\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
            
            # Update failed experiments
            if exp_id not in self.progress['failed_experiments']:
                self.progress['failed_experiments'].append(exp_id)
            
            self.progress['current_experiment'] = None
            self.save_progress()
            
            return False
    
    def calculate_eta(self) -> Optional[str]:
        """Calculate estimated time remaining"""
        completed = len(self.progress['completed_experiments'])
        if completed == 0:
            return None
        
        # Calculate average time per experiment
        total_time = sum(exp['duration'] for exp in self.timing_data['experiments'].values())
        avg_time = total_time / completed
        
        remaining_experiments = self.total_experiments - completed
        eta_seconds = remaining_experiments * avg_time
        
        return str(timedelta(seconds=int(eta_seconds)))
    
    def print_progress_summary(self):
        """Print comprehensive progress summary"""
        completed = len(self.progress['completed_experiments'])
        failed = len(self.progress['failed_experiments'])
        remaining = self.total_experiments - completed - failed
        
        progress_percent = (completed / self.total_experiments) * 100 if self.total_experiments > 0 else 0
        
        print("\n" + "="*80)
        print("🚀 MAC COMPREHENSIVE EXPERIMENTS PROGRESS")
        print("="*80)
        print(f"📊 Overall Progress: {completed}/{self.total_experiments} ({progress_percent:.1f}%)")
        print(f"✅ Completed: {completed}")
        print(f"❌ Failed: {failed}")
        print(f"⏳ Remaining: {remaining}")
        
        if self.progress['current_experiment']:
            print(f"🔄 Current: {self.progress['current_experiment']}")
        
        eta = self.calculate_eta()
        if eta:
            print(f"⏰ ETA: {eta}")
        
        # Dataset breakdown
        print(f"\n📋 Dataset Breakdown:")
        for dataset, config in self.config['datasets'].items():
            if config['enabled']:
                dataset_completed = sum(1 for exp in self.progress['completed_experiments'] 
                                      if exp.startswith(dataset))
                dataset_total = len(config['snr_levels'])
                dataset_percent = (dataset_completed / dataset_total) * 100 if dataset_total > 0 else 0
                
                progress_bar = "█" * int(dataset_percent // 5) + "░" * (20 - int(dataset_percent // 5))
                print(f"  {dataset}: [{progress_bar}] {dataset_completed}/{dataset_total} ({dataset_percent:.1f}%)")
        
        # System resources
        resources = self.get_system_resources()
        print(f"\n💻 System Resources:")
        print(f"  CPU: {resources['cpu_percent']:.1f}%")
        print(f"  Memory: {resources['memory_percent']:.1f}%")
        print(f"  GPU Memory: {resources['gpu_memory_percent']:.1f}%")
        
        print("="*80 + "\n")
    
    def run_all_experiments(self):
        """Run all configured experiments"""
        self.logger.info("🚀 Starting comprehensive MAC experiments")
        self.progress['status'] = 'running'
        self.save_progress()
        
        try:
            for dataset, config in self.config['datasets'].items():
                if not config['enabled']:
                    self.logger.info(f"Skipping disabled dataset: {dataset}")
                    continue
                
                self.logger.info(f"📂 Processing dataset: {dataset}")
                
                for snr in config['snr_levels']:
                    self.print_progress_summary()
                    
                    success = self.run_single_experiment(dataset, snr)
                    
                    if not success and self.config['automation']['resume_on_failure']:
                        self.logger.info(f"Retrying failed experiment: {dataset}_SNR_{snr}")
                        for retry in range(self.config['automation']['max_retries']):
                            self.logger.info(f"Retry {retry + 1}/{self.config['automation']['max_retries']}")
                            if self.run_single_experiment(dataset, snr):
                                break
                    
                    # Cleanup between experiments
                    if self.config['automation']['cleanup_between_runs']:
                        self.cleanup_gpu_memory()
                        time.sleep(2)  # Brief pause for system stability
            
            # Mark completion
            self.progress['status'] = 'completed'
            self.progress['end_time'] = datetime.now().isoformat()
            self.save_progress()
            
            self.logger.info("🎉 All experiments completed!")
            self.print_final_summary()
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ Experiments interrupted by user")
            self.progress['status'] = 'interrupted'
            self.save_progress()
            
        except Exception as e:
            self.logger.error(f"💥 Critical error in experiment runner: {e}")
            self.progress['status'] = 'error'
            self.save_progress()
            raise
    
    def print_final_summary(self):
        """Print final experiment summary"""
        completed = len(self.progress['completed_experiments'])
        failed = len(self.progress['failed_experiments'])
        total_time = time.time() - datetime.fromisoformat(self.progress['start_time']).timestamp()
        
        print("\n" + "="*80)
        print("🏁 FINAL EXPERIMENT SUMMARY")
        print("="*80)
        print(f"✅ Successfully completed: {completed}/{self.total_experiments}")
        print(f"❌ Failed experiments: {failed}")
        print(f"⏱️ Total execution time: {timedelta(seconds=int(total_time))}")
        print(f"📁 Results saved to: {self.output_dir}")
        
        if completed > 0:
            avg_time = total_time / completed
            print(f"📊 Average time per experiment: {timedelta(seconds=int(avg_time))}")
        
        print("\n🔍 Next steps:")
        print("  1. Run analysis: python analyze_results.py")
        print("  2. Generate report: python generate_final_report.py")
        print("  3. Check results in:", self.output_dir)
        print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run comprehensive MAC experiments")
    parser.add_argument("--config", default="automation_config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["RML201610A", "RML201610B", "RML2018"],
                       help="Run only specific datasets")
    
    args = parser.parse_args()
    
    # Create runner
    runner = ComprehensiveExperimentRunner(args.config)
    
    # Modify config for selective datasets
    if args.datasets:
        for dataset in runner.config['datasets']:
            runner.config['datasets'][dataset]['enabled'] = dataset in args.datasets
        runner.total_experiments = runner.calculate_total_experiments()
    
    # Run experiments
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
