#!/usr/bin/env python3
"""
Overnight MAC Training Script
Trains MAC model on all 3 datasets sequentially while you sleep!

Usage: python overnight_training.py --start-immediately
"""

import os
import sys
import time
import json
import argparse
import subprocess
import traceback
from datetime import datetime
import torch
import shutil

class OvernightTrainer:
    def __init__(self, output_dir="overnight_training_results"):
        self.output_dir = output_dir
        self.start_time = time.time()
        self.datasets = ['RML201610A', 'RML201610B', 'RML2018']
        self.epochs_per_dataset = 200
        self.batch_size = 128
        
        # Training results tracking
        self.results = {
            'start_time': datetime.now().isoformat(),
            'datasets': {},
            'total_time': 0,
            'successful_datasets': 0,
            'failed_datasets': 0,
            'summary': {}
        }
        
        # Create output directory structure
        self.setup_output_directories()
        
        # Initialize main log
        self.main_log = open(f"{self.output_dir}/training_summary.log", "w")
        self.log("🚀 Overnight MAC Training Started!")
        self.log(f"Training Configuration:")
        self.log(f"  - Datasets: {self.datasets}")
        self.log(f"  - Epochs per dataset: {self.epochs_per_dataset}")
        self.log(f"  - Batch size: {self.batch_size}")
        self.log(f"  - Total estimated epochs: {len(self.datasets) * self.epochs_per_dataset}")
        
    def setup_output_directories(self):
        """Create organized directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        for dataset in self.datasets:
            dataset_dir = f"{self.output_dir}/{dataset}"
            os.makedirs(dataset_dir, exist_ok=True)
            os.makedirs(f"{dataset_dir}/model_checkpoints", exist_ok=True)
            os.makedirs(f"{dataset_dir}/training_dashboard", exist_ok=True)
            os.makedirs(f"{dataset_dir}/logs", exist_ok=True)
    
    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.main_log.write(log_message + "\n")
        self.main_log.flush()
    
    def estimate_completion_time(self, current_dataset_idx):
        """Estimate when training will complete"""
        if current_dataset_idx == 0:
            return "Calculating..."
        
        elapsed = time.time() - self.start_time
        avg_time_per_dataset = elapsed / current_dataset_idx
        remaining_datasets = len(self.datasets) - current_dataset_idx
        eta_seconds = avg_time_per_dataset * remaining_datasets
        
        eta_hours = eta_seconds / 3600
        completion_time = datetime.fromtimestamp(time.time() + eta_seconds)
        
        return f"{eta_hours:.1f}h (Complete by {completion_time.strftime('%H:%M:%S')})"
    
    def check_system_resources(self):
        """Check if system is ready for training"""
        self.log("🔍 Checking system resources...")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        self.log(f"  - CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            self.log(f"  - GPU Count: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                self.log(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Check disk space
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / 1e9
        self.log(f"  - Free Disk Space: {free_gb:.1f}GB")
        
        if free_gb < 5:
            self.log("⚠️  Warning: Low disk space!")
        
        return cuda_available
    
    def train_single_dataset(self, dataset_name, dataset_idx):
        """Train MAC model on a single dataset"""
        self.log(f"\n{'='*60}")
        self.log(f"🎯 STARTING TRAINING ON {dataset_name}")
        self.log(f"{'='*60}")
        self.log(f"Dataset {dataset_idx + 1}/{len(self.datasets)}")
        self.log(f"ETA: {self.estimate_completion_time(dataset_idx)}")
        
        dataset_start_time = time.time()
        dataset_dir = f"{self.output_dir}/{dataset_name}"
        
        # Initialize dataset results
        self.results['datasets'][dataset_name] = {
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'epochs_completed': 0,
            'best_loss': float('inf'),
            'training_time': 0,
            'error_message': None
        }
        
        try:
            # Prepare training command
            training_cmd = [
                'python', 'Pretraing_MAC.PY',
                '--ab_choose', dataset_name,
                '--epochs', str(self.epochs_per_dataset),
                '--batch_size', str(self.batch_size),
                '--model_path', f"{dataset_dir}/model_checkpoints/",
                '--tb_path', f"{dataset_dir}/logs/",
                '--save_freq', '40'  # Save every 40 epochs
            ]
            
            self.log(f"Training command: {' '.join(training_cmd)}")
            
            # Start training process
            log_file = f"{dataset_dir}/training_output.log"
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    training_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                self.log(f"Training process started (PID: {process.pid})")
                self.log(f"Logs: {log_file}")
                
                # Monitor training progress
                while process.poll() is None:
                    time.sleep(30)  # Check every 30 seconds
                    elapsed = time.time() - dataset_start_time
                    self.log(f"  Training {dataset_name}: {elapsed/60:.1f} minutes elapsed...")
                
                # Process completed
                return_code = process.returncode
                
            training_time = time.time() - dataset_start_time
            
            if return_code == 0:
                self.log(f"✅ {dataset_name} training completed successfully!")
                self.log(f"   Training time: {training_time/3600:.2f} hours")
                
                # Update results
                self.results['datasets'][dataset_name].update({
                    'status': 'completed',
                    'epochs_completed': self.epochs_per_dataset,
                    'training_time': training_time,
                    'end_time': datetime.now().isoformat()
                })
                
                # Move training dashboard files
                self.organize_training_outputs(dataset_name)
                
                self.results['successful_datasets'] += 1
                return True
                
            else:
                self.log(f"❌ {dataset_name} training failed with return code {return_code}")
                self.results['datasets'][dataset_name].update({
                    'status': 'failed',
                    'training_time': training_time,
                    'error_message': f"Process failed with return code {return_code}",
                    'end_time': datetime.now().isoformat()
                })
                self.results['failed_datasets'] += 1
                return False
                
        except Exception as e:
            self.log(f"❌ {dataset_name} training failed with exception: {str(e)}")
            self.log(f"   Error details: {traceback.format_exc()}")
            
            training_time = time.time() - dataset_start_time
            self.results['datasets'][dataset_name].update({
                'status': 'failed',
                'training_time': training_time,
                'error_message': str(e),
                'end_time': datetime.now().isoformat()
            })
            self.results['failed_datasets'] += 1
            return False
    
    def organize_training_outputs(self, dataset_name):
        """Organize training outputs into proper directories"""
        dataset_dir = f"{self.output_dir}/{dataset_name}"
        
        try:
            # Move training dashboard files
            dashboard_source = "training_dashboard"
            dashboard_dest = f"{dataset_dir}/training_dashboard"
            
            if os.path.exists(dashboard_source):
                # Copy dashboard files
                for file in os.listdir(dashboard_source):
                    src_file = os.path.join(dashboard_source, file)
                    dst_file = os.path.join(dashboard_dest, file)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
                
                self.log(f"   Dashboard files copied to {dashboard_dest}")
            
            # Move tensorboard logs if they exist
            tb_source_pattern = "2018pretrain_logs_"
            for item in os.listdir("."):
                if item.startswith(tb_source_pattern):
                    src_path = os.path.join(".", item)
                    dst_path = os.path.join(f"{dataset_dir}/logs", item)
                    if os.path.exists(src_path):
                        shutil.move(src_path, dst_path)
                        self.log(f"   Tensorboard logs moved to {dst_path}")
            
            self.log(f"   Training outputs organized for {dataset_name}")
            
        except Exception as e:
            self.log(f"   Warning: Error organizing outputs for {dataset_name}: {e}")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        total_time = time.time() - self.start_time
        self.results['total_time'] = total_time
        self.results['end_time'] = datetime.now().isoformat()
        
        self.log(f"\n{'='*60}")
        self.log(f"🏁 OVERNIGHT TRAINING COMPLETED!")
        self.log(f"{'='*60}")
        
        # Training summary
        self.log(f"📊 TRAINING SUMMARY:")
        self.log(f"   Total time: {total_time/3600:.2f} hours")
        self.log(f"   Successful datasets: {self.results['successful_datasets']}/{len(self.datasets)}")
        self.log(f"   Failed datasets: {self.results['failed_datasets']}/{len(self.datasets)}")
        
        # Per-dataset results
        self.log(f"\n📋 DETAILED RESULTS:")
        for dataset_name in self.datasets:
            if dataset_name in self.results['datasets']:
                dataset_result = self.results['datasets'][dataset_name]
                status_emoji = "✅" if dataset_result['status'] == 'completed' else "❌"
                
                self.log(f"   {status_emoji} {dataset_name}:")
                self.log(f"      Status: {dataset_result['status']}")
                self.log(f"      Training time: {dataset_result['training_time']/3600:.2f}h")
                
                if dataset_result['status'] == 'completed':
                    self.log(f"      Epochs completed: {dataset_result['epochs_completed']}")
                else:
                    self.log(f"      Error: {dataset_result.get('error_message', 'Unknown error')}")
        
        # System performance summary
        self.log(f"\n⚡ PERFORMANCE SUMMARY:")
        if self.results['successful_datasets'] > 0:
            avg_time_per_dataset = total_time / len(self.datasets)
            total_epochs = self.results['successful_datasets'] * self.epochs_per_dataset
            avg_time_per_epoch = total_time / total_epochs if total_epochs > 0 else 0
            
            self.log(f"   Average time per dataset: {avg_time_per_dataset/3600:.2f}h")
            self.log(f"   Total epochs trained: {total_epochs}")
            self.log(f"   Average time per epoch: {avg_time_per_epoch*60:.1f} minutes")
        
        # Generate final recommendations
        self.log(f"\n🎯 RECOMMENDATIONS:")
        if self.results['successful_datasets'] == len(self.datasets):
            self.log(f"   🎉 Perfect! All datasets trained successfully!")
            self.log(f"   ✨ Check individual dashboards for detailed analysis")
            self.log(f"   📊 Compare model performance across datasets")
        elif self.results['successful_datasets'] > 0:
            self.log(f"   ⚠️  Partial success: {self.results['successful_datasets']} datasets completed")
            self.log(f"   🔄 Consider retraining failed datasets with adjusted parameters")
        else:
            self.log(f"   💥 No datasets completed successfully")
            self.log(f"   🔍 Check error logs and system configuration")
        
        self.log(f"\n📁 Results saved to: {self.output_dir}")
        
        # Save results to JSON
        results_file = f"{self.output_dir}/training_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"📄 Full results JSON: {results_file}")
        
        # Generate wake-up summary
        self.generate_wake_up_summary()
    
    def generate_wake_up_summary(self):
        """Generate a concise wake-up summary"""
        summary_file = f"{self.output_dir}/WAKE_UP_SUMMARY.txt"
        
        with open(summary_file, 'w') as f:
            f.write("🌅 GOOD MORNING! YOUR OVERNIGHT TRAINING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            total_time = self.results['total_time']
            f.write(f"⏰ Training Duration: {total_time/3600:.1f} hours\n")
            f.write(f"✅ Successful: {self.results['successful_datasets']}/{len(self.datasets)} datasets\n")
            f.write(f"❌ Failed: {self.results['failed_datasets']}/{len(self.datasets)} datasets\n\n")
            
            f.write("📊 DATASET RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            for dataset_name in self.datasets:
                if dataset_name in self.results['datasets']:
                    result = self.results['datasets'][dataset_name]
                    status_emoji = "✅" if result['status'] == 'completed' else "❌"
                    f.write(f"{status_emoji} {dataset_name}: {result['status']}\n")
                    
                    if result['status'] == 'completed':
                        f.write(f"   └── {result['epochs_completed']} epochs in {result['training_time']/3600:.1f}h\n")
                    else:
                        f.write(f"   └── Error: {result.get('error_message', 'Unknown')}\n")
            
            f.write(f"\n📁 Check detailed results in: {self.output_dir}/\n")
            
            if self.results['successful_datasets'] == len(self.datasets):
                f.write("\n🎉 PERFECT NIGHT! All models trained successfully!\n")
                f.write("Time to analyze those beautiful loss curves! ☕\n")
            elif self.results['successful_datasets'] > 0:
                f.write(f"\n✨ Good progress! {self.results['successful_datasets']} models ready for analysis.\n")
            else:
                f.write("\n😴 Something went wrong. Check the logs! ☕\n")
        
        self.log(f"📋 Wake-up summary: {summary_file}")
    
    def run_overnight_training(self):
        """Main training loop"""
        try:
            # System check
            if not self.check_system_resources():
                self.log("⚠️  Warning: CUDA not available - training may be slow!")
            
            # Train each dataset sequentially
            for idx, dataset_name in enumerate(self.datasets):
                success = self.train_single_dataset(dataset_name, idx)
                
                if success:
                    self.log(f"✅ {dataset_name} completed successfully!")
                else:
                    self.log(f"❌ {dataset_name} failed - continuing with next dataset...")
                
                # Brief pause between datasets
                if idx < len(self.datasets) - 1:
                    self.log("⏸️  Brief pause before next dataset...")
                    time.sleep(10)
            
            # Generate final report
            self.generate_final_report()
            
        except KeyboardInterrupt:
            self.log("\n🛑 Training interrupted by user!")
            self.generate_final_report()
        except Exception as e:
            self.log(f"\n💥 Unexpected error: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
            self.generate_final_report()
        finally:
            self.main_log.close()


def main():
    parser = argparse.ArgumentParser(description='Overnight MAC Training Script')
    parser.add_argument('--start-immediately', action='store_true',
                       help='Start training immediately without confirmation')
    parser.add_argument('--output-dir', default='overnight_training_results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Epochs per dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    if not args.start_immediately:
        print("🚀 MAC Overnight Training Script")
        print("=" * 40)
        print("This will train MAC models on:")
        print("  - RML2016.10A (200 epochs)")
        print("  - RML2016.10B (200 epochs)")
        print("  - RML2018 (200 epochs)")
        print(f"\nEstimated time: 3-6 hours")
        print(f"Results will be saved to: {args.output_dir}")
        
        confirm = input("\nStart overnight training? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            print("Training cancelled.")
            return
    
    # Create trainer and start
    trainer = OvernightTrainer(output_dir=args.output_dir)
    trainer.epochs_per_dataset = args.epochs
    trainer.batch_size = args.batch_size
    
    print(f"\n🌙 Starting overnight training...")
    print(f"💤 Go to sleep! Check results in the morning.")
    print(f"📁 Results: {args.output_dir}/WAKE_UP_SUMMARY.txt")
    
    trainer.run_overnight_training()


if __name__ == "__main__":
    main()
