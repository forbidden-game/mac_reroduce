#!/usr/bin/env python3
"""
Dataset Analysis & Visualization Script for RadioML Datasets
Provides comprehensive analysis of RML2016.10a, RML2016.10b, and RML2018 datasets
"""

import os
import sys
import pickle
import numpy as np

# Fix matplotlib backend to prevent segfaults
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import signal
from sklearn.metrics import confusion_matrix
import h5py
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

# Set matplotlib parameters for memory efficiency
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['agg.path.chunksize'] = 10000

class DatasetAnalyzer:
    def __init__(self, output_dir="analysis_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Modulation class names
        self.rml2016a_classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        self.rml2016b_classes = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        self.rml2018_classes = [
            'OOK', 'BPSK', 'QPSK', '8PSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 
            'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK',
            'BFSK', 'CPFSK', 'PAM4', 'QAM16', 'QAM64', 'GFSK', '8PSK', 'AM-DSB'
        ]
        
    def load_pickle_data(self, file_path):
        """Load pickle file safely"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f, encoding='iso-8859-1')
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def analyze_rml2016a(self):
        """Analyze RML2016.10a dataset"""
        print("=" * 60)
        print("ANALYZING RML2016.10a DATASET")
        print("=" * 60)
        
        # Try to load raw data first
        raw_file = 'data/RML2016.10a_dict.pkl'
        if not os.path.exists(raw_file):
            print(f"Raw data file {raw_file} not found!")
            print("Trying to analyze from processed data...")
            
            # Try to load from processed data like RML2016.10b
            processed_dir = 'data/processed/RML2016.10a/'
            if os.path.exists(processed_dir):
                return self._analyze_processed_rml2016a(processed_dir)
            else:
                print(f"Processed data directory {processed_dir} also not found!")
                print("Skipping RML2016.10a analysis.")
                return None
            
        Xd = self.load_pickle_data(raw_file)
        if Xd is None:
            return None
            
        # Get modulation types and SNR values
        mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]
        
        # Collect statistics
        stats = {
            'total_samples': 0,
            'num_classes': len(mods),
            'num_snr_levels': len(snrs),
            'snr_range': (min(snrs), max(snrs)),
            'modulations': mods,
            'snr_levels': snrs,
            'samples_per_mod': defaultdict(int),
            'samples_per_snr': defaultdict(int),
            'samples_per_mod_snr': {}
        }
        
        # Analyze data distribution
        for mod in mods:
            for snr in snrs:
                if (mod, snr) in Xd:
                    n_samples = Xd[(mod, snr)].shape[0]
                    stats['total_samples'] += n_samples
                    stats['samples_per_mod'][mod] += n_samples
                    stats['samples_per_snr'][snr] += n_samples
                    stats['samples_per_mod_snr'][(mod, snr)] = n_samples
        
        # Print statistics
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Number of modulation classes: {stats['num_classes']}")
        print(f"Number of SNR levels: {stats['num_snr_levels']}")
        print(f"SNR range: {stats['snr_range'][0]} to {stats['snr_range'][1]} dB")
        print(f"Modulations: {mods}")
        print(f"SNR levels: {snrs}")
        
        # Create visualizations
        self._plot_class_distribution(stats, 'RML2016.10a', self.rml2016a_classes)
        self._plot_snr_distribution(stats, 'RML2016.10a')
        self._plot_heatmap(stats, 'RML2016.10a', mods, snrs)
        
        # Analyze signal characteristics
        self._analyze_signal_characteristics(Xd, mods, snrs, 'RML2016.10a', self.rml2016a_classes)
        
        return stats
    
    def _analyze_processed_rml2016a(self, processed_dir):
        """Analyze RML2016.10a from processed data directory"""
        print(f"Analyzing processed data from: {processed_dir}")
        
        # Check for processed data files
        sample_files = [f for f in os.listdir(processed_dir) if 'train_dataset' in f]
        if not sample_files:
            print("No processed dataset files found!")
            return None
            
        stats = {
            'total_samples': 0,
            'num_classes': 11,  # RML2016.10a has 11 classes
            'num_snr_levels': 0,
            'modulations': self.rml2016a_classes,
            'samples_per_snr': defaultdict(int),
            'files_found': []
        }
        
        # Analyze processed files (similar to RML2016.10b structure)
        for filename in os.listdir(processed_dir):
            if 'train_dataset' in filename:
                try:
                    snr = int(filename.split('_')[0])
                    file_path = os.path.join(processed_dir, filename)
                    dataset = self.load_pickle_data(file_path)
                    if dataset:
                        n_samples = len(dataset)
                        stats['total_samples'] += n_samples
                        stats['samples_per_snr'][snr] += n_samples
                        stats['files_found'].append(filename)
                except:
                    continue
        
        stats['num_snr_levels'] = len(set([int(f.split('_')[0]) for f in stats['files_found'] if f.split('_')[0].isdigit()]))
        
        print(f"Total training samples: {stats['total_samples']:,}")
        print(f"Number of modulation classes: {stats['num_classes']}")
        print(f"Number of SNR levels: {stats['num_snr_levels']}")
        print(f"Files found: {len(stats['files_found'])}")
        
        # Create basic visualizations
        if stats['total_samples'] > 0:
            self._plot_snr_distribution(stats, 'RML2016.10a')
        
        return stats
    
    def analyze_rml2016b(self):
        """Analyze RML2016.10b dataset"""
        print("=" * 60)
        print("ANALYZING RML2016.10b DATASET")
        print("=" * 60)
        
        # Check for processed data
        processed_dir = 'data/processed/RML2016.10b/'
        if not os.path.exists(processed_dir):
            print(f"Processed data directory {processed_dir} not found!")
            return None
            
        # Load one file to get structure
        sample_files = [f for f in os.listdir(processed_dir) if 'train_dataset' in f]
        if not sample_files:
            print("No processed dataset files found!")
            return None
            
        stats = {
            'total_samples': 0,
            'num_classes': 10,
            'num_snr_levels': 0,
            'modulations': self.rml2016b_classes,
            'samples_per_snr': defaultdict(int),
            'files_found': []
        }
        
        # Analyze processed files
        for filename in os.listdir(processed_dir):
            if 'train_dataset' in filename:
                snr = filename.split('_')[0]
                file_path = os.path.join(processed_dir, filename)
                dataset = self.load_pickle_data(file_path)
                if dataset:
                    n_samples = len(dataset)
                    stats['total_samples'] += n_samples
                    stats['samples_per_snr'][int(snr)] += n_samples
                    stats['files_found'].append(filename)
                    stats['num_snr_levels'] += 1
        
        stats['num_snr_levels'] = len(set([int(f.split('_')[0]) for f in stats['files_found']]))
        
        print(f"Total training samples: {stats['total_samples']:,}")
        print(f"Number of modulation classes: {stats['num_classes']}")
        print(f"Number of SNR levels: {stats['num_snr_levels']}")
        print(f"Files found: {len(stats['files_found'])}")
        
        return stats
    
    def analyze_rml2018(self):
        """Analyze RML2018 dataset"""
        print("=" * 60)
        print("ANALYZING RML2018 DATASET")
        print("=" * 60)
        
        # Check for processed data
        processed_dir = 'data/processed/RML2018/'
        if not os.path.exists(processed_dir):
            print(f"Processed data directory {processed_dir} not found!")
            return None
            
        sample_files = [f for f in os.listdir(processed_dir) if 'train_dataset' in f]
        if not sample_files:
            print("No processed dataset files found!")
            return None
            
        stats = {
            'total_samples': 0,
            'num_classes': 24,
            'num_snr_levels': 0,
            'samples_per_snr': defaultdict(int),
            'files_found': []
        }
        
        # Analyze processed files
        for filename in os.listdir(processed_dir):
            if 'train_dataset' in filename:
                # Extract SNR from filename like "RML2018_MV4_snr_6_train_dataset"
                parts = filename.split('_')
                snr_idx = parts.index('snr') + 1
                snr = int(parts[snr_idx])
                
                file_path = os.path.join(processed_dir, filename)
                dataset = self.load_pickle_data(file_path)
                if dataset:
                    n_samples = len(dataset)
                    stats['total_samples'] += n_samples
                    stats['samples_per_snr'][snr] += n_samples
                    stats['files_found'].append(filename)
        
        stats['num_snr_levels'] = len(set([int(f.split('_')[f.split('_').index('snr') + 1]) for f in stats['files_found']]))
        
        print(f"Total training samples: {stats['total_samples']:,}")
        print(f"Number of modulation classes: {stats['num_classes']}")
        print(f"Number of SNR levels: {stats['num_snr_levels']}")
        print(f"Files found: {len(stats['files_found'])}")
        
        return stats
    
    def _plot_class_distribution(self, stats, dataset_name, class_names):
        """Plot distribution of samples per modulation class"""
        if 'samples_per_mod' not in stats:
            return
            
        plt.figure(figsize=(12, 6))
        mods = list(stats['samples_per_mod'].keys())
        counts = [stats['samples_per_mod'][mod] for mod in mods]
        
        # Map to readable names
        if dataset_name == 'RML2016.10a':
            readable_names = [class_names[mods.index(mod)] if mod in mods else mod for mod in mods]
        else:
            readable_names = mods
            
        plt.bar(range(len(mods)), counts, color=sns.color_palette("husl", len(mods)))
        plt.xlabel('Modulation Type')
        plt.ylabel('Number of Samples')
        plt.title(f'{dataset_name} - Samples per Modulation Class')
        plt.xticks(range(len(mods)), readable_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_snr_distribution(self, stats, dataset_name):
        """Plot distribution of samples per SNR level"""
        plt.figure(figsize=(12, 6))
        snrs = sorted(stats['samples_per_snr'].keys())
        counts = [stats['samples_per_snr'][snr] for snr in snrs]
        
        plt.bar(snrs, counts, color='skyblue', alpha=0.7)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Number of Samples')
        plt.title(f'{dataset_name} - Samples per SNR Level')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_snr_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_heatmap(self, stats, dataset_name, mods, snrs):
        """Plot heatmap of samples per (modulation, SNR) combination"""
        if 'samples_per_mod_snr' not in stats:
            return
            
        # Create matrix
        matrix = np.zeros((len(mods), len(snrs)))
        for i, mod in enumerate(mods):
            for j, snr in enumerate(snrs):
                if (mod, snr) in stats['samples_per_mod_snr']:
                    matrix[i, j] = stats['samples_per_mod_snr'][(mod, snr)]
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(matrix, 
                   xticklabels=snrs, 
                   yticklabels=mods,
                   annot=True, 
                   fmt='g', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Number of Samples'})
        plt.xlabel('SNR (dB)')
        plt.ylabel('Modulation Type')
        plt.title(f'{dataset_name} - Sample Distribution Heatmap')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_signal_characteristics(self, Xd, mods, snrs, dataset_name, class_names):
        """Analyze and visualize signal characteristics"""
        print("\nAnalyzing signal characteristics...")
        
        # Select a few modulations and SNR levels for detailed analysis
        selected_mods = mods[:6] if len(mods) >= 6 else mods
        selected_snr = 10 if 10 in snrs else snrs[len(snrs)//2]
        
        # Plot constellation diagrams
        self._plot_constellation_diagrams(Xd, selected_mods, selected_snr, dataset_name, class_names)
        
        # Plot time series examples
        self._plot_time_series_examples(Xd, selected_mods, selected_snr, dataset_name, class_names)
        
        # Plot power spectral density
        self._plot_psd_examples(Xd, selected_mods, selected_snr, dataset_name, class_names)
    
    def _plot_constellation_diagrams(self, Xd, mods, snr, dataset_name, class_names):
        """Plot constellation diagrams for different modulations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, mod in enumerate(mods[:6]):
            if (mod, snr) in Xd:
                data = Xd[(mod, snr)]
                # Take first sample
                i_data = data[0, 0, :]  # I channel
                q_data = data[0, 1, :]  # Q channel
                
                axes[i].scatter(i_data, q_data, alpha=0.6, s=1)
                if dataset_name == 'RML2016.10a':
                    title = class_names[mods.index(mod)] if mod in mods else mod
                else:
                    title = mod
                axes[i].set_title(f'{title} @ {snr}dB')
                axes[i].set_xlabel('In-phase')
                axes[i].set_ylabel('Quadrature')
                axes[i].grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(mods), 6):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'{dataset_name} - Constellation Diagrams')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_constellations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series_examples(self, Xd, mods, snr, dataset_name, class_names):
        """Plot time series examples for different modulations"""
        fig, axes = plt.subplots(len(mods), 1, figsize=(12, 2*len(mods)))
        if len(mods) == 1:
            axes = [axes]
        
        for i, mod in enumerate(mods):
            if (mod, snr) in Xd:
                data = Xd[(mod, snr)]
                # Take first sample
                i_data = data[0, 0, :]  # I channel
                q_data = data[0, 1, :]  # Q channel
                time_axis = np.arange(len(i_data))
                
                axes[i].plot(time_axis, i_data, label='I', alpha=0.8)
                axes[i].plot(time_axis, q_data, label='Q', alpha=0.8)
                if dataset_name == 'RML2016.10a':
                    title = class_names[mods.index(mod)] if mod in mods else mod
                else:
                    title = mod
                axes[i].set_title(f'{title} @ {snr}dB')
                axes[i].set_ylabel('Amplitude')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Sample Index')
        plt.suptitle(f'{dataset_name} - Time Series Examples')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_psd_examples(self, Xd, mods, snr, dataset_name, class_names):
        """Plot power spectral density examples"""
        plt.figure(figsize=(12, 8))
        
        for i, mod in enumerate(mods):
            if (mod, snr) in Xd:
                data = Xd[(mod, snr)]
                # Take first sample and compute complex signal
                complex_signal = data[0, 0, :] + 1j * data[0, 1, :]
                
                # Compute PSD
                freqs, psd = signal.welch(complex_signal, nperseg=64, return_onesided=False)
                freqs = np.fft.fftshift(freqs)
                psd = np.fft.fftshift(psd)
                
                if dataset_name == 'RML2016.10a':
                    label = class_names[mods.index(mod)] if mod in mods else mod
                else:
                    label = mod
                plt.semilogy(freqs, psd, label=label, alpha=0.8)
        
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Power Spectral Density')
        plt.title(f'{dataset_name} - Power Spectral Density @ {snr}dB')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_psd.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, all_stats):
        """Generate a comprehensive summary report"""
        print("=" * 60)
        print("GENERATING SUMMARY REPORT")
        print("=" * 60)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        datasets = []
        total_samples = []
        num_classes = []
        num_snr_levels = []
        
        for dataset_name, stats in all_stats.items():
            if stats:
                datasets.append(dataset_name)
                total_samples.append(stats['total_samples'])
                num_classes.append(stats['num_classes'])
                num_snr_levels.append(stats['num_snr_levels'])
        
        # Create subplot for comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total samples
        bars1 = ax1.bar(datasets, total_samples, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_ylabel('Total Samples')
        ax1.set_title('Total Samples per Dataset')
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars1, total_samples):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_samples)*0.01,
                    f'{val:,}', ha='center', va='bottom')
        
        # Number of classes
        bars2 = ax2.bar(datasets, num_classes, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_ylabel('Number of Classes')
        ax2.set_title('Number of Modulation Classes')
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, num_classes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', va='bottom')
        
        # Number of SNR levels
        bars3 = ax3.bar(datasets, num_snr_levels, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_ylabel('Number of SNR Levels')
        ax3.set_title('Number of SNR Levels')
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars3, num_snr_levels):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', va='bottom')
        
        # Summary table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for i, dataset in enumerate(datasets):
            table_data.append([
                dataset,
                f"{total_samples[i]:,}",
                str(num_classes[i]),
                str(num_snr_levels[i])
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Dataset', 'Total Samples', 'Classes', 'SNR Levels'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax4.set_title('Dataset Comparison Summary', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Write text report
        with open(f'{self.output_dir}/analysis_report.txt', 'w') as f:
            f.write("RadioML Dataset Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset_name, stats in all_stats.items():
                if stats:
                    f.write(f"{dataset_name}:\n")
                    f.write(f"  Total Samples: {stats['total_samples']:,}\n")
                    f.write(f"  Classes: {stats['num_classes']}\n")
                    f.write(f"  SNR Levels: {stats['num_snr_levels']}\n")
                    if 'snr_range' in stats:
                        f.write(f"  SNR Range: {stats['snr_range'][0]} to {stats['snr_range'][1]} dB\n")
                    f.write("\n")
        
        print(f"Analysis complete! Results saved to '{self.output_dir}/' directory")
        print(f"Generated files:")
        for file in os.listdir(self.output_dir):
            print(f"  - {file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze RadioML datasets')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['rml2016a', 'rml2016b', 'rml2018', 'all'],
                       default=['all'],
                       help='Datasets to analyze')
    parser.add_argument('--output-dir', default='analysis_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer(args.output_dir)
    all_stats = {}
    
    datasets_to_analyze = ['rml2016a', 'rml2016b', 'rml2018'] if 'all' in args.datasets else args.datasets
    
    for dataset in datasets_to_analyze:
        if dataset == 'rml2016a':
            all_stats['RML2016.10a'] = analyzer.analyze_rml2016a()
        elif dataset == 'rml2016b':
            all_stats['RML2016.10b'] = analyzer.analyze_rml2016b()
        elif dataset == 'rml2018':
            all_stats['RML2018'] = analyzer.analyze_rml2018()
    
    # Generate summary report
    analyzer.generate_summary_report(all_stats)

if __name__ == "__main__":
    main()
