#!/usr/bin/env python3
"""
Verification script for corrected 6:3:1 dataset splits.
This script verifies that our corrected preprocessing has achieved proper 6:3:1 splits.
"""

import pickle
import torch
import os
import numpy as np
from collections import Counter

def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data

def analyze_dataset_split(dataset_name, base_path, snr_values, file_patterns):
    """Analyze dataset splits for all SNR values."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {dataset_name} DATASET SPLITS")
    print(f"{'='*60}")
    
    total_stats = {}
    
    for snr in snr_values:
        print(f"\n--- SNR {snr} dB ---")
        
        # Load train, val, test datasets
        splits_data = {}
        splits_found = {}
        
        for split_name, pattern in file_patterns.items():
            if snr == "ALL":
                filename = os.path.join(base_path, pattern.format(snr="ALL_SNR"))
            else:
                filename = os.path.join(base_path, pattern.format(snr=snr))
            
            splits_found[split_name] = os.path.exists(filename)
            
            if splits_found[split_name]:
                try:
                    dataset = load_pickle(filename)
                    splits_data[split_name] = dataset
                    print(f"✓ {split_name}: {len(dataset)} samples")
                except Exception as e:
                    print(f"✗ Error loading {split_name}: {e}")
                    splits_found[split_name] = False
            else:
                print(f"✗ {split_name}: File not found")
        
        # Calculate split ratios if all splits are found
        if all(splits_found.values()):
            train_size = len(splits_data['train'])
            val_size = len(splits_data['val'])
            test_size = len(splits_data['test'])
            total_size = train_size + val_size + test_size
            
            train_ratio = train_size / total_size
            val_ratio = val_size / total_size
            test_ratio = test_size / total_size
            
            print(f"Train: {train_size:,} ({train_ratio:.3f}) - Target: 0.600")
            print(f"Val:   {val_size:,} ({val_ratio:.3f}) - Target: 0.300")
            print(f"Test:  {test_size:,} ({test_ratio:.3f}) - Target: 0.100")
            print(f"Total: {total_size:,}")
            
            # Check if ratios are close to 6:3:1
            train_ok = abs(train_ratio - 0.6) < 0.01
            val_ok = abs(val_ratio - 0.3) < 0.01
            test_ok = abs(test_ratio - 0.1) < 0.01
            
            if train_ok and val_ok and test_ok:
                print("✓ Split ratios are CORRECT (within 1% tolerance)")
            else:
                print("✗ Split ratios are INCORRECT")
                if not train_ok:
                    print(f"  - Train ratio error: {abs(train_ratio - 0.6):.3f}")
                if not val_ok:
                    print(f"  - Val ratio error: {abs(val_ratio - 0.3):.3f}")
                if not test_ok:
                    print(f"  - Test ratio error: {abs(test_ratio - 0.1):.3f}")
            
            # Analyze class distribution
            print("\nClass distribution analysis:")
            for split_name, dataset in splits_data.items():
                labels = dataset.tensors[1]
                class_counts = Counter(labels.numpy())
                print(f"{split_name.capitalize()}: {len(class_counts)} classes, "
                      f"samples per class: {min(class_counts.values())}-{max(class_counts.values())}")
            
            total_stats[snr] = {
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size,
                'total_size': total_size,
                'ratios': (train_ratio, val_ratio, test_ratio),
                'ratios_correct': train_ok and val_ok and test_ok
            }
        else:
            print("Cannot calculate ratios - missing split files")
            total_stats[snr] = {'error': 'missing files'}
    
    return total_stats

def main():
    print("DATASET SPLIT VERIFICATION")
    print("="*60)
    print("Verifying that corrected preprocessing achieved proper 6:3:1 splits")
    print("Target ratios: Train=60%, Validation=30%, Test=10%")
    
    # RML2016.10A verification
    rml2016a_snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    rml2016a_patterns = {
        'train': '{snr}_train_MV_dataset',
        'val': '{snr}_val_MV_dataset',
        'test': '{snr}_test_MV_dataset'
    }
    
    rml2016a_stats = analyze_dataset_split(
        "RML2016.10A",
        "data/processed/RML2016.10a",
        rml2016a_snrs,
        rml2016a_patterns
    )
    
    # RML2016.10B verification
    rml2016b_snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    rml2016b_patterns = {
        'train': '{snr}_MT4_train_dataset',
        'val': '{snr}_MT4_val_dataset',
        'test': '{snr}_MT4_test_dataset'
    }
    
    rml2016b_stats = analyze_dataset_split(
        "RML2016.10B",
        "data/processed/RML2016.10b",
        rml2016b_snrs,
        rml2016b_patterns
    )
    
    # RML2018 verification (check if new files exist)
    print(f"\n{'='*60}")
    print("CHECKING RML2018 DATASET")
    print(f"{'='*60}")
    
    rml2018_base = "data/processed/RML2018"
    rml2018_snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    
    # Check for validation files
    val_files_exist = 0
    for snr in rml2018_snrs:
        val_file = os.path.join(rml2018_base, f"RML2018_MV4_snr_{snr}_val_dataset")
        if os.path.exists(val_file):
            val_files_exist += 1
    
    if val_files_exist > 0:
        print(f"Found {val_files_exist} validation files for RML2018")
        rml2018_patterns = {
            'train': 'RML2018_MV4_snr_{snr}_train_dataset',
            'val': 'RML2018_MV4_snr_{snr}_val_dataset',
            'test': 'RML2018_MV4_snr_{snr}_test_dataset'
        }
        rml2018_stats = analyze_dataset_split(
            "RML2018",
            rml2018_base,
            [0, 10],  # Test with a few SNRs
            rml2018_patterns
        )
    else:
        print("No validation files found for RML2018 - preprocessing was not completed")
        print("RML2018 still needs proper 6:3:1 preprocessing")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    def summarize_stats(dataset_name, stats):
        print(f"\n{dataset_name}:")
        correct_count = 0
        total_count = 0
        for snr, data in stats.items():
            if 'ratios_correct' in data:
                total_count += 1
                if data['ratios_correct']:
                    correct_count += 1
        
        if total_count > 0:
            print(f"  ✓ {correct_count}/{total_count} SNR levels have correct 6:3:1 splits")
            if correct_count == total_count:
                print(f"  ✓ ALL splits are CORRECT!")
            else:
                print(f"  ⚠ {total_count - correct_count} splits need attention")
        else:
            print(f"  ✗ No valid splits found")
    
    summarize_stats("RML2016.10A", rml2016a_stats)
    summarize_stats("RML2016.10B", rml2016b_stats)
    
    if val_files_exist > 0:
        summarize_stats("RML2018", rml2018_stats)
    else:
        print(f"\nRML2018:")
        print(f"  ⚠ Needs corrected preprocessing to create validation splits")
    
    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
