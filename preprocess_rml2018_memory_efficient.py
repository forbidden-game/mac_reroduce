#!/usr/bin/env python3
"""
Memory-efficient RML2018 preprocessing with correct 6:3:1 splits.
Processes large dataset in chunks to avoid memory issues.
"""

import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import os
from collections import Counter
import gc
import sys

def create_corrected_splits_memory_efficient():
    """Create corrected 6:3:1 splits for RML2018 with memory efficiency."""
    
    print("============================================================")
    print("CREATING CORRECTED 6:3:1 SPLITS FOR RML2018 (MEMORY EFFICIENT)")
    print("============================================================")
    
    # Load data info first to plan memory usage
    data_file = "data/raw/GOLD_XYZ_OSC.0001_1024.hdf5"
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return
    
    print(f"Loading data info from {data_file}")
    
    with h5py.File(data_file, 'r') as f:
        print(f"Available keys in HDF5 file: {list(f.keys())}")
        
        # Get dataset info without loading into memory
        X_shape = f['X'].shape
        Y_shape = f['Y'].shape
        Z_shape = f['Z'].shape
        
        print(f"Data shape: {X_shape}")
        print(f"Labels shape: {Y_shape}")
        print(f"SNR values shape: {Z_shape}")
        
        total_samples = X_shape[0]
        print(f"Total samples: {total_samples:,}")
        
        # Load SNR values and labels to determine splits
        print("Loading SNR values and labels...")
        snr_values = f['Z'][:]
        labels = f['Y'][:]
        
        # Get unique SNR values
        unique_snrs = np.unique(snr_values)
        print(f"Unique SNR values: {unique_snrs}")
        
        # Get unique classes
        unique_classes = np.unique(np.argmax(labels, axis=1))
        n_classes = len(unique_classes)
        print(f"Number of classes: {n_classes}")
        print(f"Label range: {unique_classes.min()} to {unique_classes.max()}")
        
        # Convert one-hot labels to class indices
        label_indices = np.argmax(labels, axis=1)
        
        print("\nCreating 6:3:1 splits...")
        
        # Process each SNR separately to manage memory
        for snr in unique_snrs:
            print(f"\nProcessing SNR {snr} dB...")
            
            # Find indices for this SNR
            snr_mask = (snr_values.flatten() == snr)
            snr_indices = np.where(snr_mask)[0]
            snr_samples = len(snr_indices)
            
            print(f"  Samples at SNR {snr}: {snr_samples:,}")
            
            if snr_samples == 0:
                print(f"  Skipping SNR {snr} (no samples)")
                continue
            
            # Calculate split sizes (6:3:1)
            train_size = int(snr_samples * 0.6)
            val_size = int(snr_samples * 0.3)
            test_size = snr_samples - train_size - val_size  # Remaining samples
            
            print(f"  Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
            
            # Create stratified splits by class
            train_indices = []
            val_indices = []
            test_indices = []
            
            # Get labels for this SNR
            snr_labels = label_indices[snr_mask]
            
            # Process each class separately for stratified sampling
            for class_idx in unique_classes:
                class_mask = (snr_labels == class_idx)
                class_indices_in_snr = np.where(class_mask)[0]
                class_global_indices = snr_indices[class_indices_in_snr]
                
                n_class_samples = len(class_global_indices)
                if n_class_samples == 0:
                    continue
                
                # Shuffle class samples
                np.random.seed(42)  # For reproducibility
                shuffled_class_indices = np.random.permutation(class_global_indices)
                
                # Calculate class split sizes
                class_train_size = int(n_class_samples * 0.6)
                class_val_size = int(n_class_samples * 0.3)
                class_test_size = n_class_samples - class_train_size - class_val_size
                
                # Split class samples
                train_indices.extend(shuffled_class_indices[:class_train_size])
                val_indices.extend(shuffled_class_indices[class_train_size:class_train_size + class_val_size])
                test_indices.extend(shuffled_class_indices[class_train_size + class_val_size:])
            
            # Convert to numpy arrays and SORT for HDF5 compatibility
            train_indices = np.sort(np.array(train_indices))
            val_indices = np.sort(np.array(val_indices))
            test_indices = np.sort(np.array(test_indices))
            
            print(f"  Final split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
            
            # Load and save data in chunks to manage memory
            chunk_size = 10000  # Process 10k samples at a time
            
            for split_name, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
                if len(indices) == 0:
                    continue
                    
                print(f"  Processing {split_name} split...")
                
                # Process in chunks
                all_data = []
                all_labels = []
                all_snr_labels = []
                
                for chunk_start in range(0, len(indices), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(indices))
                    chunk_indices = indices[chunk_start:chunk_end]
                    
                    print(f"    Loading chunk {chunk_start//chunk_size + 1}/{(len(indices)-1)//chunk_size + 1}")
                    
                    # Load data chunk
                    chunk_data = f['X'][chunk_indices]
                    chunk_labels = f['Y'][chunk_indices]
                    chunk_snrs = f['Z'][chunk_indices]
                    
                    # Transpose data from (samples, 1024, 2) to (samples, 2, 1024)
                    chunk_data = np.transpose(chunk_data, (0, 2, 1))
                    
                    # Convert to tensors
                    chunk_data_tensor = torch.from_numpy(chunk_data).float()
                    chunk_labels_tensor = torch.from_numpy(np.argmax(chunk_labels, axis=1)).long()
                    chunk_indices_tensor = torch.from_numpy(chunk_indices).long()
                    
                    all_data.append(chunk_data_tensor)
                    all_labels.append(chunk_labels_tensor)
                    all_snr_labels.append(chunk_indices_tensor)
                    
                    # Force garbage collection
                    del chunk_data, chunk_labels, chunk_snrs
                    gc.collect()
                
                # Concatenate all chunks
                print(f"    Concatenating {len(all_data)} chunks...")
                final_data = torch.cat(all_data, dim=0)
                final_labels = torch.cat(all_labels, dim=0)
                final_indices = torch.cat(all_snr_labels, dim=0)
                
                # Create dataset
                dataset = TensorDataset(final_data, final_labels, final_indices)
                
                # Verify class distribution
                class_counts = Counter(final_labels.numpy())
                print(f"    {split_name.capitalize()}: {len(dataset)} samples, {len(class_counts)} classes")
                print(f"    Samples per class: {min(class_counts.values())}-{max(class_counts.values())}")
                
                # Save dataset
                output_dir = "data/processed/RML2018"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = f"RML2018_MV4_snr_{snr}_{split_name}_dataset"
                filepath = os.path.join(output_dir, filename)
                
                print(f"    Saving to {filepath}...")
                with open(filepath, 'wb') as f_out:
                    pickle.dump(dataset, f_out)
                
                # Clean up memory
                del all_data, all_labels, all_snr_labels, final_data, final_labels, final_indices, dataset
                gc.collect()
                
                print(f"    ✓ {split_name.capitalize()} split saved successfully")
            
            print(f"✓ SNR {snr} dB completed")
            
            # Force garbage collection between SNRs
            gc.collect()
    
    print("\n" + "="*60)
    print("RML2018 PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)

def verify_rml2018_splits():
    """Verify the created RML2018 splits."""
    print("\n" + "="*60)
    print("VERIFYING RML2018 SPLITS")
    print("="*60)
    
    base_dir = "data/processed/RML2018"
    snr_values = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    
    for snr in snr_values[:3]:  # Check first 3 SNRs
        print(f"\nSNR {snr} dB:")
        
        splits = {}
        for split_name in ['train', 'val', 'test']:
            filename = f"RML2018_MV4_snr_{snr}_{split_name}_dataset"
            filepath = os.path.join(base_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        dataset = pickle.load(f)
                    splits[split_name] = len(dataset)
                    print(f"  {split_name.capitalize()}: {len(dataset):,} samples")
                except Exception as e:
                    print(f"  {split_name.capitalize()}: Error loading - {e}")
            else:
                print(f"  {split_name.capitalize()}: File not found")
        
        if len(splits) == 3:
            total = sum(splits.values())
            train_ratio = splits['train'] / total
            val_ratio = splits['val'] / total
            test_ratio = splits['test'] / total
            
            print(f"  Ratios - Train: {train_ratio:.3f}, Val: {val_ratio:.3f}, Test: {test_ratio:.3f}")
            
            if abs(train_ratio - 0.6) < 0.01 and abs(val_ratio - 0.3) < 0.01 and abs(test_ratio - 0.1) < 0.01:
                print(f"  ✓ Split ratios are CORRECT")
            else:
                print(f"  ✗ Split ratios are INCORRECT")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Memory efficient RML2018 preprocessing")
    print("This will process the large dataset in chunks to avoid memory issues.")
    
    try:
        create_corrected_splits_memory_efficient()
        verify_rml2018_splits()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
