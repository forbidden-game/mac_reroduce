#!/usr/bin/env python3
"""
CORRECTED Preprocessing script for RML2018 dataset
Implements proper 6:3:1 split as specified in the MAC paper
Handles HDF5 format and preserves exact data structure
"""
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import os
import h5py
from tqdm import tqdm

def save_pickle(data, file_name):
    """Save data to pickle file"""
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def load_pickle(file_name):
    """Load pickle file"""
    with open(file_name, "rb") as f:
        data = pickle.load(f, encoding='iso-8859-1')
    return data

def create_631_split_rml2018():
    """
    Create proper 6:3:1 splits for RML2018 dataset
    Based on the format information provided by user
    """
    print("=" * 60)
    print("CREATING CORRECTED 6:3:1 SPLITS FOR RML2018")
    print("=" * 60)
    
    # Load raw data
    filename = 'data/raw/GOLD_XYZ_OSC.0001_1024.hdf5'
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        return False
    
    print(f"Loading data from {filename}")
    
    with h5py.File(filename, 'r') as f:
        print(f"Available keys in HDF5 file: {list(f.keys())}")
        
        # Load data, labels, and SNR values
        if 'X' in f.keys():
            data = f['X'][:]  # Signal data
            print(f"Data shape: {data.shape}")
        else:
            print("Error: 'X' key not found in HDF5 file")
            return False
            
        if 'Y' in f.keys():
            labels = f['Y'][:]  # Modulation labels
            print(f"Labels shape: {labels.shape}")
        else:
            print("Error: 'Y' key not found in HDF5 file")
            return False
            
        if 'Z' in f.keys():
            snr_values = f['Z'][:]  # SNR values
            print(f"SNR values shape: {snr_values.shape}")
            if len(snr_values.shape) > 1:
                snr_values = snr_values.flatten()
        else:
            print("Error: 'Z' key not found in HDF5 file")
            return False
    
    # Process labels (convert from one-hot if needed)
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        # One-hot encoded, convert to integer labels
        labels = np.argmax(labels, axis=1)
    labels = labels.astype(np.int64)
    
    # Ensure data is in correct format (samples, 2, 1024)
    print(f"Original data shape: {data.shape}")
    if len(data.shape) == 3 and data.shape[2] == 2:
        # Transpose from (samples, 1024, 2) to (samples, 2, 1024)
        print("Transposing data from (samples, 1024, 2) to (samples, 2, 1024)")
        data = np.transpose(data, (0, 2, 1))
    elif len(data.shape) == 3 and data.shape[1] == 2:
        print("Data already in correct format (samples, 2, 1024)")
    else:
        print(f"Warning: Unexpected data shape {data.shape}")
        # Try to reshape assuming we have the right number of elements
        if data.shape[0] > 1000:  # Likely samples dimension
            samples = data.shape[0]
            remaining = np.prod(data.shape[1:])
            if remaining == 2048:  # 2 * 1024
                data = data.reshape(samples, 2, 1024)
                print(f"Reshaped to: {data.shape}")
            else:
                print(f"Cannot reshape data properly")
                return False
    
    print(f"Final data shape: {data.shape}")
    
    # RML2018 class names (24 classes total)
    class_names = [
        'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
        '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', 
        '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 
        'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
    ]
    
    # Get unique values
    unique_snrs = np.unique(snr_values)
    unique_classes = np.unique(labels)
    
    print(f"Unique SNR values: {unique_snrs}")
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Label range: {labels.min()} to {labels.max()}")
    print(f"Total samples: {data.shape[0]:,}")
    
    # Use same seed for reproducibility
    np.random.seed(2018)
    
    # Create 6:3:1 splits for entire dataset
    print(f"\nCreating 6:3:1 splits...")
    
    # For each (modulation, SNR) combination, create stratified split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for mod_class in unique_classes:
        for snr in unique_snrs:
            # Find samples for this (mod, snr) combination
            mask = (labels == mod_class) & (snr_values == snr)
            class_snr_indices = np.where(mask)[0]
            
            if len(class_snr_indices) == 0:
                continue
                
            # Create 6:3:1 split for this combination
            n_samples = len(class_snr_indices)
            n_train = int(0.6 * n_samples)
            n_val = int(0.3 * n_samples) 
            n_test = n_samples - n_train - n_val
            
            # Shuffle indices for this combination
            np.random.shuffle(class_snr_indices)
            
            # Split
            train_indices.extend(class_snr_indices[:n_train])
            val_indices.extend(class_snr_indices[n_train:n_train+n_val])
            test_indices.extend(class_snr_indices[n_train+n_val:])
    
    # Convert to numpy arrays and shuffle
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    # Split data
    X_train = data[train_indices]
    X_val = data[val_indices]
    X_test = data[test_indices]
    
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]
    
    n_total = len(data)
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_indices)} samples ({len(train_indices)/n_total*100:.1f}%)")
    print(f"Val:   {len(val_indices)} samples ({len(val_indices)/n_total*100:.1f}%)")
    print(f"Test:  {len(test_indices)} samples ({len(test_indices)/n_total*100:.1f}%)")
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")
    
    print(f"\nLabel shapes:")
    print(f"y_train: {y_train.shape}")
    print(f"y_val:   {y_val.shape}")
    print(f"y_test:  {y_test.shape}")
    
    # Create TensorDatasets for each SNR level
    print(f"\nSaving individual SNR datasets...")
    for snr in tqdm(unique_snrs, desc="Processing SNR levels"):
        # Get train indices for this SNR
        train_snr_mask = snr_values[train_indices] == snr
        val_snr_mask = snr_values[val_indices] == snr
        test_snr_mask = snr_values[test_indices] == snr
        
        if not np.any(train_snr_mask):
            continue
            
        # Extract data for this SNR
        X_snr_train = X_train[train_snr_mask]
        X_snr_val = X_val[val_snr_mask]
        X_snr_test = X_test[test_snr_mask]
        
        y_snr_train = y_train[train_snr_mask]
        y_snr_val = y_val[val_snr_mask]
        y_snr_test = y_test[test_snr_mask]
        
        # Create TensorDatasets for this SNR
        snr_train_dataset = TensorDataset(
            torch.from_numpy(X_snr_train).float(),
            torch.from_numpy(y_snr_train).long(),
            torch.arange(len(X_snr_train))
        )
        
        snr_val_dataset = TensorDataset(
            torch.from_numpy(X_snr_val).float(),
            torch.from_numpy(y_snr_val).long(),
            torch.arange(len(X_snr_val))
        )
        
        snr_test_dataset = TensorDataset(
            torch.from_numpy(X_snr_test).float(),
            torch.from_numpy(y_snr_test).long(),
            torch.arange(len(X_snr_test))
        )
        
        # Save with MV4 naming convention for compatibility (same as util.py expects)
        save_pickle(snr_train_dataset, f'data/processed/RML2018/RML2018_MV4_snr_{snr}_train_dataset')
        save_pickle(snr_val_dataset, f'data/processed/RML2018/RML2018_MV4_snr_{snr}_val_dataset')
        save_pickle(snr_test_dataset, f'data/processed/RML2018/RML2018_MV4_snr_{snr}_test_dataset')
    
    # Save split information for verification
    split_info = {
        'class_names': class_names,
        'unique_snrs': unique_snrs.tolist(),
        'unique_classes': unique_classes.tolist(),
        'train_indices': train_indices.tolist(),
        'val_indices': val_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices),
        'total_size': n_total,
        'train_ratio': len(train_indices) / n_total,
        'val_ratio': len(val_indices) / n_total,
        'test_ratio': len(test_indices) / n_total
    }
    
    save_pickle(split_info, 'data/processed/RML2018/split_info.pkl')
    
    print(f"\n✅ RML2018 preprocessing complete!")
    print(f"Generated {len(unique_snrs)} SNR-specific datasets")
    print(f"Split ratios: Train={split_info['train_ratio']:.3f}, Val={split_info['val_ratio']:.3f}, Test={split_info['test_ratio']:.3f}")
    
    return True

def verify_rml2018_splits():
    """Verify the generated splits are correct"""
    print("\n" + "=" * 60)
    print("VERIFYING RML2018 SPLITS")
    print("=" * 60)
    
    # Load split info
    split_info = load_pickle('data/processed/RML2018/split_info.pkl')
    
    print(f"Total samples: {split_info['total_size']:,}")
    print(f"Train: {split_info['train_size']:,} ({split_info['train_ratio']:.1%})")
    print(f"Val:   {split_info['val_size']:,} ({split_info['val_ratio']:.1%})")
    print(f"Test:  {split_info['test_size']:,} ({split_info['test_ratio']:.1%})")
    
    # Check if ratios are close to 6:3:1
    expected_ratios = [0.6, 0.3, 0.1]
    actual_ratios = [split_info['train_ratio'], split_info['val_ratio'], split_info['test_ratio']]
    
    print(f"\n6:3:1 Ratio Check:")
    for name, expected, actual in zip(['Train', 'Val', 'Test'], expected_ratios, actual_ratios):
        diff = abs(expected - actual)
        status = "✅" if diff < 0.01 else "❌"
        print(f"{status} {name}: Expected {expected:.1%}, Got {actual:.1%} (diff: {diff:.3f})")
    
    # Test loading a few datasets
    print(f"\nTesting dataset loading...")
    try:
        # Test one SNR dataset (check what SNRs we have)
        available_snrs = split_info['unique_snrs']
        test_snr = available_snrs[len(available_snrs)//2]  # Pick middle SNR
        
        train_file = f'data/processed/RML2018/RML2018_MV4_snr_{test_snr}_train_dataset'
        val_file = f'data/processed/RML2018/RML2018_MV4_snr_{test_snr}_val_dataset'
        test_file = f'data/processed/RML2018/RML2018_MV4_snr_{test_snr}_test_dataset'
        
        train_dataset = load_pickle(train_file)
        val_dataset = load_pickle(val_file)
        test_dataset = load_pickle(test_file)
        
        print(f"✅ SNR={test_snr} datasets loaded successfully")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val:   {len(val_dataset)} samples")
        print(f"   Test:  {len(test_dataset)} samples")
        
        # Test data format
        sample_data, sample_label, sample_idx = train_dataset[0]
        print(f"✅ Data format verified: shape={sample_data.shape}, label={sample_label.item()}")
        print(f"   Expected shape: (2, 1024), Got: {sample_data.shape}")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Create corrected 6:3:1 splits
    success = create_631_split_rml2018()
    
    if success:
        # Verify the splits
        verify_rml2018_splits()
    else:
        print("❌ Failed to create RML2018 splits")
