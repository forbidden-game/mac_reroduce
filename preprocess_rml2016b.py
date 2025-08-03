#!/usr/bin/env python3
"""
CORRECTED Preprocessing script for RML2016.10b dataset
Implements proper 6:3:1 split as specified in the MAC paper
Preserves exact data format from data201610b.py
"""
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import os
from tqdm import tqdm

def load_pickle(file_name):
    """Load pickle file"""
    with open(file_name, "rb") as f:
        data = pickle.load(f, encoding='iso-8859-1')
    return data

def save_pickle(data, file_name):
    """Save data to pickle file"""
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def create_631_split_rml2016b():
    """
    Create proper 6:3:1 splits for RML2016.10b dataset
    Following the exact logic from data201610b.py but with correct ratios
    """
    print("=" * 60)
    print("CREATING CORRECTED 6:3:1 SPLITS FOR RML2016.10b")
    print("=" * 60)
    
    # Load raw data
    filename = 'data/raw/RML2016.10b.dat'
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        return False
    
    print(f"Loading data from {filename}")
    Xd = load_pickle(filename)
    
    # Get modulation types and SNR values (same as original)
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]
    print(f"Modulations: {mods}")
    print(f"SNR levels: {snrs}")
    print(f"Number of modulations: {len(mods)}")
    print(f"Number of SNR levels: {len(snrs)}")
    
    # Use same seed as original for reproducibility
    np.random.seed(2016)
    
    X = []
    lbl = []
    train_idx = []
    val_idx = []
    a = 0
    
    print("\nCreating 6:3:1 splits...")
    
    # Build full dataset first (same as original)
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])  # ndarray(6000,2,128)
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
            
            # CORRECTED: Create 6:3:1 split instead of 6:4
            # For each (mod, snr) combination with 6000 samples:
            # - 3600 samples for training (60%)
            # - 1800 samples for validation (30%) 
            # - 600 samples for testing (10%)
            
            total_range = range(a * 6000, (a + 1) * 6000)
            train_samples = list(np.random.choice(total_range, size=3600, replace=False))
            remaining_samples = list(set(total_range) - set(train_samples))
            val_samples = list(np.random.choice(remaining_samples, size=1800, replace=False))
            
            train_idx += train_samples
            val_idx += val_samples
            a += 1
    
    # Stack data (same as original)
    X = np.vstack(X)  # Total samples depend on mods * snrs * 6000
    print(f"Total samples: {X.shape[0]}")
    print(f"Label count: {len(lbl)}")
    
    # Create test indices (remaining samples)
    n_examples = X.shape[0]
    test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))
    
    # Shuffle indices (same as original)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    # Split data
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_idx)} samples ({len(train_idx)/n_examples*100:.1f}%)")
    print(f"Val:   {len(val_idx)} samples ({len(val_idx)/n_examples*100:.1f}%)")
    print(f"Test:  {len(test_idx)} samples ({len(test_idx)/n_examples*100:.1f}%)")
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")
    
    # Create labels (same as original logic)
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1
    
    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    
    print(f"\nLabel shapes:")
    print(f"Y_train: {Y_train.shape}")
    print(f"Y_val:   {Y_val.shape}")
    print(f"Y_test:  {Y_test.shape}")
    
    # Convert to integer labels for MAC framework
    y_train_int = np.argmax(Y_train, axis=1)
    y_val_int = np.argmax(Y_val, axis=1)
    y_test_int = np.argmax(Y_test, axis=1)
    
    # Create TensorDatasets (MAC framework format)
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train_int).long(),
        torch.arange(len(X_train))
    )
    
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val_int).long(),
        torch.arange(len(X_val))
    )
    
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test_int).long(),
        torch.arange(len(X_test))
    )
    
    # Save datasets for each SNR level
    print(f"\nSaving individual SNR datasets...")
    for snr in tqdm(snrs, desc="Processing SNR levels"):
        # Filter data for this SNR level
        snr_train_indices = [i for i in train_idx if lbl[i][1] == snr]
        snr_val_indices = [i for i in val_idx if lbl[i][1] == snr]
        snr_test_indices = [i for i in test_idx if lbl[i][1] == snr]
        
        if len(snr_train_indices) == 0:
            continue
            
        # Extract data for this SNR
        X_snr_train = X[snr_train_indices]
        X_snr_val = X[snr_val_indices]
        X_snr_test = X[snr_test_indices]
        
        y_snr_train = np.array([mods.index(lbl[i][0]) for i in snr_train_indices])
        y_snr_val = np.array([mods.index(lbl[i][0]) for i in snr_val_indices])
        y_snr_test = np.array([mods.index(lbl[i][0]) for i in snr_test_indices])
        
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
        
        # Save with MT4 naming convention for compatibility (same as util.py expects)
        save_pickle(snr_train_dataset, f'data/processed/RML2016.10b/{snr}_MT4_train_dataset')
        save_pickle(snr_val_dataset, f'data/processed/RML2016.10b/{snr}_MT4_val_dataset')
        save_pickle(snr_test_dataset, f'data/processed/RML2016.10b/{snr}_MT4_test_dataset')
    
    # Save split information for verification
    split_info = {
        'mods': mods,
        'snrs': snrs,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'total_size': n_examples,
        'train_ratio': len(train_idx) / n_examples,
        'val_ratio': len(val_idx) / n_examples,
        'test_ratio': len(test_idx) / n_examples
    }
    
    save_pickle(split_info, 'data/processed/RML2016.10b/split_info.pkl')
    
    print(f"\n✅ RML2016.10b preprocessing complete!")
    print(f"Generated {len(snrs)} SNR-specific datasets")
    print(f"Split ratios: Train={split_info['train_ratio']:.3f}, Val={split_info['val_ratio']:.3f}, Test={split_info['test_ratio']:.3f}")
    
    return True

def verify_rml2016b_splits():
    """Verify the generated splits are correct"""
    print("\n" + "=" * 60)
    print("VERIFYING RML2016.10b SPLITS")
    print("=" * 60)
    
    # Load split info
    split_info = load_pickle('data/processed/RML2016.10b/split_info.pkl')
    
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
        # Test one SNR dataset (e.g., SNR=6)
        if 6 in split_info['snrs']:
            train_6 = load_pickle('data/processed/RML2016.10b/6_MT4_train_dataset')
            val_6 = load_pickle('data/processed/RML2016.10b/6_MT4_val_dataset')
            test_6 = load_pickle('data/processed/RML2016.10b/6_MT4_test_dataset')
            
            print(f"✅ SNR=6 datasets loaded successfully")
            print(f"   Train: {len(train_6)} samples")
            print(f"   Val:   {len(val_6)} samples")
            print(f"   Test:  {len(test_6)} samples")
            
            # Test data format
            sample_data, sample_label, sample_idx = train_6[0]
            print(f"✅ Data format verified: shape={sample_data.shape}, label={sample_label.item()}")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Create corrected 6:3:1 splits
    success = create_631_split_rml2016b()
    
    if success:
        # Verify the splits
        verify_rml2016b_splits()
    else:
        print("❌ Failed to create RML2016.10b splits")
