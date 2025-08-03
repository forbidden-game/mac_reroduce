"""
Preprocessing script for RML2016.10a dataset
Creates the required train/test dataset files for MAC training
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
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def preprocess_rml2016a():
    """
    Preprocess RML2016.10a dataset and create train/test files for each SNR level
    """
    print("Loading RML2016.10a dataset...")
    
    # Load raw data
    filename = 'data/RML2016.10a_dict.pkl'
    Xd = load_pickle(filename)
    
    # Get modulation types and SNR values
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]
    print(f"Modulation types: {mods}")
    print(f"SNR values: {snrs}")
    print(f"Number of modulations: {len(mods)}")
    print(f"Number of SNR levels: {len(snrs)}")
    
    # Set random seed for reproducibility
    np.random.seed(2016)
    torch.manual_seed(2016)
    
    # Create datasets for each SNR level
    for snr in tqdm(snrs, desc="Processing SNR levels"):
        print(f"\nProcessing SNR = {snr} dB")
        
        # Collect data for this SNR level
        X_snr = []
        y_snr = []
        
        for mod_idx, mod in enumerate(mods):
            if (mod, snr) in Xd:
                data = Xd[(mod, snr)]  # Shape: (1000, 2, 128)
                X_snr.append(data)
                # Create labels for this modulation type
                labels = np.full(data.shape[0], mod_idx, dtype=np.int64)
                y_snr.append(labels)
        
        if len(X_snr) == 0:
            print(f"No data found for SNR {snr}, skipping...")
            continue
            
        # Stack data
        X_snr = np.vstack(X_snr)  # Shape: (num_samples, 2, 128)
        y_snr = np.concatenate(y_snr)  # Shape: (num_samples,)
        
        print(f"Total samples for SNR {snr}: {X_snr.shape[0]}")
        print(f"Data shape: {X_snr.shape}")
        print(f"Label shape: {y_snr.shape}")
        
        # Create train/test split (80/20)
        n_samples = X_snr.shape[0]
        n_train = int(0.8 * n_samples)
        
        # Create indices and shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Split data
        X_train = X_snr[train_indices]
        y_train = y_snr[train_indices]
        X_test = X_snr[test_indices]
        y_test = y_snr[test_indices]
        
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).long()
        train_idx_tensor = torch.arange(len(X_train_tensor))
        
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).long()
        test_idx_tensor = torch.arange(len(X_test_tensor))
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_idx_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_idx_tensor)
        
        # Save datasets
        train_filename = f'data/{snr}_train_MV_dataset'
        test_filename = f'data/{snr}_test_MV_dataset'
        
        save_pickle(train_dataset, train_filename)
        save_pickle(test_dataset, test_filename)
        
        print(f"Saved {train_filename}")
        print(f"Saved {test_filename}")
    
    # Create ALL SNR dataset
    print("\nCreating ALL SNR dataset...")
    
    X_all = []
    y_all = []
    
    for mod_idx, mod in enumerate(mods):
        for snr in snrs:
            if (mod, snr) in Xd:
                data = Xd[(mod, snr)]
                X_all.append(data)
                labels = np.full(data.shape[0], mod_idx, dtype=np.int64)
                y_all.append(labels)
    
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    
    print(f"Total samples for ALL SNR: {X_all.shape[0]}")
    
    # Train/test split for ALL data
    n_samples_all = X_all.shape[0]
    n_train_all = int(0.8 * n_samples_all)
    
    indices_all = np.arange(n_samples_all)
    np.random.shuffle(indices_all)
    
    train_indices_all = indices_all[:n_train_all]
    test_indices_all = indices_all[n_train_all:]
    
    X_train_all = X_all[train_indices_all]
    y_train_all = y_all[train_indices_all]
    X_test_all = X_all[test_indices_all]
    y_test_all = y_all[test_indices_all]
    
    # Convert to tensors
    X_train_all_tensor = torch.from_numpy(X_train_all).float()
    y_train_all_tensor = torch.from_numpy(y_train_all).long()
    train_idx_all_tensor = torch.arange(len(X_train_all_tensor))
    
    X_test_all_tensor = torch.from_numpy(X_test_all).float()
    y_test_all_tensor = torch.from_numpy(y_test_all).long()
    test_idx_all_tensor = torch.arange(len(X_test_all_tensor))
    
    # Create datasets
    train_dataset_all = TensorDataset(X_train_all_tensor, y_train_all_tensor, train_idx_all_tensor)
    test_dataset_all = TensorDataset(X_test_all_tensor, y_test_all_tensor, test_idx_all_tensor)
    
    # Save ALL datasets
    save_pickle(train_dataset_all, 'data/train_ALL_SNR_MV_dataset')
    save_pickle(test_dataset_all, 'data/test_ALL_SNR_MV_dataset')
    
    print("Saved data/train_ALL_SNR_MV_dataset")
    print("Saved data/test_ALL_SNR_MV_dataset")
    
    print("\nPreprocessing completed successfully!")
    print(f"Generated datasets for {len(snrs)} SNR levels plus ALL SNR dataset")

def verify_datasets():
    """Verify that the generated datasets can be loaded correctly"""
    print("\nVerifying generated datasets...")
    
    # Test loading SNR=6 dataset (the one causing the original error)
    try:
        train_data = load_pickle('data/6_train_MV_dataset')
        test_data = load_pickle('data/6_test_MV_dataset')
        
        print(f"SNR=6 train dataset: {len(train_data)} samples")
        print(f"SNR=6 test dataset: {len(test_data)} samples")
        
        # Check tensor shapes
        sample_data, sample_label, sample_idx = train_data[0]
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample label: {sample_label}")
        print(f"Sample index: {sample_idx}")
        
        print("✅ SNR=6 datasets verified successfully!")
        
    except Exception as e:
        print(f"❌ Error verifying SNR=6 datasets: {e}")
    
    # Test ALL SNR dataset
    try:
        train_all = load_pickle('data/train_ALL_SNR_MV_dataset')
        test_all = load_pickle('data/test_ALL_SNR_MV_dataset')
        
        print(f"ALL SNR train dataset: {len(train_all)} samples")
        print(f"ALL SNR test dataset: {len(test_all)} samples")
        
        print("✅ ALL SNR datasets verified successfully!")
        
    except Exception as e:
        print(f"❌ Error verifying ALL SNR datasets: {e}")

if __name__ == "__main__":
    # Run preprocessing
    preprocess_rml2016a()
    
    # Verify results
    verify_datasets()
