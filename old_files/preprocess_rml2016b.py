#!/usr/bin/env python3
"""
Preprocessing script for RML2016.10b dataset
Handles .dat format and generates MT4 datasets as expected by the training code
"""

import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset
import os
from sklearn.model_selection import train_test_split

def load_rml2016b_pickle(file_path):
    """
    Load RML2016.10b dataset from pickle file (.dat extension but pickle format)
    """
    print(f"Loading RML2016.10b dataset from {file_path}")
    
    # Load the pickle file (despite .dat extension)
    with open(file_path, 'rb') as f:
        Xd = pickle.load(f, encoding='iso-8859-1')
    
    # Get modulation types and SNR values
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]
    print(f"Modulation types: {mods}")
    print(f"SNR values: {snrs}")
    print(f"Number of modulations: {len(mods)}")
    print(f"Number of SNR levels: {len(snrs)}")
    
    return Xd, mods, snrs

def save_tensor_dataset(data, labels, indices, filename):
    """Save data as TensorDataset in pickle format"""
    dataset = TensorDataset(
        torch.FloatTensor(data),
        torch.LongTensor(labels), 
        torch.LongTensor(indices)
    )
    
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved {len(data)} samples to {filename}")

def main():
    # Paths
    input_file = "data/raw/RML2016.10b.dat"
    output_dir = "data/processed/RML2016.10b/"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the dataset
        Xd, mods, snrs = load_rml2016b_pickle(input_file)
        
        # Set random seed for reproducibility
        np.random.seed(2016)
        
        # Process each SNR level
        for snr in snrs:
            print(f"\nProcessing SNR = {snr} dB")
            
            # Collect data for this SNR level
            X_snr = []
            y_snr = []
            
            for mod_idx, mod in enumerate(mods):
                if (mod, snr) in Xd:
                    data = Xd[(mod, snr)]  # Shape: (samples, 2, 128)
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
            
            # Create indices for datasets (required by MAC framework)
            train_idx_tensor = np.arange(len(X_train))
            test_idx_tensor = np.arange(len(X_test))
            
            # Save datasets with MT4 naming convention as expected by util.py
            train_filename = os.path.join(output_dir, f"{snr}_MT4_train_dataset")
            test_filename = os.path.join(output_dir, f"{snr}_MT4_test_dataset")
            
            save_tensor_dataset(X_train, y_train, train_idx_tensor, train_filename)
            save_tensor_dataset(X_test, y_test, test_idx_tensor, test_filename)
            
            print(f"Saved {train_filename}")
            print(f"Saved {test_filename}")
        
        print("\nRML2016.10b preprocessing completed successfully!")
        print(f"Generated datasets for {len(snrs)} SNR levels")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Note: This is a template script. You may need to adjust the data loading")
        print("logic based on the actual format of your RML2016.10b .dat file.")

if __name__ == "__main__":
    main()
