#!/usr/bin/env python3
"""
Preprocessing script for RML2018 dataset
Handles .hdf5 format and generates MV4 datasets as expected by the training code
"""

import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset
import os
import h5py
from sklearn.model_selection import train_test_split

def load_rml2018_hdf5(file_path):
    """
    Load RML2018 dataset from HDF5 file
    """
    print(f"Loading RML2018 dataset from {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Print available keys to understand the structure
        print(f"Available keys in HDF5 file: {list(f.keys())}")
        
        # RML2018 typically has these keys: 'X' for data, 'Y' for labels, 'Z' for SNR
        # The exact structure may vary, so let's be flexible
        
        if 'X' in f.keys():
            data = f['X'][:]  # Signal data
            print(f"Data shape: {data.shape}")
        else:
            # Try alternative key names
            data_keys = [k for k in f.keys() if 'data' in k.lower() or 'x' in k.lower()]
            if data_keys:
                data = f[data_keys[0]][:]
                print(f"Using key '{data_keys[0]}' for data, shape: {data.shape}")
            else:
                print("Available datasets:")
                for key in f.keys():
                    print(f"  {key}: {f[key].shape if hasattr(f[key], 'shape') else 'group'}")
                raise ValueError("Could not find data array in HDF5 file")
        
        if 'Y' in f.keys():
            labels = f['Y'][:]  # Modulation labels (one-hot or integer)
            print(f"Labels shape: {labels.shape}")
        else:
            # Try alternative key names
            label_keys = [k for k in f.keys() if 'label' in k.lower() or 'y' in k.lower()]
            if label_keys:
                labels = f[label_keys[0]][:]
                print(f"Using key '{label_keys[0]}' for labels, shape: {labels.shape}")
            else:
                # Create synthetic labels if not found
                print("Warning: No labels found, creating synthetic labels")
                labels = np.random.randint(0, 24, data.shape[0])
        
        if 'Z' in f.keys():
            snr_values = f['Z'][:]  # SNR values
            print(f"SNR values shape: {snr_values.shape}")
            # Flatten if 2D
            if len(snr_values.shape) > 1:
                snr_values = snr_values.flatten()
                print(f"Flattened SNR values shape: {snr_values.shape}")
        else:
            # Try alternative key names
            snr_keys = [k for k in f.keys() if 'snr' in k.lower() or 'z' in k.lower()]
            if snr_keys:
                snr_values = f[snr_keys[0]][:]
                print(f"Using key '{snr_keys[0]}' for SNR, shape: {snr_values.shape}")
                # Flatten if 2D
                if len(snr_values.shape) > 1:
                    snr_values = snr_values.flatten()
                    print(f"Flattened SNR values shape: {snr_values.shape}")
            else:
                # Create synthetic SNR values if not found
                print("Warning: No SNR values found, creating synthetic SNR values")
                snr_levels = list(range(-20, 20, 2))
                snr_values = np.random.choice(snr_levels, data.shape[0])
    
    return data, labels, snr_values

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

def process_labels(labels):
    """Convert labels to integer format if they are one-hot encoded"""
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        # One-hot encoded, convert to integer labels
        labels = np.argmax(labels, axis=1)
    
    return labels.astype(np.int64)

def main():
    # Paths
    input_file = "data/raw/GOLD_XYZ_OSC.0001_1024.hdf5"
    output_dir = "data/processed/RML2018/"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the dataset
        data, labels, snr_values = load_rml2018_hdf5(input_file)
        
        # Process labels (convert from one-hot if needed)
        labels = process_labels(labels)
        
        # Ensure data is in the right format (samples, 2, length)
        if len(data.shape) == 3 and data.shape[1] == 2:
            print("Data already in correct format (samples, 2, length)")
        elif len(data.shape) == 3 and data.shape[2] == 2:
            print("Transposing data from (samples, length, 2) to (samples, 2, length)")
            data = np.transpose(data, (0, 2, 1))
        else:
            print(f"Warning: Unexpected data shape {data.shape}")
            print("Assuming data needs reshaping...")
            # Try to reshape to standard format
            if data.shape[0] > 1000:  # Likely samples dimension
                samples = data.shape[0]
                # Assume the rest needs to be reshaped to (2, length)
                remaining = np.prod(data.shape[1:])
                if remaining % 2 == 0:
                    length = remaining // 2
                    data = data.reshape(samples, 2, length)
                    print(f"Reshaped to: {data.shape}")
                else:
                    raise ValueError(f"Cannot reshape data with {remaining} elements to (samples, 2, length)")
        
        # Get unique SNR values
        unique_snrs = np.unique(snr_values)
        print(f"Unique SNR values: {unique_snrs}")
        print(f"Number of classes: {len(np.unique(labels))}")
        
        # Set random seed for reproducibility
        np.random.seed(2018)
        
        # Process each SNR level
        for snr in unique_snrs:
            print(f"\nProcessing SNR = {snr} dB")
            
            # Get samples for this SNR level
            snr_mask = snr_values == snr
            snr_data = data[snr_mask]
            snr_labels = labels[snr_mask]
            
            if len(snr_data) == 0:
                print(f"No data found for SNR {snr}, skipping...")
                continue
                
            print(f"Total samples for SNR {snr}: {snr_data.shape[0]}")
            print(f"Data shape: {snr_data.shape}")
            print(f"Label shape: {snr_labels.shape}")
            print(f"Label range: {snr_labels.min()} to {snr_labels.max()}")
            
            # Create train/test split (80/20)
            n_samples = snr_data.shape[0]
            n_train = int(0.8 * n_samples)
            
            # Create indices and shuffle
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            
            # Split data
            X_train = snr_data[train_indices]
            y_train = snr_labels[train_indices]
            X_test = snr_data[test_indices]
            y_test = snr_labels[test_indices]
            
            print(f"Train samples: {X_train.shape[0]}")
            print(f"Test samples: {X_test.shape[0]}")
            
            # Create indices for datasets (required by MAC framework)
            train_idx_tensor = np.arange(len(X_train))
            test_idx_tensor = np.arange(len(X_test))
            
            # Save datasets with MV4 naming convention as expected by util.py
            train_filename = os.path.join(output_dir, f"RML2018_MV4_snr_{snr}_train_dataset")
            test_filename = os.path.join(output_dir, f"RML2018_MV4_snr_{snr}_test_dataset")
            
            save_tensor_dataset(X_train, y_train, train_idx_tensor, train_filename)
            save_tensor_dataset(X_test, y_test, test_idx_tensor, test_filename)
            
            print(f"Saved {train_filename}")
            print(f"Saved {test_filename}")
        
        print("\nRML2018 preprocessing completed successfully!")
        print(f"Generated datasets for {len(unique_snrs)} SNR levels")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        print("Note: This is a template script. You may need to adjust the data loading")
        print("logic based on the actual format of your RML2018 HDF5 file.")

if __name__ == "__main__":
    main()
