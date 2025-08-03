from __future__ import print_function

import pywt
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
import pickle
from torch.utils.data import Dataset
import random
import numpy as np
import os

def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data


def load_RML2016(args):
    """Load RML2016 dataset with CORRECTED 6:3:1 splits.
    Returns train, validation, and test data loaders.
    """
    print(f"Loading {args.ab_choose} dataset with SNR {args.snr_tat}...")
    
    # Map dataset choice to actual directory names
    if args.ab_choose == "RML201610A":
        dataset_dir = "RML2016.10a"
        base_path = args.RML2016a_path
    elif args.ab_choose == "RML201610B":
        dataset_dir = "RML2016.10b"
        base_path = args.RML2016b_path
    elif args.ab_choose == "RML2018":
        dataset_dir = "RML2018"
        base_path = args.RML2018_path
    else:
        raise ValueError(f"Unknown dataset: {args.ab_choose}")
    
    processed_path = os.path.join(base_path, "processed", dataset_dir)
    
    # Check if the corrected datasets directory exists
    if not os.path.exists(processed_path):
        print(f"ERROR: Corrected datasets not found at {processed_path}")
        print("Please run the preprocessing scripts first:")
        if args.ab_choose == "RML201610A":
            print("  python preprocess_rml2016a_correct.py")
        elif args.ab_choose == "RML201610B":
            print("  python preprocess_rml2016b_correct.py")
        elif args.ab_choose == "RML2018":
            print("  python preprocess_rml2018_memory_efficient.py")
        raise FileNotFoundError("Corrected datasets not found")
    
    # Handle SNR selection
    if args.snr_tat == "ALL":
        # Use ALL SNR datasets
        train_file = "train_ALL_SNR_MV_dataset"
        val_file = "val_ALL_SNR_MV_dataset"
        test_file = "test_ALL_SNR_MV_dataset"
        print("Using ALL SNR levels")
    else:
        # Use specific SNR level with corrected naming
        selected_snr = args.snr_tat
        train_file = f"{selected_snr}_train_MV_dataset"
        val_file = f"{selected_snr}_val_MV_dataset"
        test_file = f"{selected_snr}_test_MV_dataset"
        print(f"Using SNR {selected_snr} dB")
    
    # Load datasets
    datasets = {}
    for split, filename in [("train", train_file), ("val", val_file), ("test", test_file)]:
        filepath = os.path.join(processed_path, filename)
        if os.path.exists(filepath):
            print(f"Loading {split} data from {filepath}")
            with open(filepath, 'rb') as f:
                datasets[split] = pickle.load(f)
            print(f"  {split.capitalize()}: {len(datasets[split])} samples")
        else:
            print(f"ERROR: {split} dataset not found at {filepath}")
            raise FileNotFoundError(f"{split} dataset not found")
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    kwargs = {'num_workers': args.num_workers if hasattr(args, 'num_workers') else 4,
              'pin_memory': True} if torch.cuda.is_available() else {}
    
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
        drop_last=True
    )
    
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs,
        drop_last=False
    )
    
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs,
        drop_last=False
    )
    
    # Get data info
    n_data = len(datasets["train"])
    n_val_data = len(datasets["val"])
    
    # Create index for contrastive learning (using training data indices)
    indices = list(range(n_data))
    
    print(f"Successfully loaded {args.ab_choose} dataset:")
    print(f"  Train: {n_data} samples")
    print(f"  Validation: {n_val_data} samples") 
    print(f"  Test: {len(datasets['test'])} samples")
    print(f"  SNR: {args.snr_tat}")
    
    return train_loader, val_loader, test_loader, n_data, n_val_data, indices


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, opt, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        if opt.ab_choose == "SP_RML201610B":
            pred [pred == 2] = 12
            pred [pred > 2] -= 1
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    meter = AverageMeter()
