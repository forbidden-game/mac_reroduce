#!/usr/bin/env python3
"""
Feature Visualization Tools for MAC Models
Extract and visualize learned features from trained MAC models
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score, classification_report
from sklearn.cluster import KMeans
import umap
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from util import load_RML2016
from models.backbone import MAC_backbone

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FeatureVisualizer:
    def __init__(self, output_dir="feature_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Modulation class names
        self.class_names = {
            'RML201610A': ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],
            'RML201610B': ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],
            'RML2018': [f'Class_{i}' for i in range(24)]  # Generic names for RML2018
        }
        
    def load_model(self, checkpoint_path, dataset_type, feat_dim=128):
        """Load trained MAC model from checkpoint"""
        print(f"Loading model from {checkpoint_path}")
        
        # Determine number of classes
        if dataset_type == 'RML201610A':
            num_classes = 11
        elif dataset_type == 'RML201610B':
            num_classes = 10
        elif dataset_type == 'RML2018':
            num_classes = 24
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Create model
        model = MAC_backbone(feat_dim, num_classes)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            
        print(f"Model loaded successfully for {dataset_type}")
        return model
    
    def extract_features(self, model, data_loader, dataset_type, max_samples=5000):
        """Extract features from trained model"""
        print("Extracting features from model...")
        
        features_l = []
        features_td = []
        features_td1 = []
        features_td2 = []
        features_td3 = []
        features_sd = []
        labels = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(tqdm(data_loader, desc="Extracting features")):
                if sample_count >= max_samples:
                    break
                    
                inputs = inputs.float()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                # Extract features from all domains
                feat_l, feat_TD, feat_TD1, feat_TD2, feat_TD3, feat_SD1 = model(inputs, 'AN', 'ALL', 'pretrain')
                
                # Move to CPU and store
                features_l.append(feat_l.cpu().numpy())
                features_td.append(feat_TD.cpu().numpy())
                features_td1.append(feat_TD1.cpu().numpy())
                features_td2.append(feat_TD2.cpu().numpy())
                features_td3.append(feat_TD3.cpu().numpy())
                features_sd.append(feat_SD1.cpu().numpy())
                labels.append(targets.numpy())
                
                sample_count += inputs.size(0)
        
        # Concatenate all features
        features = {
            'l_domain': np.vstack(features_l),
            'td_domain': np.vstack(features_td),
            'td1_domain': np.vstack(features_td1),
            'td2_domain': np.vstack(features_td2),
            'td3_domain': np.vstack(features_td3),
            'sd_domain': np.vstack(features_sd)
        }
        labels = np.concatenate(labels)
        
        print(f"Extracted features from {len(labels)} samples")
        print(f"Feature dimensions: {features['l_domain'].shape[1]}")
        
        return features, labels
    
    def compute_embeddings(self, features, labels, method='tsne', n_components=2):
        """Compute 2D embeddings using t-SNE, UMAP, or PCA"""
        print(f"Computing {method.upper()} embeddings...")
        
        embeddings = {}
        
        for domain_name, feat in features.items():
            print(f"Processing {domain_name}...")
            
            if method.lower() == 'tsne':
                embedding = TSNE(
                    n_components=n_components, 
                    perplexity=30, 
                    random_state=42,
                    n_iter=1000
                ).fit_transform(feat)
            elif method.lower() == 'umap':
                embedding = umap.UMAP(
                    n_components=n_components,
                    random_state=42,
                    min_dist=0.1,
                    n_neighbors=15
                ).fit_transform(feat)
            elif method.lower() == 'pca':
                embedding = PCA(
                    n_components=n_components,
                    random_state=42
                ).fit_transform(feat)
            else:
                raise ValueError(f"Unknown embedding method: {method}")
            
            embeddings[domain_name] = embedding
        
        return embeddings
    
    def plot_feature_embeddings(self, embeddings, labels, dataset_type, method='tsne'):
        """Plot 2D feature embeddings for all domains"""
        class_names = self.class_names[dataset_type]
        n_classes = len(class_names)
        
        # Create color palette
        colors = sns.color_palette("husl", n_classes)
        
        # Create subplot for each domain
        n_domains = len(embeddings)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (domain_name, embedding) in enumerate(embeddings.items()):
            ax = axes[idx]
            
            # Plot each class with different color
            for class_idx in range(n_classes):
                mask = labels == class_idx
                if np.sum(mask) > 0:
                    ax.scatter(
                        embedding[mask, 0], 
                        embedding[mask, 1],
                        c=[colors[class_idx]], 
                        label=class_names[class_idx],
                        alpha=0.6,
                        s=20
                    )
            
            ax.set_title(f'{domain_name.replace("_", " ").title()} Domain')
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Remove unused subplots
        for idx in range(n_domains, 6):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'{dataset_type} - {method.upper()} Feature Embeddings', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_type}_{method}_embeddings.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_class_separation(self, features, labels, dataset_type):
        """Analyze class separation using silhouette scores"""
        print("Analyzing class separation...")
        
        separation_scores = {}
        
        for domain_name, feat in features.items():
            # Compute silhouette score
            sil_score = silhouette_score(feat, labels)
            separation_scores[domain_name] = sil_score
            print(f"{domain_name}: Silhouette Score = {sil_score:.3f}")
        
        # Plot separation scores
        plt.figure(figsize=(10, 6))
        domains = list(separation_scores.keys())
        scores = list(separation_scores.values())
        
        bars = plt.bar(range(len(domains)), scores, color=sns.color_palette("viridis", len(domains)))
        plt.xlabel('Feature Domain')
        plt.ylabel('Silhouette Score')
        plt.title(f'{dataset_type} - Class Separation Analysis')
        plt.xticks(range(len(domains)), [d.replace('_', ' ').title() for d in domains], rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add score values on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_type}_separation_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return separation_scores
    
    def plot_confusion_matrix(self, model, data_loader, dataset_type):
        """Generate confusion matrix for classification performance"""
        print("Generating confusion matrix...")
        
        class_names = self.class_names[dataset_type]
        all_preds = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for inputs, targets, _ in tqdm(data_loader, desc="Computing predictions"):
                inputs = inputs.float()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                # Get predictions (using l_domain features for classification)
                feat_l, _, _, _, _, _ = model(inputs, 'AN', 'ALL', 'pretrain')
                
                # Simple classification using the features
                # Note: This is a simplified approach - in practice you'd need the classification head
                preds = torch.argmax(feat_l, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.numpy())
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, 
                   xticklabels=class_names,
                   yticklabels=class_names,
                   annot=True,
                   fmt='d',
                   cmap='Blues',
                   cbar_kws={'label': 'Number of Samples'})
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{dataset_type} - Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_type}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print classification report
        report = classification_report(all_labels, all_preds, 
                                     target_names=class_names,
                                     output_dict=True)
        
        # Save classification report
        with open(f'{self.output_dir}/{dataset_type}_classification_report.txt', 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(all_labels, all_preds, target_names=class_names))
            f.write(f"\n\nOverall Accuracy: {report['accuracy']:.3f}\n")
            f.write(f"Macro Average F1: {report['macro avg']['f1-score']:.3f}\n")
            f.write(f"Weighted Average F1: {report['weighted avg']['f1-score']:.3f}\n")
        
        return cm, report
    
    def analyze_feature_evolution(self, checkpoint_dir, dataset_type, args):
        """Analyze how features evolve during training"""
        print("Analyzing feature evolution across epochs...")
        
        # Find all checkpoint files
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith('ckpt_epoch_') and file.endswith('.pth'):
                epoch = int(file.split('_')[2].split('.')[0])
                checkpoint_files.append((epoch, os.path.join(checkpoint_dir, file)))
        
        checkpoint_files.sort()
        
        if len(checkpoint_files) < 2:
            print("Not enough checkpoints found for evolution analysis")
            return
        
        # Prepare data loader
        args.ab_choose = dataset_type
        train_loader, _, _, _ = load_RML2016(args)
        
        evolution_data = []
        
        for epoch, checkpoint_path in checkpoint_files[:5]:  # Analyze first 5 checkpoints
            print(f"Processing epoch {epoch}...")
            
            # Load model
            model = self.load_model(checkpoint_path, dataset_type)
            
            # Extract features
            features, labels = self.extract_features(model, train_loader, dataset_type, max_samples=1000)
            
            # Compute separation score for l_domain
            sil_score = silhouette_score(features['l_domain'], labels)
            
            evolution_data.append({
                'epoch': epoch,
                'silhouette_score': sil_score,
                'features': features['l_domain'][:100],  # Store small subset for visualization
                'labels': labels[:100]
            })
        
        # Plot evolution of separation scores
        epochs = [d['epoch'] for d in evolution_data]
        scores = [d['silhouette_score'] for d in evolution_data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, scores, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Epoch')
        plt.ylabel('Silhouette Score')
        plt.title(f'{dataset_type} - Feature Separation Evolution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_type}_feature_evolution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return evolution_data
    
    def generate_comprehensive_report(self, features, labels, embeddings, separation_scores, dataset_type):
        """Generate comprehensive feature analysis report"""
        print("Generating comprehensive analysis report...")
        
        class_names = self.class_names[dataset_type]
        
        # Create feature statistics
        feature_stats = {}
        for domain_name, feat in features.items():
            feature_stats[domain_name] = {
                'mean': np.mean(feat, axis=0),
                'std': np.std(feat, axis=0),
                'min': np.min(feat, axis=0),
                'max': np.max(feat, axis=0)
            }
        
        # Plot feature statistics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature means
        ax = axes[0, 0]
        for domain_name, stats in feature_stats.items():
            ax.plot(stats['mean'], label=domain_name.replace('_', ' ').title(), alpha=0.7)
        ax.set_title('Feature Means Across Domains')
        ax.set_xlabel('Feature Dimension')
        ax.set_ylabel('Mean Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Feature standard deviations
        ax = axes[0, 1]
        for domain_name, stats in feature_stats.items():
            ax.plot(stats['std'], label=domain_name.replace('_', ' ').title(), alpha=0.7)
        ax.set_title('Feature Standard Deviations')
        ax.set_xlabel('Feature Dimension')
        ax.set_ylabel('Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Separation scores bar plot
        ax = axes[1, 0]
        domains = list(separation_scores.keys())
        scores = list(separation_scores.values())
        bars = ax.bar(range(len(domains)), scores, color=sns.color_palette("viridis", len(domains)))
        ax.set_title('Class Separation Scores')
        ax.set_xlabel('Feature Domain')
        ax.set_ylabel('Silhouette Score')
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels([d.replace('_', ' ').title() for d in domains], rotation=45)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        # Class distribution
        ax = axes[1, 1]
        unique_labels, counts = np.unique(labels, return_counts=True)
        bars = ax.bar(unique_labels, counts, color=sns.color_palette("husl", len(unique_labels)))
        ax.set_title('Class Distribution in Features')
        ax.set_xlabel('Class Index')
        ax.set_ylabel('Number of Samples')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_type}_comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Write detailed report
        with open(f'{self.output_dir}/{dataset_type}_feature_analysis_report.txt', 'w') as f:
            f.write(f"Feature Analysis Report for {dataset_type}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Dataset: {dataset_type}\n")
            f.write(f"Number of samples analyzed: {len(labels)}\n")
            f.write(f"Number of classes: {len(class_names)}\n")
            f.write(f"Feature dimensions: {features['l_domain'].shape[1]}\n\n")
            
            f.write("Class Names:\n")
            for i, name in enumerate(class_names):
                f.write(f"  {i}: {name}\n")
            f.write("\n")
            
            f.write("Separation Scores by Domain:\n")
            for domain, score in separation_scores.items():
                f.write(f"  {domain.replace('_', ' ').title()}: {score:.4f}\n")
            f.write("\n")
            
            f.write("Feature Statistics Summary:\n")
            for domain_name, stats in feature_stats.items():
                f.write(f"\n{domain_name.replace('_', ' ').title()} Domain:\n")
                f.write(f"  Mean range: {np.min(stats['mean']):.4f} to {np.max(stats['mean']):.4f}\n")
                f.write(f"  Std range: {np.min(stats['std']):.4f} to {np.max(stats['std']):.4f}\n")
                f.write(f"  Overall range: {np.min(stats['min']):.4f} to {np.max(stats['max']):.4f}\n")
        
        print(f"Analysis complete! Results saved to '{self.output_dir}/' directory")

def create_args_for_dataset(dataset_type, snr_tat=6):
    """Create args object for loading dataset"""
    class Args:
        def __init__(self):
            self.ab_choose = dataset_type
            self.snr_tat = snr_tat
            self.N_shot = 0  # Use full dataset
            self.batch_size = 64
            self.threads = 4
            
    return Args()

def main():
    parser = argparse.ArgumentParser(description='Visualize features from trained MAC models')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', required=True,
                       choices=['RML201610A', 'RML201610B', 'RML2018'],
                       help='Dataset type')
    parser.add_argument('--snr', type=int, default=6,
                       help='SNR level to analyze')
    parser.add_argument('--output-dir', default='feature_analysis',
                       help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='Maximum number of samples to analyze')
    parser.add_argument('--embedding-method', default='tsne',
                       choices=['tsne', 'umap', 'pca'],
                       help='Embedding method for visualization')
    parser.add_argument('--evolution-dir', 
                       help='Directory containing multiple checkpoints for evolution analysis')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = FeatureVisualizer(args.output_dir)
    
    # Create dataset args
    dataset_args = create_args_for_dataset(args.dataset, args.snr)
    
    # Load data
    print("Loading dataset...")
    train_loader, test_loader, _, _ = load_RML2016(dataset_args)
    
    # Load model
    model = visualizer.load_model(args.checkpoint, args.dataset)
    
    # Extract features
    features, labels = visualizer.extract_features(model, test_loader, args.dataset, args.max_samples)
    
    # Compute embeddings
    embeddings = visualizer.compute_embeddings(features, labels, args.embedding_method)
    
    # Generate visualizations
    visualizer.plot_feature_embeddings(embeddings, labels, args.dataset, args.embedding_method)
    
    # Analyze class separation
    separation_scores = visualizer.analyze_class_separation(features, labels, args.dataset)
    
    # Generate confusion matrix
    try:
        cm, report = visualizer.plot_confusion_matrix(model, test_loader, args.dataset)
    except:
        print("Note: Confusion matrix generation requires classification head - skipping")
    
    # Evolution analysis if directory provided
    if args.evolution_dir:
        try:
            evolution_data = visualizer.analyze_feature_evolution(args.evolution_dir, args.dataset, dataset_args)
        except Exception as e:
            print(f"Evolution analysis failed: {e}")
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report(features, labels, embeddings, separation_scores, args.dataset)

if __name__ == "__main__":
    main()
