"""
Plotting utility functions for ParselQ project
Handles: training curves, confusion matrices, VA distributions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import pandas as pd


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_training_curves(train_losses: List[float], 
                         val_losses: Optional[List[float]] = None,
                         train_metrics: Optional[Dict[str, List[float]]] = None,
                         val_metrics: Optional[Dict[str, List[float]]] = None,
                         save_path: str = 'results/training_curves.pdf',
                         title: str = 'Training Progress'):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        train_metrics: Dict of metric_name -> list of values (optional)
        val_metrics: Dict of metric_name -> list of values (optional)
        save_path: Path to save plot
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Determine number of subplots needed
    num_plots = 1  # Loss plot
    if train_metrics:
        num_plots += len(train_metrics)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses and len(val_losses) > 0:
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot additional metrics
    if train_metrics:
        for idx, (metric_name, values) in enumerate(train_metrics.items(), 1):
            if idx < len(axes):
                ax = axes[idx]
                ax.plot(epochs, values, 'b-', label=f'Train {metric_name}', linewidth=2)
                
                if val_metrics and metric_name in val_metrics:
                    ax.plot(epochs, val_metrics[metric_name], 'r-', 
                           label=f'Val {metric_name}', linewidth=2)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} Progress')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Training curves saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         save_path: str = 'results/confusion_matrix.pdf',
                         title: str = 'Confusion Matrix',
                         normalize: bool = False):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Path to save plot
        title: Plot title
        normalize: Whether to normalize by row
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Confusion matrix saved to {save_path}")


def plot_va_distribution(valence: List[float], 
                        arousal: List[float],
                        save_path: str = 'results/va_distribution.pdf',
                        title: str = 'Valence-Arousal Distribution'):
    """
    Plot valence-arousal distribution (scatter + histograms)
    
    Args:
        valence: List of valence scores
        arousal: List of arousal scores
        save_path: Path to save plot
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1:, :-1])
    scatter = ax_main.scatter(valence, arousal, alpha=0.5, s=20, c=valence, cmap='coolwarm')
    ax_main.set_xlabel('Valence', fontsize=12)
    ax_main.set_ylabel('Arousal', fontsize=12)
    ax_main.set_xlim(0, 10)
    ax_main.set_ylim(0, 10)
    ax_main.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    ax_main.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    ax_main.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax_main, label='Valence')
    
    # Valence histogram (top)
    ax_valence = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_valence.hist(valence, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax_valence.set_ylabel('Frequency')
    ax_valence.set_title('Valence Distribution')
    ax_valence.grid(True, alpha=0.3)
    
    # Arousal histogram (right)
    ax_arousal = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_arousal.hist(arousal, bins=30, orientation='horizontal', 
                    color='salmon', edgecolor='black', alpha=0.7)
    ax_arousal.set_xlabel('Frequency')
    ax_arousal.set_title('Arousal Distribution')
    ax_arousal.grid(True, alpha=0.3)
    
    # Statistics box
    ax_stats = fig.add_subplot(gs[0, -1])
    ax_stats.axis('off')
    stats_text = (
        f"Statistics:\n"
        f"N = {len(valence)}\n\n"
        f"Valence:\n"
        f"  Mean: {np.mean(valence):.2f}\n"
        f"  Std: {np.std(valence):.2f}\n\n"
        f"Arousal:\n"
        f"  Mean: {np.mean(arousal):.2f}\n"
        f"  Std: {np.std(arousal):.2f}\n\n"
        f"Correlation:\n"
        f"  r = {np.corrcoef(valence, arousal)[0,1]:.3f}"
    )
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] VA distribution plot saved to {save_path}")


def plot_label_distribution(labels: List, 
                            save_path: str = 'results/label_distribution.pdf',
                            title: str = 'Label Distribution',
                            top_n: Optional[int] = 20):
    """
    Plot distribution of labels (for classification tasks)
    
    Args:
        labels: List of labels
        save_path: Path to save plot
        title: Plot title
        top_n: Show only top N labels (None for all)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Count labels
    label_counts = pd.Series(labels).value_counts()
    
    if top_n:
        label_counts = label_counts.head(top_n)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=label_counts.values, y=label_counts.index, palette='viridis')
    
    # Add count labels on bars
    for i, v in enumerate(label_counts.values):
        ax.text(v + 0.1, i, str(v), va='center')
    
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Label', fontsize=12)
    plt.title(title + (f' (Top {top_n})' if top_n else ''), fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Label distribution plot saved to {save_path}")


def plot_text_length_distribution(text_lengths: List[int],
                                  save_path: str = 'results/text_length_dist.pdf',
                                  title: str = 'Text Length Distribution'):
    """
    Plot distribution of text lengths
    
    Args:
        text_lengths: List of text lengths (word counts)
        save_path: Path to save plot
        title: Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add statistics
    mean_len = np.mean(text_lengths)
    median_len = np.median(text_lengths)
    
    plt.axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.1f}')
    plt.axvline(median_len, color='green', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
    
    plt.xlabel('Text Length (words)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Text length distribution saved to {save_path}")


def plot_comparison_bars(values: Dict[str, float],
                         save_path: str = 'results/comparison.pdf',
                         title: str = 'Model Comparison',
                         ylabel: str = 'Score'):
    """
    Plot comparison bar chart (e.g., comparing different models)
    
    Args:
        values: Dictionary of {name: value}
        save_path: Path to save plot
        title: Plot title
        ylabel: Y-axis label
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    names = list(values.keys())
    scores = list(values.values())
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = plt.bar(names, scores, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Comparison plot saved to {save_path}")


def save_all_plots(results_dict: Dict, output_dir: str = 'results'):
    """
    Save all plots from results dictionary
    
    Args:
        results_dict: Dictionary containing various results and metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Training curves
    if 'train_losses' in results_dict:
        plot_training_curves(
            results_dict['train_losses'],
            results_dict.get('val_losses'),
            save_path=os.path.join(output_dir, 'training_curves.pdf')
        )
    
    # Confusion matrix
    if 'confusion_matrix' in results_dict and 'class_names' in results_dict:
        plot_confusion_matrix(
            results_dict['confusion_matrix'],
            results_dict['class_names'],
            save_path=os.path.join(output_dir, 'confusion_matrix.pdf')
        )
    
    # VA distribution
    if 'valence' in results_dict and 'arousal' in results_dict:
        plot_va_distribution(
            results_dict['valence'],
            results_dict['arousal'],
            save_path=os.path.join(output_dir, 'va_distribution.pdf')
        )
    
    # Label distribution
    if 'labels' in results_dict:
        plot_label_distribution(
            results_dict['labels'],
            save_path=os.path.join(output_dir, 'label_distribution.pdf')
        )
    
    print(f"[✓] All plots saved to {output_dir}")


if __name__ == "__main__":
    # Test plotting utilities
    print("Testing plot utilities...")
    
    os.makedirs('test_plots', exist_ok=True)
    
    # Test training curves
    train_losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3]
    val_losses = [1.1, 0.9, 0.7, 0.6, 0.55, 0.5, 0.48]
    plot_training_curves(train_losses, val_losses, 
                        save_path='test_plots/training_curves.pdf')
    
    # Test confusion matrix
    cm = np.array([[50, 10, 5], [8, 60, 7], [3, 5, 52]])
    plot_confusion_matrix(cm, ['Class A', 'Class B', 'Class C'],
                        save_path='test_plots/confusion_matrix.pdf')
    
    # Test VA distribution
    np.random.seed(42)
    valence = np.random.normal(5, 2, 500).clip(1, 9)
    arousal = np.random.normal(5, 1.5, 500).clip(1, 9)
    plot_va_distribution(valence, arousal, save_path='test_plots/va_distribution.pdf')
    
    # Test label distribution
    labels = ['climate_change'] * 50 + ['renewable_energy'] * 40 + ['pollution'] * 30
    plot_label_distribution(labels, save_path='test_plots/label_distribution.pdf')
    
    print("\n[✓] All plot tests completed! Check test_plots/ folder")