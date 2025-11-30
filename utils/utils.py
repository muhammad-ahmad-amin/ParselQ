"""
Core utility functions for ParselQ project
Handles: checkpointing, metrics, logging, device management
"""

import os
import random
import logging
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    confusion_matrix
)


def set_seed(seed=42):
    """
    Fix random seeds for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[✓] Random seed set to {seed}")


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU)
    
    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[✓] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("[✓] Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("[✓] Using CPU")
    
    return device


def count_parameters(model):
    """
    Count trainable parameters in a model
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model has {total:,} trainable parameters")
    return total


def save_checkpoint(model, optimizer, epoch, loss, filepath, **kwargs):
    """
    Save model checkpoint with training state
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch (int): Current epoch
        loss (float): Current loss
        filepath (str): Path to save checkpoint
        **kwargs: Additional info to save (metrics, config, etc.)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    print(f"[✓] Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint and restore training state
    
    Args:
        filepath (str): Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optimizer (optional)
        device (str): Device to load model to
        
    Returns:
        dict: Checkpoint dictionary with epoch, loss, etc.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"[✓] Checkpoint loaded from: {filepath}")
    print(f"    - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"    - Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return checkpoint


def compute_metrics(y_true, y_pred, task_type='regression'):
    """
    Compute evaluation metrics based on task type
    
    Args:
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted labels (numpy array)
        task_type (str): 'regression' or 'classification'
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    if task_type == 'regression':
        # For valence-arousal regression
        if y_true.ndim == 2 and y_true.shape[1] == 2:
            # Separate metrics for valence and arousal
            metrics['mse_valence'] = mean_squared_error(y_true[:, 0], y_pred[:, 0])
            metrics['mse_arousal'] = mean_squared_error(y_true[:, 1], y_pred[:, 1])
            metrics['mae_valence'] = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
            metrics['mae_arousal'] = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
            
            # Overall metrics
            metrics['mse_overall'] = mean_squared_error(y_true, y_pred)
            metrics['mae_overall'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse_overall'] = np.sqrt(metrics['mse_overall'])
        else:
            # Single dimension regression
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
    
    elif task_type == 'classification':
        # For aspect classification
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Get unique labels
        labels = np.unique(np.concatenate([y_true, y_pred]))
        
        # Classification report as dict
        report = classification_report(
            y_true, y_pred, 
            labels=labels, 
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels)
    
    return metrics


def setup_logging(log_file=None, level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_file (str): Path to log file (optional)
        level: Logging level
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger


def print_training_info(epoch, total_epochs, train_loss, val_loss=None, 
                       train_metrics=None, val_metrics=None):
    """
    Print formatted training progress
    
    Args:
        epoch (int): Current epoch
        total_epochs (int): Total number of epochs
        train_loss (float): Training loss
        val_loss (float): Validation loss (optional)
        train_metrics (dict): Training metrics (optional)
        val_metrics (dict): Validation metrics (optional)
    """
    print(f"\n{'='*60}")
    print(f"Epoch [{epoch}/{total_epochs}]")
    print(f"{'='*60}")
    print(f"Train Loss: {train_loss:.4f}")
    
    if val_loss is not None:
        print(f"Val Loss:   {val_loss:.4f}")
    
    if train_metrics:
        print("\nTraining Metrics:")
        for key, val in train_metrics.items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.4f}")
    
    if val_metrics:
        print("\nValidation Metrics:")
        for key, val in val_metrics.items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.4f}")
    
    print(f"{'='*60}\n")


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    """
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' or 'max' - whether to minimize or maximize metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        """
        Check if training should stop
        
        Args:
            score (float): Current metric value
            epoch (int): Current epoch
            
        Returns:
            bool: True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"[!] Early stopping triggered at epoch {epoch}")
                print(f"    Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        
        return False


if __name__ == "__main__":
    # Test utilities
    print("Testing ParselQ utilities...")
    
    # Test seed setting
    set_seed(42)
    
    # Test device detection
    device = get_device()
    
    # Test metrics computation
    y_true = np.array([[5.0, 6.0], [4.0, 7.0], [6.0, 5.0]])
    y_pred = np.array([[5.2, 5.8], [3.9, 7.2], [6.1, 4.9]])
    metrics = compute_metrics(y_true, y_pred, task_type='regression')
    print("\nRegression Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3)
    for epoch in range(10):
        loss = 1.0 - epoch * 0.1 + (0.05 if epoch > 5 else 0)
        if early_stop(loss, epoch):
            break
        print(f"Epoch {epoch}: loss={loss:.4f}, counter={early_stop.counter}")
    
    print("\n[✓] All utility tests passed!")