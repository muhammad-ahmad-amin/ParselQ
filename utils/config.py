"""
Configuration management for ParselQ project
Centralized configuration for all baseline models
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
import os


@dataclass
class Config:
    """Base configuration class for all models"""
    
    # Model parameters
    model_name: str = "baseline"
    model_type: str = "classification"  # or 'regression'
    
    # Data paths
    train_path: str = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
    dev_path: str = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"
    test_path: Optional[str] = None
    
    # Training parameters
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adam"  # 'adam', 'adamw', 'sgd'
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "plateau"  # 'plateau', 'linear', 'cosine'
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    warmup_steps: int = 0
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.0001
    
    # Text processing
    max_length: int = 128
    remove_stopwords: bool = False
    lowercase: bool = True
    
    # Model architecture (for neural models)
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    
    # Output directories
    output_dir: str = "models"
    results_dir: str = "results"
    plots_dir: str = "plots"
    logs_dir: str = "logs"
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "auto"  # 'auto', 'cuda', 'cpu', 'mps'
    
    # Logging
    log_interval: int = 10
    save_interval: int = 1
    verbose: bool = True
    
    def __post_init__(self):
        """Create directories after initialization"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath: str):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[✓] Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __str__(self):
        """Pretty print config"""
        lines = ["\n" + "="*60, "Configuration", "="*60]
        for key, value in self.to_dict().items():
            lines.append(f"  {key:.<30} {value}")
        lines.append("="*60 + "\n")
        return "\n".join(lines)


def get_baseline_a_config() -> Config:
    """
    Configuration for Baseline A (Logistic Regression + TF-IDF)
    """
    config = Config(
        model_name="baseline_a_logreg",
        model_type="classification",
        
        # No neural training parameters needed
        batch_size=None,
        epochs=None,
        learning_rate=None,
        
        # TF-IDF parameters (handled in model)
        max_length=None,  # Not used for TF-IDF
        
        # Text preprocessing
        remove_stopwords=True,
        lowercase=True,
        
        # Logistic Regression (handled by sklearn)
        # max_iter set in model_baseline_A.py
        
        output_dir="models",
        results_dir="results",
        plots_dir="plots",
        seed=42
    )
    return config


def get_baseline_b_config() -> Config:
    """
    Configuration for Baseline B (BiLSTM Regression)
    """
    config = Config(
        model_name="baseline_b_bilstm",
        model_type="regression",
        
        # Training
        batch_size=16,
        epochs=10,
        learning_rate=1e-3,
        weight_decay=0.0,
        
        # Architecture
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        
        # Text
        max_length=128,
        
        # Optimizer & Scheduler
        optimizer="adam",
        use_scheduler=True,
        scheduler_type="plateau",
        scheduler_patience=2,
        scheduler_factor=0.5,
        
        # Early stopping
        use_early_stopping=True,
        patience=3,
        
        output_dir="models",
        results_dir="results",
        seed=42
    )
    return config


def get_baseline_d_config() -> Config:
    """
    Configuration for Baseline D (DistilBERT Regression)
    """
    config = Config(
        model_name="baseline_d_distilbert",
        model_type="regression",
        
        # Training
        batch_size=16,
        epochs=6,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Architecture
        dropout=0.2,
        
        # Text
        max_length=128,
        
        # Optimizer & Scheduler
        optimizer="adamw",
        use_scheduler=True,
        scheduler_type="linear",
        warmup_steps=0,
        
        # Early stopping
        use_early_stopping=False,  # DistilBERT trains for fixed epochs
        
        output_dir="models",
        results_dir="results",
        seed=42
    )
    return config


def get_fine_tuned_config() -> Config:
    """
    Configuration for fine-tuned BERT model
    """
    config = Config(
        model_name="fine_tuned_bert",
        model_type="regression",
        
        # Training
        batch_size=8,
        epochs=5,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Architecture
        dropout=0.1,
        
        # Text
        max_length=256,
        
        # Optimizer & Scheduler
        optimizer="adamw",
        use_scheduler=True,
        scheduler_type="linear",
        warmup_steps=100,
        
        # Early stopping
        use_early_stopping=True,
        patience=2,
        
        output_dir="models",
        results_dir="results",
        seed=42
    )
    return config


if __name__ == "__main__":
    # Test configurations
    print("Testing configuration management...")
    
    # Test Baseline A
    config_a = get_baseline_a_config()
    print(config_a)
    config_a.save("test_config_a.json")
    
    # Test Baseline B
    config_b = get_baseline_b_config()
    print(config_b)
    
    # Test Baseline D
    config_d = get_baseline_d_config()
    print(config_d)
    
    # Test loading
    loaded_config = Config.load("test_config_a.json")
    print("\nLoaded config:")
    print(loaded_config)
    
    # Cleanup
    os.remove("test_config_a.json")
    
    print("\n[✓] All configuration tests passed!")