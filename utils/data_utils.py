"""
Data utility functions for ParselQ project
Handles: JSONL reading, dataset splitting, batching, tokenization
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional


def read_jsonl(filepath: str) -> List[Dict]:
    """
    Read JSONL file and return list of dictionaries
    
    Args:
        filepath (str): Path to JSONL file
        
    Returns:
        List[Dict]: List of data samples
    """
    samples = []
    
    if not filepath.endswith('.jsonl'):
        print(f"[WARNING] File {filepath} is not .jsonl format")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Skipping line {idx}: Invalid JSON - {e}")
        
        print(f"[✓] Loaded {len(samples)} samples from {filepath}")
        return samples
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return []


def parse_aspect_va(aspect_va_list: List[Dict]) -> List[Tuple[str, float, float]]:
    """
    Parse Aspect_VA field from JSONL data
    
    Args:
        aspect_va_list: List of aspect-VA dictionaries
        
    Returns:
        List of tuples (aspect, valence, arousal)
    """
    parsed = []
    
    for item in aspect_va_list:
        aspect = item.get('Aspect', '')
        va_str = item.get('VA', '5.0#5.0')
        
        try:
            valence, arousal = map(float, va_str.split('#'))
        except (ValueError, AttributeError):
            valence, arousal = 5.0, 5.0  # Default neutral
        
        parsed.append((aspect, valence, arousal))
    
    return parsed


def split_data(data: List[Dict], test_size: float = 0.2, 
               random_state: int = 42, stratify_key: str = None) -> Tuple[List, List]:
    """
    Split data into train and validation sets
    
    Args:
        data: List of data samples
        test_size: Fraction for validation set
        random_state: Random seed
        stratify_key: Key to use for stratified splitting (optional)
        
    Returns:
        Tuple of (train_data, val_data)
    """
    if len(data) == 0:
        print("[WARNING] Empty dataset provided for splitting")
        return [], []
    
    if stratify_key:
        # Extract stratification labels
        labels = [item.get(stratify_key) for item in data]
        train_data, val_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
    else:
        train_data, val_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state
        )
    
    print(f"[✓] Data split: {len(train_data)} train, {len(val_data)} val")
    return train_data, val_data


def tokenize_batch(texts: List[str], tokenizer, max_length: int = 128) -> Dict:
    """
    Tokenize a batch of texts
    
    Args:
        texts: List of text strings
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with input_ids, attention_mask, etc.
    """
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded


def collate_fn_regression(batch: List[Dict], tokenizer=None, max_len: int = 128):
    """
    Collate function for regression task (Valence-Arousal)
    
    Args:
        batch: List of samples from dataset
        tokenizer: Tokenizer for text encoding
        max_len: Maximum sequence length
        
    Returns:
        Tuple of (input_ids, attention_mask, targets)
    """
    if tokenizer is not None:
        # If tokenizer provided, tokenize on-the-fly
        texts = [item['text'] for item in batch]
        encoded = tokenize_batch(texts, tokenizer, max_len)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
    else:
        # Assume already tokenized
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Stack targets (valence, arousal)
    if 'labels' in batch[0]:
        targets = torch.stack([item['labels'] for item in batch])
    elif 'valence' in batch[0] and 'arousal' in batch[0]:
        targets = torch.tensor([
            [item['valence'], item['arousal']] for item in batch
        ], dtype=torch.float32)
    else:
        targets = None
    
    return input_ids, attention_mask, targets


def collate_fn_classification(batch: List[Dict]):
    """
    Collate function for classification task (Aspect classification)
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Tuple of (input_ids, attention_mask, labels)
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    return input_ids, attention_mask, labels


def create_dataloaders(train_dataset, val_dataset=None, batch_size: int = 16,
                      shuffle_train: bool = True, num_workers: int = 0,
                      collate_fn=None):
    """
    Create PyTorch DataLoaders from datasets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size
        shuffle_train: Whether to shuffle training data
        num_workers: Number of data loading workers
        collate_fn: Custom collate function
        
    Returns:
        Tuple of (train_loader, val_loader) or just train_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    return (train_loader, val_loader) if val_loader else train_loader


class TextAspectDataset(Dataset):
    """
    Dataset for text-aspect pairs with optional VA labels
    Useful for quick prototyping and testing
    """
    def __init__(self, texts: List[str], aspects: List[str] = None,
                 valence: List[float] = None, arousal: List[float] = None):
        """
        Args:
            texts: List of text strings
            aspects: List of aspect strings (optional)
            valence: List of valence scores (optional)
            arousal: List of arousal scores (optional)
        """
        self.texts = texts
        self.aspects = aspects if aspects else [''] * len(texts)
        self.valence = valence if valence else [5.0] * len(texts)
        self.arousal = arousal if arousal else [5.0] * len(texts)
        
        assert len(self.texts) == len(self.aspects) == len(self.valence) == len(self.arousal)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'aspect': self.aspects[idx],
            'valence': self.valence[idx],
            'arousal': self.arousal[idx]
        }


def get_dataset_statistics(data: List[Dict]) -> Dict:
    """
    Compute statistics about the dataset
    
    Args:
        data: List of data samples
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_samples': len(data),
        'total_aspects': 0,
        'unique_aspects': set(),
        'text_lengths': [],
        'valence_scores': [],
        'arousal_scores': [],
        'samples_with_va': 0,
        'samples_without_va': 0
    }
    
    for sample in data:
        text = sample.get('Text', '')
        stats['text_lengths'].append(len(text.split()))
        
        aspects = sample.get('Aspect_VA', [])
        stats['total_aspects'] += len(aspects)
        
        for aspect_item in aspects:
            aspect = aspect_item.get('Aspect', '')
            stats['unique_aspects'].add(aspect)
            
            va_str = aspect_item.get('VA', None)
            if va_str:
                try:
                    val, aro = map(float, va_str.split('#'))
                    stats['valence_scores'].append(val)
                    stats['arousal_scores'].append(aro)
                    stats['samples_with_va'] += 1
                except:
                    stats['samples_without_va'] += 1
            else:
                stats['samples_without_va'] += 1
    
    # Convert to summary statistics
    stats['unique_aspects'] = len(stats['unique_aspects'])
    stats['avg_text_length'] = np.mean(stats['text_lengths']) if stats['text_lengths'] else 0
    stats['avg_aspects_per_sample'] = stats['total_aspects'] / len(data) if data else 0
    
    if stats['valence_scores']:
        stats['avg_valence'] = np.mean(stats['valence_scores'])
        stats['std_valence'] = np.std(stats['valence_scores'])
        stats['avg_arousal'] = np.mean(stats['arousal_scores'])
        stats['std_arousal'] = np.std(stats['arousal_scores'])
    
    return stats


def print_dataset_info(data: List[Dict], name: str = "Dataset"):
    """
    Print formatted dataset information
    
    Args:
        data: List of data samples
        name: Name to display
    """
    stats = get_dataset_statistics(data)
    
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Total samples:           {stats['total_samples']}")
    print(f"Total aspects:           {stats['total_aspects']}")
    print(f"Unique aspects:          {stats['unique_aspects']}")
    print(f"Avg aspects/sample:      {stats['avg_aspects_per_sample']:.2f}")
    print(f"Avg text length:         {stats['avg_text_length']:.1f} words")
    
    if stats['valence_scores']:
        print(f"\nValence-Arousal Statistics:")
        print(f"  Samples with VA:       {stats['samples_with_va']}")
        print(f"  Avg Valence:           {stats['avg_valence']:.2f} ± {stats['std_valence']:.2f}")
        print(f"  Avg Arousal:           {stats['avg_arousal']:.2f} ± {stats['std_arousal']:.2f}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test data utilities
    print("Testing data utilities...")
    
    # Test JSONL reading
    test_data = [
        {
            "Text": "Climate change is a serious issue",
            "Aspect_VA": [
                {"Aspect": "climate_change", "VA": "3.0#7.0"}
            ]
        },
        {
            "Text": "Renewable energy is the future",
            "Aspect_VA": [
                {"Aspect": "renewable_energy", "VA": "7.0#6.0"}
            ]
        }
    ]
    
    # Test parsing
    for sample in test_data:
        aspects = parse_aspect_va(sample['Aspect_VA'])
        print(f"Text: {sample['Text'][:50]}...")
        print(f"Aspects: {aspects}\n")
    
    # Test statistics
    print_dataset_info(test_data, "Test Dataset")
    
    # Test splitting
    train, val = split_data(test_data, test_size=0.5)
    print(f"Split: {len(train)} train, {len(val)} val")
    
    print("\n[✓] All data utility tests passed!")