#!/usr/bin/env python3
"""
ParselQ Pipeline Runner
Master script to run complete pipeline with all utilities integrated
"""

import os
import sys
import argparse
from datetime import datetime

# Import utilities
from utils import (
    set_seed, get_device, setup_logging,
    read_jsonl, print_dataset_info, split_data,
    get_baseline_a_config, get_baseline_b_config, get_baseline_d_config,
    plot_va_distribution, plot_label_distribution
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run ParselQ Pipeline')
    
    parser.add_argument('--model', type=str, default='baseline_b',
                    choices=['baseline_a', 'baseline_b', 'baseline_d', 'all'],
                    help='Which model to run')
    
    parser.add_argument('--train-path', type=str,
                    default='task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl',
                    help='Path to training data')
    
    parser.add_argument('--dev-path', type=str,
                    default='task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl',
                    help='Path to dev data')
    
    parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
    
    parser.add_argument('--test-only', action='store_true',
                    help='Run integration tests only')
    
    parser.add_argument('--eda-only', action='store_true',
                    help='Run EDA only (no training)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                    help='Output directory for results')
    
    return parser.parse_args()


def run_eda(train_path, dev_path, output_dir):
    """
    Run Exploratory Data Analysis
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    train_data = read_jsonl(train_path)
    dev_data = read_jsonl(dev_path)
    
    if len(train_data) == 0:
        print("[ERROR] No training data loaded!")
        return False
    
    # Print dataset info
    print_dataset_info(train_data, "Training Set")
    if len(dev_data) > 0:
        print_dataset_info(dev_data, "Development Set")
    
    # Extract VA scores for visualization
    valence_scores = []
    arousal_scores = []
    aspects = []
    
    for sample in train_data:
        for aspect_item in sample.get('Aspect_VA', []):
            va_str = aspect_item.get('VA', None)
            aspect = aspect_item.get('Aspect', '')
            
            if va_str:
                try:
                    val, aro = map(float, va_str.split('#'))
                    valence_scores.append(val)
                    arousal_scores.append(aro)
                    aspects.append(aspect)
                except:
                    pass
    
    # Plot VA distribution
    if valence_scores and arousal_scores:
        eda_dir = os.path.join(output_dir, 'eda')
        os.makedirs(eda_dir, exist_ok=True)
        
        plot_va_distribution(
            valence_scores, arousal_scores,
            save_path=os.path.join(eda_dir, 'va_distribution.pdf')
        )
    
    # Plot aspect distribution
    if aspects:
        plot_label_distribution(
            aspects,
            save_path=os.path.join(eda_dir, 'aspect_distribution.pdf'),
            title='Aspect Distribution'
        )
    
    print("\n[✓] EDA completed successfully!")
    return True


def run_baseline_a(config):
    """Run Baseline A (Logistic Regression)"""
    print("\n" + "="*60)
    print("RUNNING BASELINE A: Logistic Regression + TF-IDF")
    print("="*60)
    
    try:
        import model_baseline_A
        print("\n[✓] Baseline A completed!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Baseline A failed: {e}")
        return False


def run_baseline_b(config):
    """Run Baseline B (BiLSTM)"""
    print("\n" + "="*60)
    print("RUNNING BASELINE B: BiLSTM Regression")
    print("="*60)
    
    try:
        import model_baseline_B
        print("\n[✓] Baseline B completed!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Baseline B failed: {e}")
        return False


def run_baseline_d(config):
    """Run Baseline D (DistilBERT)"""
    print("\n" + "="*60)
    print("RUNNING BASELINE D: DistilBERT Regression")
    print("="*60)
    
    try:
        import model_baseline_D
        print("\n[✓] Baseline D completed!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Baseline D failed: {e}")
        return False


def run_integration_tests():
    """Run integration tests"""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)
    
    try:
        from tests.test_integration import run_integration_tests
        success = run_integration_tests()
        return success
    except Exception as e:
        print(f"\n[ERROR] Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main pipeline execution"""
    args = parse_args()
    
    # Setup
    print("\n" + "="*70)
    print(" "*20 + "ParselQ Pipeline Runner")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model}")
    print(f"Random seed: {args.seed}")
    print("="*70)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Setup logging
    log_file = os.path.join('logs', f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = setup_logging(log_file)
    
    # Run integration tests if requested
    if args.test_only:
        success = run_integration_tests()
        sys.exit(0 if success else 1)
    
    # Run EDA
    if args.eda_only or args.model == 'all':
        eda_success = run_eda(args.train_path, args.dev_path, args.output_dir)
        if args.eda_only:
            sys.exit(0 if eda_success else 1)
    
    # Get configurations
    configs = {
        'baseline_a': get_baseline_a_config(),
        'baseline_b': get_baseline_b_config(),
        'baseline_d': get_baseline_d_config()
    }
    
    # Run models
    results = {}
    
    if args.model == 'all':
        for model_name in ['baseline_a', 'baseline_b', 'baseline_d']:
            config = configs[model_name]
            config.train_path = args.train_path
            config.dev_path = args.dev_path
            
            print(f"\n\n{'='*70}")
            print(f"Running {model_name.upper()}")
            print(f"{'='*70}")
            
            if model_name == 'baseline_a':
                results[model_name] = run_baseline_a(config)
            elif model_name == 'baseline_b':
                results[model_name] = run_baseline_b(config)
            elif model_name == 'baseline_d':
                results[model_name] = run_baseline_d(config)
    else:
        config = configs[args.model]
        config.train_path = args.train_path
        config.dev_path = args.dev_path
        
        if args.model == 'baseline_a':
            results[args.model] = run_baseline_a(config)
        elif args.model == 'baseline_b':
            results[args.model] = run_baseline_b(config)
        elif args.model == 'baseline_d':
            results[args.model] = run_baseline_d(config)
    
    # Print summary
    print("\n" + "="*70)
    print(" "*25 + "PIPELINE SUMMARY")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results:
        print("\nModel Results:")
        for model_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {model_name:.<40} {status}")
    
    print("\nOutputs saved to:")
    print(f"  - Models: models/")
    print(f"  - Results: results/")
    print(f"  - Plots: plots/")
    print(f"  - Logs: {log_file}")
    print("="*70 + "\n")
    
    # Exit with appropriate code
    all_success = all(results.values()) if results else False
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()