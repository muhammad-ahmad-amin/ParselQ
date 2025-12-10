"""
Integration tests for ParselQ pipeline
Tests end-to-end functionality of models and utilities
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    set_seed, get_device, save_checkpoint, load_checkpoint,
    compute_metrics, read_jsonl, split_data, get_dataset_statistics,
    Config, get_baseline_b_config
)


class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility"""
        set_seed(42)
        rand1 = np.random.rand()
        
        set_seed(42)
        rand2 = np.random.rand()
        
        self.assertEqual(rand1, rand2, "Random seeds not working correctly")
    
    def test_get_device(self):
        """Test device detection"""
        device = get_device()
        self.assertIn(str(device), ['cuda', 'cpu', 'mps'], "Invalid device detected")
    
    def test_metrics_regression(self):
        """Test regression metrics computation"""
        y_true = np.array([[5.0, 6.0], [4.0, 7.0], [6.0, 5.0]])
        y_pred = np.array([[5.2, 5.8], [3.9, 7.2], [6.1, 4.9]])
        
        metrics = compute_metrics(y_true, y_pred, task_type='regression')
        
        self.assertIn('mse_overall', metrics)
        self.assertIn('mae_overall', metrics)
        self.assertIn('mse_valence', metrics)
        self.assertIn('mse_arousal', metrics)
        self.assertTrue(metrics['mse_overall'] > 0)
    
    def test_metrics_classification(self):
        """Test classification metrics computation"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2])
        
        metrics = compute_metrics(y_true, y_pred, task_type='classification')
        
        self.assertIn('accuracy', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)


class TestDataUtilities(unittest.TestCase):
    """Test data processing utilities"""
    
    def setUp(self):
        """Create temporary test data"""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_data.jsonl')
        
        # Create sample JSONL data
        test_data = [
            {
                "Text": "Climate change is a serious issue",
                "Aspect_VA": [{"Aspect": "climate_change", "VA": "3.0#7.0"}]
            },
            {
                "Text": "Renewable energy is the future",
                "Aspect_VA": [{"Aspect": "renewable_energy", "VA": "7.0#6.0"}]
            },
            {
                "Text": "Pollution is harmful",
                "Aspect_VA": [{"Aspect": "pollution", "VA": "2.0#8.0"}]
            }
        ]
        
        with open(self.test_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_read_jsonl(self):
        """Test JSONL file reading"""
        data = read_jsonl(self.test_file)
        
        self.assertEqual(len(data), 3, "Should read 3 samples")
        self.assertIn('Text', data[0])
        self.assertIn('Aspect_VA', data[0])
    
    def test_split_data(self):
        """Test data splitting"""
        data = read_jsonl(self.test_file)
        train, val = split_data(data, test_size=0.33, random_state=42)
        
        self.assertEqual(len(train) + len(val), len(data))
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation"""
        data = read_jsonl(self.test_file)
        stats = get_dataset_statistics(data)
        
        self.assertEqual(stats['total_samples'], 3)
        self.assertIn('avg_text_length', stats)
        self.assertIn('avg_valence', stats)
        self.assertIn('avg_arousal', stats)


class TestCheckpointing(unittest.TestCase):
    """Test model checkpointing"""
    
    def setUp(self):
        """Create temporary directory for checkpoints"""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.test_dir, 'checkpoint.pth')
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoints"""
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch=5, loss=0.5,
            filepath=self.checkpoint_path,
            extra_info="test"
        )
        
        self.assertTrue(os.path.exists(self.checkpoint_path))
        
        # Load checkpoint
        new_model = torch.nn.Linear(10, 2)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        checkpoint = load_checkpoint(
            self.checkpoint_path, new_model, new_optimizer
        )
        
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertEqual(checkpoint['loss'], 0.5)
        self.assertEqual(checkpoint['extra_info'], "test")


class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        """Create temporary directory"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_config_creation(self):
        """Test config creation"""
        config = Config(model_name="test_model", batch_size=32)
        
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.batch_size, 32)
    
    def test_config_save_load(self):
        """Test config saving and loading"""
        config = Config(model_name="test_model", epochs=20)
        config_path = os.path.join(self.test_dir, 'config.json')
        
        config.save(config_path)
        self.assertTrue(os.path.exists(config_path))
        
        loaded_config = Config.load(config_path)
        self.assertEqual(loaded_config.model_name, "test_model")
        self.assertEqual(loaded_config.epochs, 20)
    
    def test_baseline_configs(self):
        """Test baseline configuration factories"""
        config_b = get_baseline_b_config()
        
        self.assertEqual(config_b.model_name, "baseline_b_bilstm")
        self.assertEqual(config_b.model_type, "regression")
        self.assertIsInstance(config_b.batch_size, int)


class TestModelIntegration(unittest.TestCase):
    """Test model loading and basic operations"""
    
    def test_simple_model_forward(self):
        """Test simple model forward pass"""
        # Create a simple regression model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Test forward pass
        x = torch.randn(4, 10)
        output = model(x)
        
        self.assertEqual(output.shape, (4, 2))
    
    def test_model_training_step(self):
        """Test basic training step"""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        # Dummy data
        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete pipeline integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.test_dir, 'test.jsonl')
        
        # Create minimal test dataset
        test_samples = [
            {
                "Text": "Sample text " + str(i),
                "Aspect_VA": [{"Aspect": f"aspect_{i%3}", "VA": f"{5+i%3}.0#{5+i%2}.0"}]
            }
            for i in range(20)
        ]
        
        with open(self.data_file, 'w', encoding='utf-8') as f:
            for item in test_samples:
                f.write(json.dumps(item) + '\n')
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir)
    
    def test_data_loading_pipeline(self):
        """Test complete data loading pipeline"""
        # Load data
        data = read_jsonl(self.data_file)
        self.assertEqual(len(data), 20)
        
        # Split data
        train, val = split_data(data, test_size=0.2, random_state=42)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        
        # Get statistics
        stats = get_dataset_statistics(train)
        self.assertIn('total_samples', stats)
        
        print("\n[✓] End-to-end data pipeline test passed")


def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestDataUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointing))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running ParselQ Integration Tests...\n")
    success = run_integration_tests()
    sys.exit(0 if success else 1)