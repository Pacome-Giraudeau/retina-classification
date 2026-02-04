# test/tests_unitaires.py
import os
import sys
import pytest
import tempfile
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from config_manager import ConfigManager
from train import (
    get_class_weights,
    FocalLoss,
    ClassBalancedBCELoss,
    RetinaMultiLabelDataset,
    build_model,
    train_epoch,
    validate_epoch
)
from evaluate import ModelEvaluator


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample config.yaml file"""
    config = {
        'paths': {
            'train_csv': os.path.join(temp_dir, 'train.csv'),
            'val_csv': os.path.join(temp_dir, 'val.csv'),
            'test_csv': os.path.join(temp_dir, 'test.csv'),
            'train_image_dir': os.path.join(temp_dir, 'train_images'),
            'val_image_dir': os.path.join(temp_dir, 'val_images'),
            'test_image_dir': os.path.join(temp_dir, 'test_images'),
            'pretrained_backbone_efficient': 'efficientnet_weights.pt',
            'pretrained_backbone_resnet18': 'resnet18_weights.pt',
            'output_model_dir': os.path.join(temp_dir, 'models'),
            'output_dir': os.path.join(temp_dir, 'results')
        },
        'model': {
            'backbone': 'resnet18',
            'task': 't1.1',
            'num_classes': 3
        },
        'hyperparameters': {
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 0.001,
            'img_size': 64,
            'weight_decay': 0.0
        },
        'grid_search': {
            'epochs': [2],
            'batch_size': [4],
            'learning_rate': [0.001],
            'img_size': [64],
            'weight_decay': [0.0]
        },
        'run_mode': 'single',
        'output': {
            'save_checkpoints': True,
            'save_probabilities': True,
            'save_metrics': True
        },
        'device': {
            'use_cuda': False
        }
    }
    
    config_path = os.path.join(temp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def sample_csv_data(temp_dir):
    """Create sample CSV files for training/validation/testing"""
    # Create image directories
    for dir_name in ['train_images', 'val_images', 'test_images']:
        os.makedirs(os.path.join(temp_dir, dir_name), exist_ok=True)
    
    # Create sample images
    def create_sample_image(path):
        img = Image.new('RGB', (64, 64), color='red')
        img.save(path)
    
    # Create train data
    train_data = {
        'id': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
        'D': [1, 0, 1, 0],
        'G': [0, 1, 0, 1],
        'A': [0, 0, 1, 1]
    }
    train_df = pd.DataFrame(train_data)
    train_csv_path = os.path.join(temp_dir, 'train.csv')
    train_df.to_csv(train_csv_path, index=False)
    
    for img_name in train_data['id']:
        create_sample_image(os.path.join(temp_dir, 'train_images', img_name))
    
    # Create val data
    val_data = {
        'id': ['img5.jpg', 'img6.jpg'],
        'D': [1, 0],
        'G': [0, 1],
        'A': [1, 0]
    }
    val_df = pd.DataFrame(val_data)
    val_csv_path = os.path.join(temp_dir, 'val.csv')
    val_df.to_csv(val_csv_path, index=False)
    
    for img_name in val_data['id']:
        create_sample_image(os.path.join(temp_dir, 'val_images', img_name))
    
    # Create test data
    test_data = {
        'id': ['img7.jpg', 'img8.jpg'],
        'D': [0, 1],
        'G': [1, 0],
        'A': [0, 1]
    }
    test_df = pd.DataFrame(test_data)
    test_csv_path = os.path.join(temp_dir, 'test.csv')
    test_df.to_csv(test_csv_path, index=False)
    
    for img_name in test_data['id']:
        create_sample_image(os.path.join(temp_dir, 'test_images', img_name))
    
    return train_csv_path, val_csv_path, test_csv_path


# ============================================================================
# CONFIG_MANAGER TESTS
# ============================================================================

class TestConfigManager:
    """Test suite for ConfigManager class"""
    
    def test_init_with_valid_config(self, sample_config):
        """Test ConfigManager initialization with valid config"""
        cm = ConfigManager(sample_config)
        assert cm.config is not None
        assert isinstance(cm.config, dict)
    
    # def test_init_with_missing_config(self):
    #     """Test ConfigManager raises error for missing config"""
    #     with pytest.raises(FileNotFoundError):
    #         ConfigManager('nonexistent_config.yaml')
    
    def test_get_paths(self, sample_config, temp_dir):
        """Test get_paths returns correct paths"""
        cm = ConfigManager(sample_config)
        paths = cm.get_paths()
        
        assert 'train_csv' in paths
        assert 'val_csv' in paths
        assert 'test_csv' in paths
        assert 'output_model_dir' in paths
        assert paths['train_csv'] == os.path.join(temp_dir, 'train.csv')
    
    def test_get_model_config(self, sample_config):
        """Test get_model_config returns correct model settings"""
        cm = ConfigManager(sample_config)
        model_config = cm.get_model_config()
        
        assert model_config['backbone'] == 'resnet18'
        assert model_config['task'] == 't1.1'
        assert model_config['num_classes'] == 3
    
    def test_get_training_config(self, sample_config):
        """Test get_training_config returns correct hyperparameters"""
        cm = ConfigManager(sample_config)
        training_config = cm.get_training_config()
        
        assert training_config['epochs'] == 2
        assert training_config['batch_size'] == 4
        assert training_config['learning_rate'] == 0.001
        assert training_config['img_size'] == 64
    
    def test_get_grid_search_config(self, sample_config):
        """Test get_grid_search_config returns grid parameters"""
        cm = ConfigManager(sample_config)
        grid_config = cm.get_grid_search_config()
        
        assert isinstance(grid_config['epochs'], list)
        assert isinstance(grid_config['batch_size'], list)
        assert 2 in grid_config['epochs']
    
    def test_get_run_mode(self, sample_config):
        """Test get_run_mode returns correct mode"""
        cm = ConfigManager(sample_config)
        run_mode = cm.get_run_mode()
        
        assert run_mode == 'single'
    
    def test_get_output_config(self, sample_config):
        """Test get_output_config returns output settings"""
        cm = ConfigManager(sample_config)
        output_config = cm.get_output_config()
        
        assert output_config['save_checkpoints'] is True
        assert output_config['save_probabilities'] is True
        assert output_config['save_metrics'] is True
    
    def test_get_device_config(self, sample_config):
        """Test get_device_config returns correct device"""
        cm = ConfigManager(sample_config)
        device_config = cm.get_device_config()
        
        assert 'device' in device_config
        assert device_config['device'] in ['cpu', 'cuda']
    
    def test_create_directories(self, sample_config, temp_dir):
        """Test create_directories creates necessary folders"""
        cm = ConfigManager(sample_config)
        paths = cm.create_directories()
        
        assert os.path.exists(paths['output_model_dir'])
        assert os.path.exists(paths['output_dir'])


# ============================================================================
# TRAIN.PY TESTS
# ============================================================================

class TestTrainFunctions:
    """Test suite for training functions"""
    
    def test_get_class_weights(self, sample_csv_data, temp_dir):
        """Test get_class_weights calculates correct weights"""
        train_csv, _, _ = sample_csv_data
        weights = get_class_weights(train_csv)
        
        assert isinstance(weights, torch.Tensor)
        assert weights.shape[0] == 3  # 3 classes
        assert torch.all(weights > 0)
    
    def test_focal_loss_forward(self):
        """Test FocalLoss forward pass"""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()
        
        loss = criterion(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert loss.item() >= 0
    
    def test_class_balanced_bce_loss(self):
        """Test ClassBalancedBCELoss forward pass"""
        class_weights = torch.tensor([1.0, 1.5, 2.0])
        criterion = ClassBalancedBCELoss(class_weights)
        
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()
        
        loss = criterion(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_build_model_resnet18(self):
        """Test build_model creates ResNet18 correctly"""
        model = build_model(backbone='resnet18', num_classes=3, pretrained=False)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'fc')
        assert model.fc.out_features == 3
    
    def test_build_model_efficientnet(self):
        """Test build_model creates EfficientNet correctly"""
        model = build_model(backbone='efficientnet', num_classes=3, pretrained=False)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'classifier')
        assert model.classifier[1].out_features == 3
    
    def test_build_model_invalid_backbone(self):
        """Test build_model raises error for invalid backbone"""
        with pytest.raises(ValueError):
            build_model(backbone='invalid_model', num_classes=3)
    
    def test_retina_dataset_length(self, sample_csv_data, temp_dir):
        """Test RetinaMultiLabelDataset returns correct length"""
        train_csv, _, _ = sample_csv_data
        train_dir = os.path.join(temp_dir, 'train_images')
        
        dataset = RetinaMultiLabelDataset(train_csv, train_dir, transform=None)
        
        assert len(dataset) == 4
    
    def test_retina_dataset_getitem(self, sample_csv_data, temp_dir):
        """Test RetinaMultiLabelDataset returns correct item"""
        train_csv, _, _ = sample_csv_data
        train_dir = os.path.join(temp_dir, 'train_images')
        
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        dataset = RetinaMultiLabelDataset(train_csv, train_dir, transform)
        img, labels = dataset[0]
        
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (3,)
    
    def test_train_epoch(self, sample_csv_data, sample_config, temp_dir):
        """Test train_epoch executes without errors"""
        from torch.utils.data import DataLoader
        from torchvision import transforms
        
        train_csv, _, _ = sample_csv_data
        train_dir = os.path.join(temp_dir, 'train_images')
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        dataset = RetinaMultiLabelDataset(train_csv, train_dir, transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        model = build_model('resnet18', num_classes=3, pretrained=False)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')
        
        loss = train_epoch(model, loader, criterion, optimizer, device)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_validate_epoch(self, sample_csv_data, temp_dir):
        """Test validate_epoch executes without errors"""
        from torch.utils.data import DataLoader
        from torchvision import transforms
        
        _, val_csv, _ = sample_csv_data
        val_dir = os.path.join(temp_dir, 'val_images')
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        dataset = RetinaMultiLabelDataset(val_csv, val_dir, transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        model = build_model('resnet18', num_classes=3, pretrained=False)
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cpu')
        
        loss = validate_epoch(model, loader, criterion, device)
        
        assert isinstance(loss, float)
        assert loss >= 0


# ============================================================================
# EVALUATE.PY TESTS
# ============================================================================

class TestModelEvaluator:
    """Test suite for ModelEvaluator class"""
    
    def test_evaluator_init(self, sample_config):
        """Test ModelEvaluator initialization"""
        cm = ConfigManager(sample_config)
        evaluator = ModelEvaluator(cm)
        
        assert evaluator.config_manager is not None
        assert evaluator.device is not None
    
    def test_load_model(self, sample_config, temp_dir):
        """Test load_model loads a saved model"""
        cm = ConfigManager(sample_config)
        evaluator = ModelEvaluator(cm)
        
        # Create and save a dummy model
        model = build_model('resnet18', num_classes=3, pretrained=False)
        model_path = os.path.join(temp_dir, 'test_model.pt')
        torch.save(model.state_dict(), model_path)
        
        # Load the model
        loaded_model = evaluator.load_model(model_path)
        
        assert isinstance(loaded_model, nn.Module)
    
    def test_prepare_test_data(self, sample_config, sample_csv_data, temp_dir):
        """Test prepare_test_data creates DataLoader"""
        cm = ConfigManager(sample_config)
        evaluator = ModelEvaluator(cm)
        
        test_loader, test_ds = evaluator.prepare_test_data(batch_size=2, img_size=64)
        
        assert test_loader is not None
        assert test_ds is not None
        assert len(test_ds) > 0
    
    def test_evaluate_model(self, sample_config, sample_csv_data, temp_dir):
        """Test evaluate_model returns predictions"""
        from torch.utils.data import DataLoader
        from torchvision import transforms
        
        cm = ConfigManager(sample_config)
        evaluator = ModelEvaluator(cm)
        
        _, _, test_csv = sample_csv_data
        test_dir = os.path.join(temp_dir, 'test_images')
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        dataset = RetinaMultiLabelDataset(test_csv, test_dir, transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        model = build_model('resnet18', num_classes=3, pretrained=False)
        
        y_true, y_pred, y_probs = evaluator.evaluate_model(model, loader, threshold=0.5)
        
        assert y_true.shape[0] == len(dataset)
        assert y_pred.shape[0] == len(dataset)
        assert y_probs.shape[0] == len(dataset)
        assert y_true.shape[1] == 3
        assert y_pred.shape[1] == 3
        assert y_probs.shape[1] == 3
    
    def test_calculate_metrics(self, sample_config):
        """Test calculate_metrics computes correct metrics"""
        cm = ConfigManager(sample_config)
        evaluator = ModelEvaluator(cm)
        
        # Create dummy predictions
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert 'disease_metrics' in metrics
        assert 'overall_metrics' in metrics
        assert 'DR' in metrics['disease_metrics']
        assert 'Glaucoma' in metrics['disease_metrics']
        assert 'AMD' in metrics['disease_metrics']
        assert 'accuracy' in metrics['overall_metrics']
        assert 'precision' in metrics['overall_metrics']
        assert 'recall' in metrics['overall_metrics']
        assert 'f1' in metrics['overall_metrics']
    
    def test_save_results(self, sample_config, sample_csv_data, temp_dir):
        """Test save_results creates output files"""
        cm = ConfigManager(sample_config)
        evaluator = ModelEvaluator(cm)
        
        _, _, test_csv = sample_csv_data
        test_dir = os.path.join(temp_dir, 'test_images')
        
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        test_ds = RetinaMultiLabelDataset(test_csv, test_dir, transform)
        
        # Create dummy predictions
        y_pred = np.array([[1, 0, 1], [0, 1, 0]])
        y_probs = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])
        
        metrics = {
            'disease_metrics': {
                'DR': {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75, 'f1': 0.77, 'kappa': 0.65},
                'Glaucoma': {'accuracy': 0.90, 'precision': 0.88, 'recall': 0.85, 'f1': 0.86, 'kappa': 0.78},
                'AMD': {'accuracy': 0.88, 'precision': 0.82, 'recall': 0.80, 'f1': 0.81, 'kappa': 0.70}
            },
            'overall_metrics': {
                'accuracy': 0.88, 'precision': 0.83, 'recall': 0.80, 'f1': 0.81, 'kappa': 0.71
            }
        }
        
        config_id = 'test_config'
        submission_path = evaluator.save_results(y_pred, y_probs, test_ds, metrics, config_id)
        
        assert os.path.exists(submission_path)
        
        # Check if other files were created
        results_dir = evaluator.paths['output_dir']
        prob_path = os.path.join(results_dir, f'probabilities_{config_id}.csv')
        metrics_path = os.path.join(results_dir, f'evaluation_metrics_{config_id}.json')
        
        assert os.path.exists(prob_path)
        assert os.path.exists(metrics_path)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for end-to-end workflows"""
    
    def test_full_pipeline_config_to_model(self, sample_config, sample_csv_data):
        """Test full pipeline from config loading to model creation"""
        cm = ConfigManager(sample_config)
        
        # Get configurations
        paths = cm.get_paths()
        model_config = cm.get_model_config()
        training_config = cm.get_training_config()
        
        # Build model
        model = build_model(
            backbone=model_config['backbone'],
            num_classes=model_config['num_classes'],
            pretrained=False
        )
        
        # Verify
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_dataset_to_dataloader_pipeline(self, sample_csv_data, temp_dir):
        """Test pipeline from dataset creation to DataLoader"""
        from torch.utils.data import DataLoader
        from torchvision import transforms
        
        train_csv, _, _ = sample_csv_data
        train_dir = os.path.join(temp_dir, 'train_images')
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        dataset = RetinaMultiLabelDataset(train_csv, train_dir, transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Get a batch
        imgs, labels = next(iter(loader))
        
        assert imgs.shape == (2, 3, 64, 64)
        assert labels.shape == (2, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
