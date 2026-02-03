# scripts/config_manager.py
import yaml
import os
from pathlib import Path
import torch

class ConfigManager:
    def __init__(self, config_path="../config.yaml"):  # ← Chemin modifié
        # Si le chemin n'existe pas, essayer dans le répertoire courant
        if not os.path.exists(config_path):
            # Essayer depuis le répertoire scripts/
            config_path = "config.yaml"
            if not os.path.exists(config_path):
                # Essayer depuis le répertoire parent
                config_path = os.path.join("..", "config.yaml")
        
        self.config_path = config_path
        self.config = self.load_config()
        
        print(f"Loading config from: {os.path.abspath(self.config_path)}")
    
    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def get_paths(self):
        paths = self.config.get('paths', {})
        base_dir = os.path.dirname(os.path.abspath(self.config_path))
        
        # Fonction pour résoudre les chemins relatifs
        def resolve_path(path):
            if path is None:
                return None
            # Si c'est un chemin absolu, le garder tel quel
            if os.path.isabs(path):
                return path
            # Sinon, le rendre relatif au répertoire du config.yaml
            return path #os.path.join(base_dir, path)
        
        return {
            'train_csv': resolve_path(paths.get('train_csv')),
            'val_csv': resolve_path(paths.get('val_csv')),
            'test_csv': resolve_path(paths.get('test_csv')),
            'train_image_dir': resolve_path(paths.get('train_image_dir')),
            'val_image_dir': resolve_path(paths.get('val_image_dir')),
            'test_image_dir': resolve_path(paths.get('test_image_dir')),
            'pretrained_backbone_efficient': resolve_path(paths.get('pretrained_backbone_efficient')),
            'pretrained_backbone_resnet18': resolve_path(paths.get('pretrained_backbone_resnet18')),
            'output_model_dir': resolve_path(paths.get('output_model_dir', './models')),
            'output_dir': resolve_path(paths.get('output_dir', './results'))
        }
    
    def get_model_config(self):
        model_config = self.config.get('model', {})
        return {
            'backbone': model_config.get('backbone', 'efficientnet'),
            'task': model_config.get('task', 't1.3'),
            'num_classes': model_config.get('num_classes', 3)
        }
    
    def get_training_config(self):
        training_config = self.config.get('hyperparameters', {})
        return {
            'epochs': training_config.get('epochs', 20),
            'batch_size': training_config.get('batch_size', 32),
            'learning_rate': training_config.get('learning_rate', 0.0001),
            'img_size': training_config.get('img_size', 256),
            'weight_decay': training_config.get('weight_decay', 0.0)
        }
    
    def get_grid_search_config(self):
        grid_config = self.config.get('grid_search', {})
        return {
            'epochs': grid_config.get('epochs', [20]),
            'batch_size': grid_config.get('batch_size', [32]),
            'learning_rate': grid_config.get('learning_rate', [0.0001]),
            'img_size': grid_config.get('img_size', [256]),
            'weight_decay': grid_config.get('weight_decay', [0.0])
        }
    
    def get_run_mode(self):
        return self.config.get('run_mode', 'single')
    
    def get_output_config(self):
        output_config = self.config.get('output', {})
        return {
            'save_checkpoints': output_config.get('save_checkpoints', True),
            'save_probabilities': output_config.get('save_probabilities', True),
            'save_metrics': output_config.get('save_metrics', True)
        }
    
    def get_device_config(self):
        device_config = self.config.get('device', {})
        use_cuda = device_config.get('use_cuda', True)
        return {
            'device': 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        }
    
    def create_directories(self):
        paths = self.get_paths()
        print(paths)
        os.makedirs(paths['output_model_dir'], exist_ok=True)
        os.makedirs(paths['output_dir'], exist_ok=True)
        return paths