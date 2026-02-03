# tests/test_config.py
import pytest
import yaml
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/')))
from config_manager import ConfigManager

def test_config_file_exists():
    """Test que le fichier config.yaml existe"""
    assert os.path.exists('config.yaml'), "config.yaml should exist"

def test_config_manager_initialization():
    """Test l'initialisation du ConfigManager"""
    config_manager = ConfigManager('config.yaml')
    assert config_manager is not None
    assert hasattr(config_manager, 'config')
    assert isinstance(config_manager.config, dict)

def test_get_paths():
    """Test la récupération des chemins"""
    config_manager = ConfigManager('config.yaml')
    paths = config_manager.get_paths()
    
    assert 'train_csv' in paths
    assert 'output_model_dir' in paths
    assert 'output_dir' in paths

def test_get_model_config():
    """Test la récupération de la config modèle"""
    config_manager = ConfigManager('config.yaml')
    model_config = config_manager.get_model_config()
    
    assert 'backbone' in model_config
    assert 'task' in model_config
    assert 'num_classes' in model_config

def test_config_valid_yaml():
    """Test que le config.yaml est un YAML valide"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    assert config is not None
    assert isinstance(config, dict)