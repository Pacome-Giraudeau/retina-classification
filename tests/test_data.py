# tests/test_data.py
import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/')))

from train import RetinaMultiLabelDataset, get_class_weights

def test_sample_data_exists():
    """Test que les données d'échantillon existent"""
    sample_files = ['tests/samples/train.csv', 'tests/samples/val.csv']
    
    for file in sample_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            assert len(df) > 0, f"{file} should not be empty"
            # Vérifier les colonnes nécessaires
            assert 'id' in df.columns
            # Vérifier qu'il y a des colonnes de labels
            label_cols = [col for col in df.columns if col != 'id']
            assert len(label_cols) > 0

def test_dataset_initialization():
    """Test l'initialisation du Dataset"""
    if os.path.exists('data/samples/train.csv'):
        dataset = RetinaMultiLabelDataset(
            csv_file='data/samples/train.csv',
            image_dir='data/samples/images',
            transform=None
        )
        assert len(dataset) > 0
        # Test un élément
        if len(dataset) > 0:
            img, labels = dataset[0]
            assert labels is not None

def test_class_weights():
    """Test le calcul des poids de classe"""
    if os.path.exists('data/samples/train.csv'):
        weights = get_class_weights('data/samples/train.csv')
        assert weights is not None
        assert len(weights) > 0