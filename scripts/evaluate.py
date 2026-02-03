# scripts/evaluate.py
import os
import sys

# Ajouter le répertoire parent au PYTHONPATH pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# Import relatif
from config_manager import ConfigManager
from train import RetinaMultiLabelDataset, build_model, get_class_weights, FocalLoss, ClassBalancedBCELoss


class ModelEvaluator:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.paths = config_manager.create_directories()
        self.model_config = config_manager.get_model_config()
        self.device_config = config_manager.get_device_config()
        self.output_config = config_manager.get_output_config()
        
        self.device = torch.device(self.device_config['device'])
        
    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        model = build_model(
            backbone=self.model_config['backbone'],
            num_classes=self.model_config['num_classes'],
            pretrained=False
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        print(f"Loaded model from {model_path}")
        return model
    
    def load_best_model(self, config_id=None):
        """Load the best model from output directory"""
        if config_id:
            model_path = os.path.join(self.paths['output_model_dir'], f"best_model_{config_id}.pt")
        else:
            # Find the most recent best model
            model_files = [f for f in os.listdir(self.paths['output_model_dir']) 
                          if f.startswith('best_model_') and f.endswith('.pt')]
            
            if not model_files:
                raise FileNotFoundError(f"No model found in {self.paths['output_model_dir']}")
            
            # Sort by modification time (most recent first)
            model_files.sort(key=lambda x: os.path.getmtime(
                os.path.join(self.paths['output_model_dir'], x)
            ), reverse=True)
            
            model_path = os.path.join(self.paths['output_model_dir'], model_files[0])
            config_id = model_files[0].replace('best_model_', '').replace('.pt', '')
        
        model = self.load_model(model_path)
        return model, config_id
    
    def prepare_test_data(self, batch_size=32, img_size=256):
        """Prepare test data loader"""
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if 'test_csv' in self.paths and os.path.exists(self.paths['test_csv']):
            test_ds = RetinaMultiLabelDataset(
                self.paths['test_csv'], 
                self.paths['test_image_dir'], 
                transform
            )
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            return test_loader, test_ds
        else:
            print("Test CSV not found. Using validation set for evaluation.")
            test_ds = RetinaMultiLabelDataset(
                self.paths['val_csv'], 
                self.paths['val_image_dir'], 
                transform
            )
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            return test_loader, test_ds
    
    def evaluate_model(self, model, test_loader, threshold=0.5):
        """Evaluate model on test data"""
        model.eval()
        y_true, y_pred, y_probs = [], [], []
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(self.device)
                outputs = model(imgs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > threshold).astype(int)
                y_true.extend(labels.numpy())
                y_pred.extend(preds)
                y_probs.extend(probs)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        return y_true, y_pred, y_probs
    
    def calculate_metrics(self, y_true, y_pred, disease_names=None):
        """Calculate evaluation metrics"""
        if disease_names is None:
            disease_names = ["DR", "Glaucoma", "AMD"]
        
        metrics = {
            'disease_metrics': {},
            'overall_metrics': {}
        }
        
        # Per-disease metrics
        for i, disease in enumerate(disease_names):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            
            # Handle cases where all predictions are the same
            if len(np.unique(y_p)) == 1 or len(np.unique(y_t)) == 1:
                metrics['disease_metrics'][disease] = {
                    'accuracy': accuracy_score(y_t, y_p),
                    'precision': precision_score(y_t, y_p, average="macro", zero_division=0),
                    'recall': recall_score(y_t, y_p, average="macro", zero_division=0),
                    'f1': f1_score(y_t, y_p, average="macro", zero_division=0),
                    'kappa': 0.0  # Cohen's kappa requires diversity
                }
            else:
                metrics['disease_metrics'][disease] = {
                    'accuracy': accuracy_score(y_t, y_p),
                    'precision': precision_score(y_t, y_p, average="macro", zero_division=0),
                    'recall': recall_score(y_t, y_p, average="macro", zero_division=0),
                    'f1': f1_score(y_t, y_p, average="macro", zero_division=0),
                    'kappa': cohen_kappa_score(y_t, y_p)
                }
        
        # Overall metrics (macro-averaged)
        metrics['overall_metrics'] = {
            'accuracy': np.mean([metrics['disease_metrics'][d]['accuracy'] for d in disease_names]),
            'precision': np.mean([metrics['disease_metrics'][d]['precision'] for d in disease_names]),
            'recall': np.mean([metrics['disease_metrics'][d]['recall'] for d in disease_names]),
            'f1': np.mean([metrics['disease_metrics'][d]['f1'] for d in disease_names]),
            'kappa': np.mean([metrics['disease_metrics'][d]['kappa'] for d in disease_names])
        }
        
        return metrics
    
    def save_results(self, y_pred, y_probs, test_ds, metrics, config_id):
        """Save evaluation results"""
        # Create results directory
        results_dir = self.paths['output_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions
        submission_df = pd.DataFrame({
            'id': test_ds.data.iloc[:, 0].values[:len(y_pred)],
            'D': y_pred[:, 0],
            'G': y_pred[:, 1],
            'A': y_pred[:, 2]
        })
        
        submission_path = os.path.join(results_dir, f"submission_{config_id}.csv")
        submission_df.to_csv(submission_path, index=False)
        
        # Save probabilities if configured
        if self.output_config.get('save_probabilities', True):
            prob_df = pd.DataFrame({
                'id': test_ds.data.iloc[:, 0].values[:len(y_probs)],
                'D_prob': y_probs[:, 0],
                'G_prob': y_probs[:, 1],
                'A_prob': y_probs[:, 2]
            })
            prob_path = os.path.join(results_dir, f"probabilities_{config_id}.csv")
            prob_df.to_csv(prob_path, index=False)
        
        # Save metrics if configured
        if self.output_config.get('save_metrics', True):
            # Add config information
            full_results = {
                'config_id': config_id,
                'task': self.model_config['task'],
                'backbone': self.model_config['backbone'],
                'model_config': self.model_config,
                'metrics': metrics
            }
            
            # Convert numpy types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_to_serializable(full_results)
            
            metrics_path = os.path.join(results_dir, f"evaluation_metrics_{config_id}.json")
            with open(metrics_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        return submission_path
    
    def print_metrics(self, metrics, config_id):
        """Print evaluation metrics in a readable format"""
        print(f"\n{'='*60}")
        print(f"Evaluation Results for {config_id}")
        print(f"{'='*60}")
        
        overall = metrics['overall_metrics']
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {overall['accuracy']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1-Score:  {overall['f1']:.4f}")
        print(f"  Kappa:     {overall['kappa']:.4f}")
        
        print(f"\nPer-Disease Metrics:")
        for disease, disease_metrics in metrics['disease_metrics'].items():
            print(f"\n  {disease}:")
            print(f"    Accuracy:  {disease_metrics['accuracy']:.4f}")
            print(f"    Precision: {disease_metrics['precision']:.4f}")
            print(f"    Recall:    {disease_metrics['recall']:.4f}")
            print(f"    F1-Score:  {disease_metrics['f1']:.4f}")
            print(f"    Kappa:     {disease_metrics['kappa']:.4f}")
        
        print(f"\n{'='*60}")

def main():
    config_manager = ConfigManager()
    evaluator = ModelEvaluator(config_manager)
    
    # Load the best model
    try:
        model, config_id = evaluator.load_best_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train a model first using train.py")
        return
    
    # Prepare test data
    test_loader, test_ds = evaluator.prepare_test_data(batch_size=32, img_size=256)
    
    # Evaluate model
    print(f"\nEvaluating model on test data...")
    y_true, y_pred, y_probs = evaluator.evaluate_model(model, test_loader)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    # Print results
    evaluator.print_metrics(metrics, config_id)
    
    # Save results
    submission_path = evaluator.save_results(y_pred, y_probs, test_ds, metrics, config_id)
    
    print(f"\nResults saved to:")
    print(f"  - Submission file: {submission_path}")
    
    # Save confusion matrices for each disease
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    disease_names = ["DR", "Glaucoma", "AMD"]
    
    for i, disease in enumerate(disease_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {disease}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = os.path.join(evaluator.paths['output_dir'], f"confusion_matrix_{disease}_{config_id}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Confusion matrix ({disease}): {cm_path}")

if __name__ == "__main__":
    main()