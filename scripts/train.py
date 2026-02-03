# scripts/train.py
import os
import sys

# Ajouter le répertoire parent au PYTHONPATH pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from PIL import Image
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Import relatif depuis le même dossier scripts
from config_manager import ConfigManager


def get_class_weights(csv_file):
    """Calculate class weights for imbalance handling"""
    df = pd.read_csv(csv_file)
    labels = df.iloc[:, 1:].values
    counts = labels.sum(axis=0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class ClassBalancedBCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits, targets):
        weights = self.class_weights.to(logits.device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weights
        )
        return loss

class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels

def build_model(backbone="resnet18", num_classes=3, pretrained=True):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model

def get_pretrained_backbone_path(backbone, config_manager):
    paths = config_manager.get_paths()
    if backbone == "resnet18":
        return paths.get('pretrained_backbone_resnet18')
    elif backbone == "efficientnet":
        return paths.get('pretrained_backbone_efficient')
    return None

def create_data_loaders(config, paths, batch_size, img_size):
    """Create train, validation, and test data loaders"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_ds = RetinaMultiLabelDataset(paths['train_csv'], paths['train_image_dir'], transform)
    val_ds = RetinaMultiLabelDataset(paths['val_csv'], paths['val_image_dir'], transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def setup_model(backbone, num_classes, task, pretrained_backbone_path, device):
    """Initialize model with appropriate settings based on task"""
    model = build_model(backbone, num_classes, pretrained=False).to(device)
    
    # Load pretrained weights if available
    if pretrained_backbone_path and os.path.exists(pretrained_backbone_path):
        state_dict = torch.load(pretrained_backbone_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_backbone_path}")
    
    # Freeze backbone for task 1.2
    if task == "t1.2":
        if backbone == "resnet18":
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        elif backbone == "efficientnet":
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        print("Frozen backbone layers for task 1.2")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    
    return train_loss / len(train_loader.dataset)

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
    
    return val_loss / len(val_loader.dataset)

def train_single_configuration(config_manager, config, config_id=None):
    """Train with a single hyperparameter configuration"""
    paths = config_manager.create_directories()
    model_config = config_manager.get_model_config()
    device_config = config_manager.get_device_config()
    
    device = torch.device(device_config['device'])
    
    # Extract training parameters
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    img_size = config['img_size']
    weight_decay = config.get('weight_decay', 0.0)
    
    # Create config ID if not provided
    if config_id is None:
        config_id = f"{model_config['task']}_{model_config['backbone']}_e{epochs}_bs{batch_size}_lr{lr}"
    
    print(f"\n{'='*60}")
    print(f"Training configuration: {config_id}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config_manager, paths, batch_size, img_size)
    
    # Setup model
    pretrained_backbone_path = get_pretrained_backbone_path(model_config['backbone'], config_manager)
    model = setup_model(
        backbone=model_config['backbone'],
        num_classes=model_config['num_classes'],
        task=model_config['task'],
        pretrained_backbone_path=pretrained_backbone_path,
        device=device
    )
    
    # Setup loss function
    if model_config['task'] == "t2_focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif model_config['task'] == "t2_cb":
        class_weights = get_class_weights(paths['train_csv'])
        criterion = ClassBalancedBCELoss(class_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Setup optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_history, val_history = [], []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(paths['output_model_dir'], f"best_model_{config_id}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")
    
    # Save training history
    history = {
        'train_loss': train_history,
        'val_loss': val_history,
        'best_val_loss': best_val_loss,
        'config_id': config_id,
        'hyperparameters': config
    }
    
    history_path = os.path.join(paths['output_dir'], f"training_history_{config_id}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed for {config_id}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"History saved to: {history_path}")
    
    return model, best_val_loss, config_id

def run_hyperparameter_search(config_manager):
    """Run grid search over hyperparameters"""
    paths = config_manager.create_directories()
    model_config = config_manager.get_model_config()
    grid_config = config_manager.get_grid_search_config()
    
    print(f"\n{'='*60}")
    print(f"Starting hyperparameter search for {model_config['backbone']} ({model_config['task']})")
    print(f"{'='*60}\n")
    
    # Generate all combinations
    keys = ['epochs', 'batch_size', 'learning_rate', 'img_size', 'weight_decay']
    values = [grid_config[k] for k in keys]
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    print(f"Total configurations to test: {len(all_combinations)}\n")
    
    results = []
    
    for i, config in enumerate(all_combinations):
        print(f"\nTesting configuration {i+1}/{len(all_combinations)}: {config}")
        
        try:
            config_id = f"{model_config['task']}_{model_config['backbone']}_config{i+1}"
            model, best_val_loss, final_config_id = train_single_configuration(
                config_manager, config, config_id
            )
            
            results.append({
                'config_id': final_config_id,
                'hyperparameters': config,
                'best_val_loss': best_val_loss
            })
            
        except Exception as e:
            print(f"Error training configuration {config}: {e}")
            continue
    
    # Save grid search results
    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(paths['output_dir'], f"grid_search_results_{model_config['task']}_{model_config['backbone']}.csv")
        results_df.to_csv(results_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Grid search complete!")
        print(f"Results saved to: {results_path}")
        print(f"\nBest configuration:")
        best_result = min(results, key=lambda x: x['best_val_loss'])
        print(f"Config ID: {best_result['config_id']}")
        print(f"Best Val Loss: {best_result['best_val_loss']:.4f}")
        print(f"Hyperparameters: {best_result['hyperparameters']}")
        print(f"{'='*60}")
        
        return best_result
    else:
        print("No successful configurations completed.")
        return None

def main():
    config_manager = ConfigManager()
    run_mode = config_manager.get_run_mode()
    
    if run_mode == "grid_search":
        best_config = run_hyperparameter_search(config_manager)
        print(f"\nBest configuration found: {best_config}")
    else:
        # Run single configuration
        training_config = config_manager.get_training_config()
        model, best_val_loss, config_id = train_single_configuration(
            config_manager, training_config
        )
        print(f"\nTraining completed successfully!")
        print(f"Model saved with config_id: {config_id}")
        print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()