import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

import numpy as np
import pandas as pd
import torch

def get_class_weights(csv_file):
    df = pd.read_csv(csv_file)
    labels = df.iloc[:, 1:].values
    counts = labels.sum(axis=0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)   # normalize
    return torch.tensor(weights, dtype=torch.float32)

# ========================
# Focal Loss Implementation
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

        pt = prob * targets + (1 - prob) * (1 - targets)  # probability of ground truth class
        focal_weight = (1 - pt) ** self.gamma

        loss = self.alpha * focal_weight * bce_loss

        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ========================
# Class-Balanced BCE Loss Implementation
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

# ========================
# Dataset preparation
# ========================
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


# ========================
# build model
# ========================
def build_model(backbone="resnet18", num_classes=3, pretrained=True):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model


# ========================
# model training and val
# ========================
def train_one_config(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
                     config, save_dir="checkpoints", pretrained_backbone=None, task="t1.3"):
    """Train model with a specific hyperparameter configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract hyperparameters from config
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['lr']
    img_size = config['img_size']
    weight_decay = config.get('weight_decay', 0.0)
    
    # Create unique identifier for this configuration
    config_id = f"{task}_{backbone}_e{epochs}_bs{batch_size}_lr{lr}_im{img_size}_wd{weight_decay}"
    
    
    affiche_infos(config, config_id)

    # transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    # # Set parameter gradients
    # for p in model.parameters():
    #     p.requires_grad = True
    
    # loss & optimizer  
    # ===== Choose Loss function based on task2 =====
    if task == "t2_focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)

    elif task == "t2_cb":
        class_weights = get_class_weights(train_csv)
        criterion = ClassBalancedBCELoss(class_weights)

    else:
        criterion = nn.BCEWithLogitsLoss()   # default
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=lr, weight_decay=weight_decay)

    # Load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict)
    
    if task == "t1.2":
        # Freeze backbone for task 1.2
        if backbone == "resnet18":
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        elif backbone == "efficientnet":
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    # Create checkpoint directory for this config
    config_checkpoint_dir = os.path.join(save_dir, "configs", config_id)
    os.makedirs(config_checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config_checkpoint_dir, f"best_model.pt")
    
    # Create results directory
    results_dir = "submission_results"
    os.makedirs(results_dir, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    train_history, val_history = [], []

    for epoch in range(epochs):
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

        train_loss /= len(train_loader.dataset)
        train_history.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        val_history.append(val_loss)

        print(f"[{config_id}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    torch.save(model.state_dict(), ckpt_path)

    # ========================
    # Testing
    # ========================
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate metrics
    disease_names = ["DR", "Glaucoma", "AMD"]
    results = {
        'config_id': config_id,
        'task': task,
        'backbone': backbone,
        'hyperparameters': config,
        # 'best_val_loss': best_val_loss,
        'disease_metrics': {},
        'overall_metrics': {}
    }

    # Per-disease metrics
    for i, disease in enumerate(disease_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        results['disease_metrics'][disease] = {
            'accuracy': accuracy_score(y_t, y_p),
            'precision': precision_score(y_t, y_p, average="macro", zero_division=0),
            'recall': recall_score(y_t, y_p, average="macro", zero_division=0),
            'f1': f1_score(y_t, y_p, average="macro", zero_division=0),
            'kappa': cohen_kappa_score(y_t, y_p)
        }

    # Overall metrics (macro-averaged across diseases)
    results['overall_metrics'] = {
        'accuracy': np.mean([results['disease_metrics'][d]['accuracy'] for d in disease_names]),
        'precision': np.mean([results['disease_metrics'][d]['precision'] for d in disease_names]),
        'recall': np.mean([results['disease_metrics'][d]['recall'] for d in disease_names]),
        'f1': np.mean([results['disease_metrics'][d]['f1'] for d in disease_names]),
        'kappa': np.mean([results['disease_metrics'][d]['kappa'] for d in disease_names])
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {config_id}")
    # print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Overall Metrics - Accuracy: {results['overall_metrics']['accuracy']:.4f}, "
          f"F1: {results['overall_metrics']['f1']:.4f}")
    
    for disease in disease_names:
        metrics = results['disease_metrics'][disease]
        print(f"{disease}: Acc={metrics['accuracy']:.4f}, "
              f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
              f"F1={metrics['f1']:.4f}, K={metrics['kappa']:.4f}")

    # Create submission file
    submission_df = pd.DataFrame({
        'id': test_ds.data.iloc[:, 0].values[:len(y_pred)],
        'D': y_pred[:, 0],
        'G': y_pred[:, 1],
        'A': y_pred[:, 2]
    })

    # Save submission with config information in filename
    submission_filename = f"submission_{config_id}.csv"
    submission_path = os.path.join(results_dir, submission_filename)
    submission_df.to_csv(submission_path, index=False)
    
    # Save probabilities for analysis
    prob_df = pd.DataFrame({
        'id': test_ds.data.iloc[:, 0].values[:len(y_probs)],
        'D_prob': y_probs[:, 0],
        'G_prob': y_probs[:, 1],
        'A_prob': y_probs[:, 2]
    })
    prob_filename = f"probabilities_{config_id}.csv"
    prob_path = os.path.join(results_dir, prob_filename)
    prob_df.to_csv(prob_path, index=False)

    # Save results summary
    results_filename = f"results_{config_id}.json"
    results_path = os.path.join(results_dir, results_filename)
    
    # Convert numpy types to Python types for JSON serialization
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
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nSaved files for {config_id}:")
    print(f"  - Submission: {submission_path}")
    print(f"  - Probabilities: {prob_path}")
    print(f"  - Results: {results_path}")
    print(f"{'='*60}\n")

    return results

def affiche_infos(config, config_id):
    print(f"\n{'='*60}")
    print(f"Training configuration: {config_id}")
    print(f"Hyperparameters: {config}")
    print(f"{'='*60}")


# ========================
# Hyperparameter grid search
# ========================
def run_hyperparameter_search(backbone, task, train_csv, val_csv, test_csv,
                              train_image_dir, val_image_dir, test_image_dir,
                              pretrained_backbone=None):
    """Run grid search over hyperparameters"""
    
    # Define hyperparameter grids
    if task == "t1.2":
        # For task 1.2 (fine-tuning only classifier)
        param_grid = {
            'epochs': [20],
            'batch_size': [32],
            'lr': [6e-4, 7e-4, 8e-4],
            'img_size': [256],
            'weight_decay': [0.0]
        }
    else:
        # For task 1.3 and task2 (full fine-tuning)
        param_grid = {
            'epochs': [20],
            'batch_size': [32],
            'lr':  [30e-5, 20e-5, 10e-5],
            'img_size': [256],
            'weight_decay': [0.0]
        }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    print(f"\n{'='*60}")
    print(f"Starting hyperparameter search for {backbone} ({task})")
    print(f"Total configurations to test: {len(all_combinations)}")
    print(f"{'='*60}\n")
    
    # Track all results
    all_results = []
    
    # Test each configuration
    for i, config in enumerate(all_combinations):
        print(f"\nTesting configuration {i+1}/{len(all_combinations)}")
        
        try:
            results = train_one_config(
                backbone=backbone,
                train_csv=train_csv,
                val_csv=val_csv,
                test_csv=test_csv,
                train_image_dir=train_image_dir,
                val_image_dir=val_image_dir,
                test_image_dir=test_image_dir,
                config=config,
                save_dir="checkpoints",
                pretrained_backbone=pretrained_backbone,
                task=task
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"Error training configuration {config}: {e}")
            continue
    
    # Create summary of all results
    if all_results:
        summary_data = []
        for result in all_results:
            summary_data.append({
                'config_id': result['config_id'],
                'task': result['task'],
                'backbone': result['backbone'],
                'epochs': result['hyperparameters']['epochs'],
                'batch_size': result['hyperparameters']['batch_size'],
                'lr': result['hyperparameters']['lr'],
                'img_size': result['hyperparameters']['img_size'],
                'weight_decay': result['hyperparameters'].get('weight_decay', 0.0),
                # 'best_val_loss': result['best_val_loss'],
                'overall_accuracy': result['overall_metrics']['accuracy'],
                'overall_f1': result['overall_metrics']['f1'],
                'overall_kappa': result['overall_metrics']['kappa']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by overall F1 score (or choose your preferred metric)
        summary_df = summary_df.sort_values('overall_f1', ascending=False)
        
        # Save summary
        summary_path = os.path.join("submission_results", f"hyperparameter_summary_{task}_{backbone}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Hyperparameter search complete!")
        print(f"Summary saved to: {summary_path}")
        print(f"\nTop 5 configurations by F1 score:")
        print(summary_df.head(5).to_string())
        print(f"{'='*60}")
        
        return summary_df
    else:
        print("No successful configurations completed.")
        return None


# ========================
# Single configuration run (for quick testing)
# ========================
def run_single_configuration(backbone, task, train_csv, val_csv, test_csv,
                             train_image_dir, val_image_dir, test_image_dir,
                             pretrained_backbone=None):
    """Run a single configuration with default/recommended hyperparameters"""
    
    if task == "t1.2":
        # Recommended for task 1.2
        config = {
            'epochs': 20,
            'batch_size': 16,
            'lr': 1e-5,
            'img_size': 256,
            'weight_decay': 0.0
        }
    else:
        # Recommended for task 1.3
        config = {
            'epochs': 25,
            'batch_size': 16,
            'lr': 3e-5,
            'img_size': 256,
            'weight_decay': 1e-5
        }
    
    results = train_one_config(
        backbone=backbone,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        config=config,
        save_dir="checkpoints",
        pretrained_backbone=pretrained_backbone,
        task=task
    )
    
    return results


# ========================
# main
# ========================
if __name__ == "__main__":
    # Path configurations
    train_csv = "train.csv"  # replace with your own train label file path
    val_csv = "val.csv"  # replace with your own validation label file path
    test_csv = "onsite_test_submission.csv"  # replace with your own test label file path
    train_image_dir = "./images/train"  # replace with your own train image folder path
    val_image_dir = "./images/val"  # replace with your own validation image folder path
    test_image_dir = "./images/onsite_test"  # replace with your own test image folder path
    pretrained_backbone = './pretrained_backbone/ckpt_resnet18_ep50.pt'  # replace with your own pretrained backbone path
    
    # Configuration
    backbone = 'resnet18'  # choices: ["resnet18", "efficientnet"]
    task = "t2_focal"  # choices: ["t1.2", "t1.3", "t2_focal", "t2_cb"]
    run_mode = "grid_search"  # choices: ["single", "grid_search"]
    
    if run_mode == "grid_search":
        # Run hyperparameter grid search
        results_summary = run_hyperparameter_search(
            backbone=backbone,
            task=task,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            train_image_dir=train_image_dir,
            val_image_dir=val_image_dir,
            test_image_dir=test_image_dir,
            pretrained_backbone=pretrained_backbone
        )
    else:
        # Run single configuration
        results = run_single_configuration(
            backbone=backbone,
            task=task,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            train_image_dir=train_image_dir,
            val_image_dir=val_image_dir,
            test_image_dir=test_image_dir,
            pretrained_backbone=pretrained_backbone
        )