"""
Training script for Baseline 1: Transfer Learning
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import yaml
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.baseline_transfer import TransferLearningBaseline
from src.data.standard_dataset import StandardDermaMNIST
from src.utils.metrics import compute_metrics


class TransferLearningTrainer:
    """Trainer for transfer learning baseline"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        # Load config
        if not os.path.isabs(config_path):
            if os.path.exists(config_path):
                self.config_path = config_path
            else:
                project_root = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                ))
                self.config_path = os.path.join(project_root, config_path)
        else:
            self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.baseline_config = self.config['baselines']['transfer_learning']
        
        # Set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Create save directory
        self.save_dir = self.baseline_config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.save_dir, 'logs')
        )
        
        # Training state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
    def build_model(self):
        """Build transfer learning model"""
        model = TransferLearningBaseline(
            num_classes=len(self.config['dataset']['class_names']),
            pretrained=self.baseline_config['pretrained'],
            freeze_backbone=self.baseline_config['freeze_backbone']
        )
        return model.to(self.device)
    
    def build_dataloaders(self):
        """Build train/val/test dataloaders"""
        train_dataset = StandardDermaMNIST(
            split='train',
            config_path=self.config_path,
            augment=True
        )
        
        val_dataset = StandardDermaMNIST(
            split='val',
            config_path=self.config_path,
            augment=False
        )
        
        test_dataset = StandardDermaMNIST(
            split='test',
            config_path=self.config_path,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.baseline_config['batch_size']),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(self.baseline_config['batch_size']),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(self.baseline_config['batch_size']),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, model, loader, criterion):
        """Validate model"""
        model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        
        # Compute metrics
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            num_classes=len(self.config['dataset']['class_names'])
        )
        
        return avg_loss, metrics
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("BASELINE 1: TRANSFER LEARNING (ResNet-50)")
        print("="*60 + "\n")
        
        # Build model
        model = self.build_model()
        
        # Build dataloaders
        train_loader, val_loader, test_loader = self.build_dataloaders()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(self.baseline_config['learning_rate']),
            weight_decay=float(self.baseline_config['weight_decay'])
        )
        
        # Learning rate scheduler
        if self.baseline_config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.baseline_config['num_epochs'])
            )
        else:
            scheduler = None
        
        # Training loop
        for epoch in range(int(self.baseline_config['num_epochs'])):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            val_loss, val_metrics = self.validate(
                model, val_loader, criterion
            )
            
            # Learning rate step
            if scheduler is not None:
                scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{int(self.baseline_config['num_epochs'])}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save checkpoint (FIXED: only save model state dict, not metrics with numpy)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': float(val_metrics['accuracy'])  # Convert to float
                }
                torch.save(
                    checkpoint,
                    os.path.join(self.save_dir, 'best_model.pth')
                )
                
                # Save metrics separately as JSON (no numpy arrays)
                metrics_to_save = {
                    'epoch': epoch,
                    'val_accuracy': float(val_metrics['accuracy']),
                    'val_macro_precision': float(val_metrics['macro_precision']),
                    'val_macro_recall': float(val_metrics['macro_recall']),
                    'val_macro_f1': float(val_metrics['macro_f1'])
                }
                with open(os.path.join(self.save_dir, 'best_metrics.json'), 'w') as f:
                    json.dump(metrics_to_save, f, indent=2)
                
                print(f"  ✓ New best model saved! (Val Acc: {self.best_val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= int(self.baseline_config['early_stopping_patience']):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining complete!")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        
        # Test on best model
        print("\nEvaluating on test set...")
        # FIXED: Add weights_only=False for PyTorch 2.6+
        checkpoint = torch.load(
            os.path.join(self.save_dir, 'best_model.pth'),
            map_location=self.device,
            weights_only=False  # Allow loading optimizer state
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_metrics = self.validate(model, test_loader, criterion)
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"  Per-class Accuracy:")
        for cls_id, acc in enumerate(test_metrics['per_class_accuracy']):
            cls_name = self.config['dataset']['class_names'][cls_id]
            print(f"    Class {cls_id} ({cls_name[:30]}...): {acc:.2f}%")
        
        # Save final results
        results = {
            'best_epoch': int(self.best_epoch + 1),
            'best_val_accuracy': float(self.best_val_acc),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_metrics': {
                'accuracy': float(test_metrics['accuracy']),
                'macro_precision': float(test_metrics['macro_precision']),
                'macro_recall': float(test_metrics['macro_recall']),
                'macro_f1': float(test_metrics['macro_f1']),
                'per_class_accuracy': test_metrics['per_class_accuracy'].tolist(),
                'per_class_precision': test_metrics['per_class_precision'].tolist(),
                'per_class_recall': test_metrics['per_class_recall'].tolist(),
                'per_class_f1': test_metrics['per_class_f1'].tolist(),
                'per_class_support': test_metrics['per_class_support'].tolist(),
                'confusion_matrix': test_metrics['confusion_matrix'].tolist()
            }
        }
        
        with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.writer.close()
        
        return results


if __name__ == "__main__":
    trainer = TransferLearningTrainer()
    results = trainer.train()