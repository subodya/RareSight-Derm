"""
Training script for Baseline 2: Standard ProtoNet
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import yaml
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.baseline_protonet import StandardProtoNet
from src.data.dataset import EpisodicDermaMNIST


class ProtoNetTrainer:
    """Trainer for Standard ProtoNet baseline"""
    
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
        
        self.baseline_config = self.config['baselines']['standard_protonet']
        
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
        self.best_episode = 0
    
    def build_model(self):
        """Build ProtoNet model"""
        model = StandardProtoNet(
            backbone=self.baseline_config['backbone'],
            pretrained=self.baseline_config['pretrained'],
            freeze_backbone=self.baseline_config['freeze_backbone'],
            feature_dim=int(self.baseline_config['feature_dim']),
            distance=self.baseline_config['distance']
        )
        return model.to(self.device)
    
    def build_datasets(self):
        """Build episodic datasets"""
        train_dataset = EpisodicDermaMNIST(
            split='train',
            config_path=self.config_path,
            download=True
        )
        
        val_dataset = EpisodicDermaMNIST(
            split='val',
            config_path=self.config_path,
            download=True
        )
        
        test_dataset = EpisodicDermaMNIST(
            split='test',
            config_path=self.config_path,
            download=True
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def train_episode(self, model, dataset, optimizer):
        """Train on one episode"""
        model.train()
        
        # Sample episode
        support_images, support_labels, query_images, query_labels = dataset.sample_episode(
            n_way=int(self.baseline_config['n_way']),
            k_shot=int(self.baseline_config['k_shot']),
            n_query=int(self.baseline_config['n_query'])
        )
        
        # Move to device
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        loss, accuracy = model.loss(
            support_images, support_labels,
            query_images, query_labels,
            n_way=int(self.baseline_config['n_way'])
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item(), accuracy
    
    def validate(self, model, dataset, n_episodes: int = 600):
        """Validate on multiple episodes"""
        model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        
        for _ in tqdm(range(n_episodes), desc="Validating"):
            # Sample episode
            support_images, support_labels, query_images, query_labels = dataset.sample_episode(
                n_way=int(self.baseline_config['n_way']),
                k_shot=int(self.baseline_config['k_shot']),
                n_query=int(self.baseline_config['n_query'])
            )
            
            # Move to device
            support_images = support_images.to(self.device)
            support_labels = support_labels.to(self.device)
            query_images = query_images.to(self.device)
            query_labels = query_labels.to(self.device)
            
            # Forward pass (no gradients)
            with torch.no_grad():
                loss, accuracy = model.loss(
                    support_images, support_labels,
                    query_images, query_labels,
                    n_way=int(self.baseline_config['n_way'])
                )
            
            total_loss += loss.item()
            total_accuracy += accuracy
        
        avg_loss = total_loss / n_episodes
        avg_accuracy = total_accuracy / n_episodes
        
        return avg_loss, avg_accuracy
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("BASELINE 2: STANDARD PROTONET (ResNet-50)")
        print("="*60 + "\n")
        
        # Build model
        model = self.build_model()
        
        # Build datasets
        train_dataset, val_dataset, test_dataset = self.build_datasets()
        
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(self.baseline_config['learning_rate']),
            weight_decay=float(self.baseline_config['weight_decay'])
        )
        
        # Learning rate scheduler
        if self.baseline_config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.baseline_config['episodes'])
            )
        else:
            scheduler = None
        
        # Training loop
        total_episodes = int(self.baseline_config['episodes'])
        save_freq = int(self.baseline_config['save_frequency'])
        
        print(f"Training for {total_episodes} episodes...")
        print(f"Validation every {save_freq} episodes\n")
        
        pbar = tqdm(range(total_episodes), desc="Training")
        
        for episode in pbar:
            # Train on one episode
            loss, accuracy = self.train_episode(model, train_dataset, optimizer)
            
            # Learning rate step
            if scheduler is not None:
                scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', loss, episode)
            self.writer.add_scalar('Accuracy/train', accuracy, episode)
            
            # Validate periodically
            if (episode + 1) % save_freq == 0:
                print(f"\n\nValidating at episode {episode + 1}...")
                
                val_loss, val_accuracy = self.validate(model, val_dataset, n_episodes=100)
                
                # Log validation
                self.writer.add_scalar('Loss/val', val_loss, episode)
                self.writer.add_scalar('Accuracy/val', val_accuracy, episode)
                
                print(f"Episode {episode + 1}/{total_episodes}")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
                
                # Save best model
                if val_accuracy > self.best_val_acc:
                    self.best_val_acc = val_accuracy
                    self.best_episode = episode
                    
                    checkpoint = {
                        'episode': episode,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_accuracy
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(self.save_dir, 'best_model.pth')
                    )
                    print(f"  ✓ New best model saved! (Val Acc: {self.best_val_acc:.2f}%)")
                
                print()  # Newline before resuming progress bar
        
        print(f"\nTraining complete!")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}% (Episode {self.best_episode + 1})")
        
        # Test on best model
        print("\nEvaluating on test set (600 episodes)...")
        checkpoint = torch.load(os.path.join(self.save_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_accuracy = self.validate(model, test_dataset, n_episodes=600)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.2f}%")
        
        # Save results
        results = {
            'best_episode': int(self.best_episode + 1),
            'best_val_accuracy': float(self.best_val_acc),
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'config': {
                'n_way': int(self.baseline_config['n_way']),
                'k_shot': int(self.baseline_config['k_shot']),
                'n_query': int(self.baseline_config['n_query']),
                'distance': self.baseline_config['distance']
            }
        }
        
        with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.writer.close()
        
        return results


if __name__ == "__main__":
    trainer = ProtoNetTrainer()
    results = trainer.train()