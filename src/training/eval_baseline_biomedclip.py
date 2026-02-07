"""
Evaluation script for Baseline 3: Zero-Shot BiomedCLIP
No training - just evaluate on test set
"""

import os
import sys
import torch
from tqdm import tqdm
import numpy as np
import yaml
import json
from typing import List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.baseline_biomedclip import ZeroShotBiomedCLIP
from src.data.standard_dataset import StandardDermaMNIST
from torch.utils.data import DataLoader
from src.utils.metrics import compute_metrics, print_metrics


class BiomedCLIPEvaluator:
    """Evaluator for zero-shot BiomedCLIP"""
    
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
        
        self.baseline_config = self.config['baselines']['zero_shot_biomedclip']
        
        # Set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Create save directory
        self.save_dir = self.baseline_config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Build text descriptions
        self.text_descriptions = self._build_text_descriptions()
    
    def _build_text_descriptions(self) -> List[str]:
        """Build text descriptions for each class"""
        template = self.baseline_config['prompt_template']
        class_names = self.config['dataset']['class_names']
        
        descriptions = []
        for cls_id in sorted(class_names.keys()):
            class_name = class_names[cls_id].lower()
            description = template.format(class_name=class_name)
            descriptions.append(description)
        
        print("\nText descriptions for zero-shot classification:")
        for i, desc in enumerate(descriptions):
            print(f"  Class {i}: {desc}")
        
        return descriptions
    
    def build_model(self):
        """Build zero-shot BiomedCLIP model"""
        model = ZeroShotBiomedCLIP(
            model_name=self.baseline_config['model_name'],
            temperature=float(self.baseline_config['temperature'])
        )
        return model.to(self.device)
    
    def build_dataloaders(self):
        """Build dataloaders"""
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
        
        return val_loader, test_loader
    
    def evaluate(self, model, loader, split_name: str):
        """Evaluate model on dataset"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_logits = []
        
        print(f"\nEvaluating on {split_name} set...")
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Evaluating"):
                images = images.to(self.device)
                
                # Zero-shot prediction
                logits, predictions = model.predict(images, self.text_descriptions)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_logits.append(logits.cpu())
        
        # Compute metrics
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            num_classes=len(self.config['dataset']['class_names'])
        )
        
        return metrics, np.concatenate(all_logits, axis=0)
    
    def run(self):
        """Main evaluation pipeline"""
        print("\n" + "="*60)
        print("BASELINE 3: ZERO-SHOT BIOMEDCLIP")
        print("="*60 + "\n")
        
        # Build model
        model = self.build_model()
        
        # Build dataloaders
        val_loader, test_loader = self.build_dataloaders()
        
        # Evaluate on validation set
        val_metrics, val_logits = self.evaluate(model, val_loader, "validation")
        
        print(f"\nValidation Results:")
        print(f"  Accuracy: {val_metrics['accuracy']:.2f}%")
        
        # Evaluate on test set
        test_metrics, test_logits = self.evaluate(model, test_loader, "test")
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"\nPer-class Test Accuracy:")
        for cls_id, acc in enumerate(test_metrics['per_class_accuracy']):
            cls_name = self.config['dataset']['class_names'][cls_id]
            support = test_metrics['per_class_support'][cls_id]
            print(f"  Class {cls_id} ({cls_name[:30]}...): {acc:.2f}% ({support} samples)")
        
        # Print detailed metrics
        print_metrics(test_metrics, list(self.config['dataset']['class_names'].values()))
        
        # Save results
        results = {
            'model': self.baseline_config['model_name'],
            'temperature': float(self.baseline_config['temperature']),
            'val_accuracy': float(val_metrics['accuracy']),
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
            },
            'text_descriptions': self.text_descriptions
        }
        
        with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {self.save_dir}/results.json")
        
        return results


if __name__ == "__main__":
    evaluator = BiomedCLIPEvaluator()
    results = evaluator.run()