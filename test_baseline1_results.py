"""
Quick test script for Baseline 1 results
"""

import torch
import torch.nn as nn
from src.training.train_baseline_transfer import TransferLearningTrainer


def main():
    """Main testing function"""
    print("\n" + "="*60)
    print("BASELINE 1: TEST RESULTS")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = TransferLearningTrainer()
    
    # Build model
    model = trainer.build_model()
    
    # Build dataloaders (will use num_workers=0 for testing)
    print("Loading test dataset...")
    from src.data.standard_dataset import StandardDermaMNIST
    from torch.utils.data import DataLoader
    
    test_dataset = StandardDermaMNIST(
        split='test',
        config_path=trainer.config_path,
        augment=False
    )
    
    # Use num_workers=0 for Windows compatibility
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0 for Windows
        pin_memory=True
    )
    
    print(f"Test set loaded: {len(test_dataset)} samples\n")
    
    # Load best checkpoint
    print("Loading best model checkpoint...")
    checkpoint = torch.load(
        'experiments/baseline/transfer_learning/best_model.pth',
        map_location=trainer.device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"✓ Validation accuracy: {checkpoint['val_acc']:.2f}%\n")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(trainer.device)
            labels = labels.to(trainer.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    from src.utils.metrics import compute_metrics
    import numpy as np
    
    test_metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        num_classes=7
    )
    
    # Print results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60 + "\n")
    
    print(f"Overall Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Macro Precision: {test_metrics['macro_precision']:.2f}%")
    print(f"Macro Recall: {test_metrics['macro_recall']:.2f}%")
    print(f"Macro F1-Score: {test_metrics['macro_f1']:.2f}%")
    
    print(f"\nPer-Class Results:")
    print(f"{'Class':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    class_names = trainer.config['dataset']['class_names']
    for cls_id in range(7):
        cls_name = class_names[cls_id][:30]
        acc = test_metrics['per_class_accuracy'][cls_id]
        prec = test_metrics['per_class_precision'][cls_id]
        rec = test_metrics['per_class_recall'][cls_id]
        f1 = test_metrics['per_class_f1'][cls_id]
        support = test_metrics['per_class_support'][cls_id]
        
        # Highlight minority classes
        marker = " ⚠️" if cls_id in [3, 6] else ""
        
        print(f"{cls_id:<8} {acc:>10.2f}% {prec:>10.2f}% {rec:>10.2f}% {f1:>10.2f}% {support:>9}{marker}")
    
    print("\n⚠️  = Minority class (<100 training samples)")
    
    # Calculate minority vs majority performance
    minority_acc = np.mean([test_metrics['per_class_accuracy'][3], 
                            test_metrics['per_class_accuracy'][6]])
    majority_acc = np.mean([test_metrics['per_class_accuracy'][i] 
                           for i in [0, 1, 2, 4, 5]])
    
    print(f"\nMajority Classes Avg Accuracy: {majority_acc:.2f}%")
    print(f"Minority Classes Avg Accuracy: {minority_acc:.2f}%")
    print(f"Performance Gap: {majority_acc - minority_acc:.2f}%")
    
    # Save results
    import json
    
    results = {
        'best_epoch': int(checkpoint['epoch'] + 1),
        'best_val_accuracy': float(checkpoint['val_acc']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_macro_precision': float(test_metrics['macro_precision']),
        'test_macro_recall': float(test_metrics['macro_recall']),
        'test_macro_f1': float(test_metrics['macro_f1']),
        'per_class_accuracy': test_metrics['per_class_accuracy'].tolist(),
        'per_class_precision': test_metrics['per_class_precision'].tolist(),
        'per_class_recall': test_metrics['per_class_recall'].tolist(),
        'per_class_f1': test_metrics['per_class_f1'].tolist(),
        'per_class_support': test_metrics['per_class_support'].tolist(),
        'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
        'minority_avg_accuracy': float(minority_acc),
        'majority_avg_accuracy': float(majority_acc)
    }
    
    output_path = 'experiments/baseline/transfer_learning/test_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()