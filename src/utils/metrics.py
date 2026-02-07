"""
Evaluation metrics for classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, List


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> Dict:
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Per-class accuracy
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1) * 100
    per_class_acc = np.nan_to_num(per_class_acc)  # Handle divide by zero
    
    # Macro averages
    macro_precision = np.mean(precision) * 100
    macro_recall = np.mean(recall) * 100
    macro_f1 = np.mean(f1) * 100
    
    metrics = {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'per_class_precision': precision * 100,
        'per_class_recall': recall * 100,
        'per_class_f1': f1 * 100,
        'per_class_support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': conf_matrix
    }
    
    return metrics


def print_metrics(metrics: Dict, class_names: List[str]):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary from compute_metrics()
        class_names: List of class names
    """
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Macro Precision: {metrics['macro_precision']:.2f}%")
    print(f"  Macro Recall: {metrics['macro_recall']:.2f}%")
    print(f"  Macro F1: {metrics['macro_f1']:.2f}%")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<40} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Support':<8}")
    print("-" * 90)
    
    for i, name in enumerate(class_names):
        print(f"{name[:37]:<40} "
              f"{metrics['per_class_accuracy'][i]:>6.2f}% "
              f"{metrics['per_class_precision'][i]:>6.2f}% "
              f"{metrics['per_class_recall'][i]:>6.2f}% "
              f"{metrics['per_class_f1'][i]:>6.2f}% "
              f"{metrics['per_class_support'][i]:>7}")


# Test
if __name__ == "__main__":
    # Dummy data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1, 2])
    
    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    
    class_names = ["Class 0", "Class 1", "Class 2"]
    print_metrics(metrics, class_names)