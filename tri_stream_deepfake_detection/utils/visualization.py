"""Visualization utilities for evaluation and training."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = ['Real', 'Fake'],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=['Predicted ' + cn for cn in class_names],
        yticklabels=class_names,
        linewidths=2,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_roc_curve(
    FRR_list: List[float],
    FAR_list: List[float],
    eer: float,
    title: str = 'ROC Curve (FAR vs FRR)',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    plt.figure(figsize=(10, 8))

    plt.plot(FAR_list, FRR_list, marker='.', label=f'ROC Curve (EER={eer:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier', alpha=0.5)

    eer_idx = np.argmin(np.abs(np.array(FAR_list) - np.array(FRR_list)))
    plt.plot(FAR_list[eer_idx], FRR_list[eer_idx], 'ro', markersize=10, label='EER Point')

    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ['loss', 'accuracy'],
    title: str = 'Training History',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(8 * num_metrics, 6))

    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 'b-o', label=f'Train {metric.capitalize()}', linewidth=2)

        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 'r-s', label=f'Val {metric.capitalize()}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
