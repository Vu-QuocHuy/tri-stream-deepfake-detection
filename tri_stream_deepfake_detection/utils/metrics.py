"""Evaluation metrics for deepfake detection."""

import numpy as np
import math
from typing import Optional, Tuple, List, Dict
from sklearn.metrics import accuracy_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)


def eval_state(probs: np.ndarray, labels: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    predict = probs >= thr
    TN = np.sum((labels == 0) & ~predict)
    FN = np.sum((labels == 1) & ~predict)
    FP = np.sum((labels == 0) & predict)
    TP = np.sum((labels == 1) & predict)
    return TN, FN, FP, TP


def calculate_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    TN, FN, FP, TP = eval_state(probs, labels, threshold)

    APCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)
    NPCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)

    ACER = (APCER + NPCER) / 2.0
    ACC = (TP + TN) / (TN + FN + FP + TP) if (TN + FN + FP + TP) > 0 else 0.0

    PRECISION_FAKE = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    PRECISION_REAL = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    RECALL_FAKE = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    RECALL_REAL = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    F1_REAL = (
        2 * (PRECISION_REAL * RECALL_REAL) / (PRECISION_REAL + RECALL_REAL)
        if (PRECISION_REAL + RECALL_REAL) > 0
        else 0.0
    )
    F1_FAKE = (
        2 * (PRECISION_FAKE * RECALL_FAKE) / (PRECISION_FAKE + RECALL_FAKE)
        if (PRECISION_FAKE + RECALL_FAKE) > 0
        else 0.0
    )
    F1_MACRO = 0.5 * (F1_REAL + F1_FAKE)
    UAR = 0.5 * (RECALL_REAL + RECALL_FAKE)  # same as balanced accuracy in binary

    # Specificity
    SPECIFICITY = RECALL_REAL

    metrics = {
        'accuracy': ACC,
        'apcer': APCER,
        'npcer': NPCER,
        'acer': ACER,
        'precision': PRECISION_FAKE,
        'recall': RECALL_FAKE,
        'f1_score': F1_FAKE,
        'precision_real': PRECISION_REAL,
        'precision_fake': PRECISION_FAKE,
        'recall_real': RECALL_REAL,
        'recall_fake': RECALL_FAKE,
        'f1_real': F1_REAL,
        'f1_fake': F1_FAKE,
        'f1_macro': F1_MACRO,
        'uar': UAR,
        'balanced_accuracy': UAR,
        'specificity': SPECIFICITY,
        'tp': int(TP),
        'tn': int(TN),
        'fp': int(FP),
        'fn': int(FN)
    }

    return metrics


def get_threshold(probs: np.ndarray, grid_density: int = 10000) -> List[float]:
    _ = probs
    thresholds = [i / float(grid_density) for i in range(grid_density + 1)]
    thresholds.append(1.1)
    return thresholds


def get_EER_states(
    probs: np.ndarray,
    labels: np.ndarray,
    grid_density: int = 10000
) -> Tuple[float, float, List[float], List[float]]:
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []

    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)

        if (FN + TP == 0):
            FRR = 1.0
            FAR = FP / float(FP + TN) if (FP + TN) > 0 else 1.0
        elif (FP + TN == 0):
            FAR = 1.0
            FRR = FN / float(FN + TP)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)

        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)

        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]

    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    optimal_thr = min_dist_states[2]

    return EER, optimal_thr, FRR_list, FAR_list


def get_HTER_at_thr(probs: np.ndarray, labels: np.ndarray, thr: float) -> float:
    TN, FN, FP, TP = eval_state(probs, labels, thr)

    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN) if (FP + TN) > 0 else 1.0
    elif (FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)

    HTER = (FAR + FRR) / 2.0
    return HTER


def calculate_comprehensive_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    preds: Optional[np.ndarray] = None,
    fixed_decision_threshold: float = 0.5,
) -> Dict[str, float]:
    _ = preds
    thr = float(fixed_decision_threshold)

    metrics = calculate_metrics(probs, labels, threshold=thr)

    EER, optimal_thr, _, _ = get_EER_states(probs, labels)
    metrics['eer'] = EER
    metrics['optimal_threshold'] = optimal_thr

    HTER = get_HTER_at_thr(probs, labels, thr)
    metrics['hter'] = HTER
    metrics['fixed_decision_threshold_used'] = thr

    optimal_preds = (probs >= optimal_thr).astype(int)
    optimal_acc = accuracy_score(labels, optimal_preds)
    metrics['accuracy_at_optimal_thr'] = optimal_acc

    try:
        auc = roc_auc_score(labels, probs)
        metrics['auc_roc'] = auc
    except ValueError:
        logger.warning("Could not calculate AUC-ROC")
        metrics['auc_roc'] = 0.0

    return metrics


DEFAULT_REPORT_KEYS = [
    "auc_roc",
    "eer",
    "balanced_accuracy",
    "accuracy",
    "f1_fake",
    "recall_fake",
    "recall_real",
    "apcer",
    "npcer",
    "acer",
    "fixed_decision_threshold_used",
    "tp",
    "tn",
    "fp",
    "fn",
]


def print_metrics(
    metrics: Dict[str, float],
    title: str = "Metrics",
    keys: Optional[List[str]] = None,
) -> None:
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    items = ((key, metrics[key]) for key in (keys or list(metrics.keys())) if key in metrics)
    for key, value in items:
        if isinstance(value, (int, np.integer)):
            print(f"{key:.<30} {value:>10d}")
        else:
            print(f"{key:.<30} {value:>10.4f}")

    print(f"{'='*60}\n")
