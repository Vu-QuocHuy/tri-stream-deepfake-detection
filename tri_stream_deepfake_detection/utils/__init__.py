"""Utility functions for training, evaluation, and visualization."""

from tri_stream_deepfake_detection.utils.metrics import (
    calculate_metrics,
    get_EER_states,
    get_HTER_at_thr,
    eval_state,
    calculate_comprehensive_metrics,
    DEFAULT_REPORT_KEYS,
    print_metrics,
)
from tri_stream_deepfake_detection.utils.logger import setup_logger, get_logger
from tri_stream_deepfake_detection.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history
)

__all__ = [
    "calculate_metrics",
    "get_EER_states",
    "get_HTER_at_thr",
    "eval_state",
    "calculate_comprehensive_metrics",
    "DEFAULT_REPORT_KEYS",
    "print_metrics",
    "setup_logger",
    "get_logger",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
]
