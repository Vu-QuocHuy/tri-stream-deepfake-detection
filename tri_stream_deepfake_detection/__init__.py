"""Tri-stream deepfake detection models and utilities."""

__version__ = "2.0.0"

from tri_stream_deepfake_detection.models import (
    MultiStreamDeepFakeDetector,
    TemporalMultiStreamDetector,
    TemporalSingleStreamDetector,
)
from tri_stream_deepfake_detection.data import (
    CombinedVideoDataset,
    VideoSequenceDataset,
    get_train_transforms,
    get_val_transforms,
)
from tri_stream_deepfake_detection.utils import setup_logger, calculate_metrics

__all__ = [
    "MultiStreamDeepFakeDetector",
    "TemporalMultiStreamDetector",
    "TemporalSingleStreamDetector",
    "VideoSequenceDataset",
    "CombinedVideoDataset",
    "get_train_transforms",
    "get_val_transforms",
    "setup_logger",
    "calculate_metrics",
]
