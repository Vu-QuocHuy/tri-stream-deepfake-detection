"""Model definitions for deepfake detection."""

from tri_stream_deepfake_detection.models.multistream import MultiStreamDeepFakeDetector
from tri_stream_deepfake_detection.models.temporal import TemporalMultiStreamDetector
from tri_stream_deepfake_detection.models.temporal_single_stream import TemporalSingleStreamDetector

__all__ = [
    "MultiStreamDeepFakeDetector",
    "TemporalMultiStreamDetector",
    "TemporalSingleStreamDetector",
]
