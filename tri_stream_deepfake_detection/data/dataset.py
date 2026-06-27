"""Video datasets for fixed-length temporal frame sequences."""

from __future__ import annotations

import os
import glob
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from albumentations import Compose
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


def _extract_video_id(filename: str) -> str:
    """Recover the video id from an extracted frame filename."""
    stem = Path(filename).stem
    if '-' not in stem:
        return stem

    video_id, suffix = stem.rsplit('-', 1)
    if re.fullmatch(r"\d+(?:\.\d+)?", suffix) is None:
        return stem

    if not video_id:
        return stem

    return video_id


def _build_video_index(
    directory: str,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
) -> Dict[str, List[str]]:
    """Group image paths by video id."""
    video_frames: Dict[str, List[str]] = defaultdict(list)
    for ext in image_extensions:
        for fpath in glob.glob(os.path.join(directory, f'*{ext}')):
            vid_id = _extract_video_id(os.path.basename(fpath))
            video_frames[vid_id].append(fpath)

    for vid_id in video_frames:
        video_frames[vid_id].sort()

    return dict(video_frames)


class VideoSequenceDataset(Dataset):
    """Return one fixed-length frame sequence and label per video."""

    def __init__(
        self,
        data_config: List[Tuple[str, int]],
        is_real: bool = True,
        transform: Optional[Compose] = None,
        n_frames: int = 8,
        sampling: str = 'uniform',
        min_frames: int = 2,
        temporal_dropout_p: float = 0.0,
        max_temporal_drop: int = 2,
        frame_shuffle_p: float = 0.0,
        clip_jpeg_p: float = 0.0,
        clip_jpeg_quality: Tuple[int, int] = (25, 85),
    ):
        self.is_real = is_real
        self.transform = transform
        self.n_frames = n_frames
        self.sampling = sampling
        self.temporal_dropout_p = float(temporal_dropout_p)
        self.max_temporal_drop = max(1, int(max_temporal_drop))
        self.frame_shuffle_p = float(frame_shuffle_p)
        self.clip_jpeg_p = float(clip_jpeg_p)
        self.clip_jpeg_quality = (
            int(clip_jpeg_quality[0]),
            int(clip_jpeg_quality[1]),
        )
        # Project-wide label convention: 0 = real, 1 = fake/manipulated.
        self.label = 0 if is_real else 1

        self.samples: List[Tuple[str, List[str]]] = []

        for directory, max_samples in data_config:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue

            video_index = _build_video_index(directory)
            valid = {
                vid: frames
                for vid, frames in video_index.items()
                if len(frames) >= min_frames
            }

            items = [(vid, frames) for vid, frames in valid.items()]
            random.shuffle(items)

            if max_samples > 0:
                items = items[:max_samples]

            self.samples.extend(items)
            logger.info(
                f"{'Real' if is_real else 'Fake'} {directory}: "
                f"{len(items)} videos, "
                f"avg {np.mean([len(f) for _, f in items]):.1f} frames/video"
            )

        logger.info(
            f"VideoSequenceDataset ({'real' if is_real else 'fake'}): "
            f"{len(self.samples)} videos total"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _select_frames(self, frame_paths: List[str]) -> List[str]:
        n = len(frame_paths)
        if n <= self.n_frames:
            if n == 1:
                selected = frame_paths * self.n_frames
            else:
                forward = list(range(n))
                backward = list(range(n - 2, 0, -1))
                pattern = forward + backward

                indices = (pattern * ((self.n_frames // len(pattern)) + 1))[:self.n_frames]
                selected = [frame_paths[i] for i in indices]
        elif self.sampling == 'random':
            idx = sorted(random.sample(range(n), self.n_frames))
            selected = [frame_paths[i] for i in idx]
        else:
            idx = np.linspace(0, n - 1, self.n_frames).astype(int)
            selected = [frame_paths[i] for i in idx]
        return selected

    def _apply_temporal_augment(self, selected: List[str]) -> List[str]:
        """Apply sequence-level path augmentations while preserving length T."""
        selected = list(selected)
        if len(selected) <= 1:
            return selected

        if self.temporal_dropout_p > 0 and random.random() < self.temporal_dropout_p:
            n_drop = random.randint(1, min(self.max_temporal_drop, len(selected) - 1))
            drop_indices = random.sample(range(len(selected)), n_drop)
            for i in drop_indices:
                if i == 0:
                    replacement = selected[1]
                elif i == len(selected) - 1:
                    replacement = selected[-2]
                else:
                    replacement = selected[i - 1] if random.random() < 0.5 else selected[i + 1]
                selected[i] = replacement

        if self.frame_shuffle_p > 0 and random.random() < self.frame_shuffle_p:
            random.shuffle(selected)

        return selected

    def _apply_clip_jpeg(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Compress all frames with one JPEG quality to mimic clip-level codec."""
        if self.clip_jpeg_p <= 0 or random.random() >= self.clip_jpeg_p:
            return frames

        q_min, q_max = self.clip_jpeg_quality
        q_min = max(1, min(100, q_min))
        q_max = max(q_min, min(100, q_max))
        quality = random.randint(q_min, q_max)

        compressed: List[np.ndarray] = []
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        for img in frames:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, encode_param)
            if not ok:
                compressed.append(img)
                continue
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec is None:
                compressed.append(img)
            else:
                compressed.append(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))
        return compressed

    def _load_frame(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        _, frame_paths = self.samples[idx]
        selected = self._select_frames(frame_paths)
        selected = self._apply_temporal_augment(selected)

        raw_frames = [self._load_frame(fpath) for fpath in selected]
        raw_frames = self._apply_clip_jpeg(raw_frames)

        frames = []
        for img in raw_frames:
            if self.transform is not None:
                img = self.transform(image=img)['image']
            frames.append(img)

        sequence = torch.stack(frames, dim=0)
        return sequence, self.label


def create_video_dataset(
    real_config: List[Tuple[str, int]],
    fake_config: List[Tuple[str, int]],
    transform: Optional[Compose] = None,
    n_frames: int = 8,
    sampling: str = 'uniform',
    temporal_dropout_p: float = 0.0,
    max_temporal_drop: int = 2,
    frame_shuffle_p: float = 0.0,
    clip_jpeg_p: float = 0.0,
    clip_jpeg_quality: Tuple[int, int] = (25, 85),
) -> 'CombinedVideoDataset':
    """Create a combined real/fake video dataset."""
    real_ds = VideoSequenceDataset(
        real_config, is_real=True, transform=transform,
        n_frames=n_frames, sampling=sampling,
        temporal_dropout_p=temporal_dropout_p,
        max_temporal_drop=max_temporal_drop,
        frame_shuffle_p=frame_shuffle_p,
        clip_jpeg_p=clip_jpeg_p,
        clip_jpeg_quality=clip_jpeg_quality,
    )
    fake_ds = VideoSequenceDataset(
        fake_config, is_real=False, transform=transform,
        n_frames=n_frames, sampling=sampling,
        temporal_dropout_p=temporal_dropout_p,
        max_temporal_drop=max_temporal_drop,
        frame_shuffle_p=frame_shuffle_p,
        clip_jpeg_p=clip_jpeg_p,
        clip_jpeg_quality=clip_jpeg_quality,
    )
    return CombinedVideoDataset(real_ds, fake_ds)


class CombinedVideoDataset(Dataset):
    """Concatenate real and fake video datasets."""

    def __init__(self, real_ds: VideoSequenceDataset, fake_ds: VideoSequenceDataset):
        self.datasets = [real_ds, fake_ds]
        self._lengths = [len(real_ds), len(fake_ds)]
        self._offsets = [0, len(real_ds)]

        labels_real = [0] * len(real_ds)
        labels_fake = [1] * len(fake_ds)
        self.labels = labels_real + labels_fake

        logger.info(
            f"CombinedVideoDataset: {len(real_ds)} real + {len(fake_ds)} fake "
            f"= {len(self)} total videos"
        )

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        for ds, offset in zip(self.datasets, self._offsets):
            if idx < offset + len(ds):
                return ds[idx - offset]
        raise IndexError(f"Index {idx} out of range")
