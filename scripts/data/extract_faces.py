#!/usr/bin/env python3
"""Extract tracked main-face crops with MTCNN."""

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
import logging

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _choose_main_track(
    frames: List[np.ndarray],
    boxes_list,
    probs_list,
    min_detection_prob: float,
) -> Tuple[List, int, int]:
    selected = []
    previous = None
    detected_frames = 0
    multi_face_frames = 0

    for frame, boxes, probs in zip(frames, boxes_list, probs_list):
        candidates = []
        if boxes is not None:
            if probs is None:
                probs = np.ones(len(boxes), dtype=np.float32)
            for box, prob in zip(boxes, probs):
                box = np.asarray(box, dtype=np.float32)
                prob = float(prob)
                if (
                    np.isfinite(box).all()
                    and np.isfinite(prob)
                    and prob >= min_detection_prob
                    and _box_area(box) > 0
                ):
                    candidates.append((box, prob))

        if len(candidates) > 1:
            multi_face_frames += 1
        if not candidates:
            selected.append(None)
            continue

        detected_frames += 1
        image_area = float(frame.shape[0] * frame.shape[1])
        if previous is None:
            chosen = max(candidates, key=lambda item: (_box_area(item[0]), item[1]))[0]
        else:
            def score(item):
                box, prob = item
                area_score = min(1.0, _box_area(box) / max(image_area * 0.25, 1.0))
                return 0.70 * _box_iou(previous, box) + 0.20 * area_score + 0.10 * prob

            chosen = max(candidates, key=score)[0]
        previous = chosen
        selected.append(chosen)

    return selected, detected_frames, multi_face_frames


def _interpolate_missing_boxes(boxes: List) -> Tuple[List, int]:
    valid = [index for index, box in enumerate(boxes) if box is not None]
    if not valid:
        return [], len(boxes)

    filled = list(boxes)
    missing = 0
    for index, box in enumerate(filled):
        if box is not None:
            continue
        missing += 1
        left = max((item for item in valid if item < index), default=None)
        right = min((item for item in valid if item > index), default=None)
        if left is None:
            filled[index] = np.asarray(filled[right], dtype=np.float32).copy()
        elif right is None:
            filled[index] = np.asarray(filled[left], dtype=np.float32).copy()
        else:
            alpha = (index - left) / float(right - left)
            filled[index] = (
                (1.0 - alpha) * np.asarray(filled[left], dtype=np.float32)
                + alpha * np.asarray(filled[right], dtype=np.float32)
            )
    return filled, missing


def _square_crop(
    frame: np.ndarray,
    box: np.ndarray,
    margin: float,
    pad_mode: str,
) -> Optional[np.ndarray]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    side = int(math.ceil(max(x2 - x1, y2 - y1) * (1.0 + margin)))
    if side < 10:
        return None

    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    crop_x1 = int(round(center_x - side / 2.0))
    crop_y1 = int(round(center_y - side / 2.0))
    crop_x2, crop_y2 = crop_x1 + side, crop_y1 + side

    left, top = max(0, -crop_x1), max(0, -crop_y1)
    right, bottom = max(0, crop_x2 - width), max(0, crop_y2 - height)
    if left or top or right or bottom:
        border_type = cv2.BORDER_REFLECT_101 if pad_mode == 'reflect' else cv2.BORDER_CONSTANT
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, border_type)

    crop = frame[
        crop_y1 + top:crop_y2 + top,
        crop_x1 + left:crop_x2 + left,
    ]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    return crop


def _remove_existing_video_frames(output_dir: str, video_id: str) -> None:
    for old_path in Path(output_dir).glob("*.jpg"):
        stem = old_path.stem
        if "-" in stem and stem.rsplit("-", 1)[0] == video_id:
            old_path.unlink()


def _check_face_quality(
    face_crop: np.ndarray,
    min_blur_score: float = 50.0,
    min_face_pixels: int = 64,
) -> bool:
    h, w = face_crop.shape[:2]
    if h < min_face_pixels or w < min_face_pixels:
        return False
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY) if face_crop.shape[2] == 3 else face_crop
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_var >= min_blur_score


class FastMTCNN:
    def __init__(
        self,
        stride: int = 1,
        resize: float = 1.0,
        margin: int = 50,
        min_face_size: int = 100,
        thresholds: Optional[List[float]] = None,
        factor: float = 0.7,
        post_process: bool = True,
        select_largest: bool = True,
        keep_all: bool = True,
        device: str = "cuda",
        square_crop: bool = True,
        square_margin: float = 0.30,
        pad_mode: str = "reflect",
        min_detection_prob: float = 0.90,
        jpeg_quality: int = 95,
    ):
        if int(stride) != 1:
            raise ValueError("Tracked main-face extraction requires stride=1")
        self.stride = 1
        self.resize = resize
        self.square_crop = square_crop
        self.square_margin = square_margin
        self.pad_mode = pad_mode
        self.min_detection_prob = float(min_detection_prob)
        self.jpeg_quality = int(jpeg_quality)

        if MTCNN is None:
            raise RuntimeError(
                "facenet-pytorch is not installed. Install it without replacing the "
                "existing PyTorch build, e.g. pip install --no-deps facenet-pytorch==2.6.0"
            )
        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]

        self.mtcnn = MTCNN(
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            post_process=post_process,
            select_largest=select_largest,
            keep_all=keep_all,
            device=device,
        )

        logger.info("FastMTCNN initialized on %s", device)

    def __call__(self, frames: List, output_dir: str, prefix: str = "face",
                 min_blur_score: float = 0.0, min_face_pixels: int = 10) -> int:
        if self.resize != 1.0:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]

        if not frames:
            return 0

        boxes_list, probs_list = self.mtcnn.detect(frames)
        selected, detected_frames, multi_face_frames = _choose_main_track(
            frames,
            boxes_list,
            probs_list,
            min_detection_prob=self.min_detection_prob,
        )
        selected, interpolated_frames = _interpolate_missing_boxes(selected)
        if not selected:
            raise RuntimeError(f"MTCNN detected no usable face in video {prefix!r}")

        _remove_existing_video_frames(output_dir, prefix)
        saved_paths = []
        try:
            for frame_index, (frame, box) in enumerate(zip(frames, selected)):
                if self.square_crop:
                    face = _square_crop(
                        frame,
                        box,
                        margin=self.square_margin,
                        pad_mode=self.pad_mode,
                    )
                else:
                    x1, y1, x2, y2 = [int(round(value)) for value in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    face = frame[y1:y2, x1:x2]

                if face is None or face.size == 0:
                    raise RuntimeError(f"Invalid face crop at sample {frame_index} for {prefix!r}")
                if min_blur_score > 0 and not _check_face_quality(
                    face,
                    min_blur_score=min_blur_score,
                    min_face_pixels=min_face_pixels,
                ):
                    raise RuntimeError(
                        f"Face crop failed quality filter at sample {frame_index} for {prefix!r}"
                    )

                filepath = Path(output_dir) / f"{prefix}-{frame_index:06d}.jpg"
                ok = cv2.imwrite(
                    str(filepath),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if not ok or not filepath.is_file() or filepath.stat().st_size == 0:
                    raise RuntimeError(f"Failed to write face crop: {filepath}")
                saved_paths.append(filepath)
        except Exception:
            for filepath in saved_paths:
                filepath.unlink(missing_ok=True)
            raise

        logger.info(
            "%s: saved=%d detected=%d interpolated=%d multi_face=%d",
            prefix,
            len(saved_paths),
            detected_frames,
            interpolated_frames,
            multi_face_frames,
        )
        return len(saved_paths)


def process_video(
    video_path: str,
    output_dir: str,
    mtcnn: FastMTCNN,
    frame_skip: int = 30,
    frames_per_video: int = 32,
    sampling_strategy: str = "uniform",
    min_blur_score: float = 0.0,
    min_face_pixels: int = 10,
    video_id: Optional[str] = None,
) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count for video: {video_path}")

    frames = []

    uniform_indices: List[int] = []
    if sampling_strategy == "uniform" and frames_per_video > 0 and total_frames > 0:
        uniform_indices = np.rint(
            np.linspace(0, total_frames - 1, frames_per_video)
        ).astype(np.int64).tolist()
    decoded_by_index = {}
    unique_uniform_indices = set(uniform_indices)
    for frame_idx in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            break

        take = False
        if uniform_indices:
            if frame_idx in unique_uniform_indices:
                decoded_by_index[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            if frame_idx % frame_skip == 0 or frame_idx == total_frames - 1:
                take = True

        if take:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if uniform_indices:
        if not decoded_by_index:
            raise RuntimeError(f"Could not decode sampled frames from video: {video_path}")
        available = sorted(decoded_by_index)
        decode_fallbacks = 0
        frames = []
        for target in uniform_indices:
            source_index = target
            if source_index not in decoded_by_index:
                source_index = min(available, key=lambda index: abs(index - target))
                decode_fallbacks += 1
            frames.append(decoded_by_index[source_index].copy())
        if decode_fallbacks:
            logger.warning("%s: decode fallbacks=%d", video_path, decode_fallbacks)

    if not frames:
        raise RuntimeError(f"No frames selected from video: {video_path}")

    resolved_video_id = video_id or Path(video_path).stem
    saved = mtcnn(
        frames,
        output_dir,
        prefix=resolved_video_id,
        min_blur_score=min_blur_score,
        min_face_pixels=min_face_pixels,
    )
    if sampling_strategy == "uniform" and frames_per_video > 0 and saved != frames_per_video:
        raise RuntimeError(
            f"Expected {frames_per_video} crops for {resolved_video_id!r}, saved {saved}"
        )
    return len(frames), saved


def process_image(
    image_path: str,
    output_dir: str,
    mtcnn: FastMTCNN
) -> Tuple[int, int]:
    image = cv2.imread(image_path)

    if image is None:
        logger.warning("Failed to read image: %s", image_path)
        return 0, 0

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_name = Path(image_path).stem
    faces = mtcnn([image_rgb], output_dir, prefix=image_name)

    return 1, faces


def main():
    parser = argparse.ArgumentParser(
        description="Extract faces from videos and images using MTCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory containing videos/images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for extracted faces")
    parser.add_argument("--clean-output", action="store_true",
                        help="Delete the output directory before extraction")
    parser.add_argument("--mode", type=str, choices=["video", "image"], default="video",
                        help="Processing mode")
    parser.add_argument("--frame-skip", type=int, default=30,
                        help="Process every Nth frame in stride mode")
    parser.add_argument("--sampling-strategy", type=str, default="uniform", choices=["uniform", "stride"],
                        help="Video frame sampling strategy")
    parser.add_argument("--frames-per-video", type=int, default=32,
                        help="Uniform mode frame quota per video")
    parser.add_argument("--stride", type=int, default=1,
                        help="Detection stride")
    parser.add_argument("--margin", type=int, default=50,
                        help="Margin around detected face")
    parser.add_argument("--min-face-size", type=int, default=100,
                        help="Minimum face size to detect")
    parser.add_argument("--min-detection-prob", type=float, default=0.90,
                        help="Minimum MTCNN confidence for a usable face box")
    parser.add_argument("--square-crop", action="store_true", default=True,
                        help="Force square crop around detected face")
    parser.add_argument("--no-square-crop", action="store_false", dest="square_crop",
                        help="Disable square crop around detected face")
    parser.add_argument("--square-margin", type=float, default=0.30,
                        help="Extra padding fraction for square crop")
    parser.add_argument("--pad-mode", type=str, default="reflect",
                        choices=["reflect", "constant"],
                        help="Padding mode for out-of-bound crops")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality for output crops")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use")
    parser.add_argument("--min-blur-score", type=float, default=0.0,
                        help="Minimum Laplacian blur variance; 0 disables")
    parser.add_argument("--min-face-pixels", type=int, default=64,
                        help="Minimum face width/height in pixels")

    args = parser.parse_args()

    if args.frames_per_video < 0:
        parser.error("--frames-per-video must be >= 0")
    if args.sampling_strategy == "uniform" and args.frames_per_video <= 0:
        parser.error("--frames-per-video must be > 0 when using uniform sampling")
    if args.frame_skip <= 0:
        parser.error("--frame-skip must be > 0")
    if args.stride != 1:
        parser.error("tracked main-face extraction requires --stride 1")
    if not 0.0 <= args.min_detection_prob <= 1.0:
        parser.error("--min-detection-prob must be in [0, 1]")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")

    if args.clean_output and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    mtcnn = FastMTCNN(
        stride=args.stride,
        margin=args.margin,
        min_face_size=args.min_face_size,
        device=device,
        square_crop=args.square_crop,
        square_margin=args.square_margin,
        pad_mode=args.pad_mode,
        min_detection_prob=args.min_detection_prob,
        jpeg_quality=args.jpeg_quality,
    )

    if args.mode == "video":
        patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    else:
        patterns = ["*.jpg", "*.jpeg", "*.png"]

    input_files = []
    for pattern in patterns:
        input_files.extend(glob.glob(os.path.join(args.input_dir, pattern)))

    input_files = sorted(set(input_files))
    if not input_files:
        logger.error("No files found in %s", args.input_dir)
        return

    logger.info("Found %d files to process", len(input_files))

    stem_counts = {}
    for filepath in input_files:
        stem = Path(filepath).stem
        stem_counts[stem] = stem_counts.get(stem, 0) + 1

    def output_video_id(filepath: str) -> str:
        stem = Path(filepath).stem
        if stem_counts[stem] == 1:
            return stem
        digest = hashlib.sha1(str(Path(filepath).resolve()).encode("utf-8")).hexdigest()[:8]
        return f"{stem}_{digest}"

    total_frames = 0
    total_faces = 0
    extraction_records = []
    extraction_errors = []

    for filepath in tqdm(input_files, desc="Processing files"):
        video_id = output_video_id(filepath)
        try:
            if args.mode == "video":
                frames, faces = process_video(
                    video_path=filepath,
                    output_dir=args.output_dir,
                    mtcnn=mtcnn,
                    frame_skip=args.frame_skip,
                    frames_per_video=args.frames_per_video,
                    sampling_strategy=args.sampling_strategy,
                    min_blur_score=args.min_blur_score,
                    min_face_pixels=args.min_face_pixels,
                    video_id=video_id,
                )
            else:
                frames, faces = process_image(
                    filepath,
                    args.output_dir,
                    mtcnn,
                )

            total_frames += frames
            total_faces += faces
            extraction_records.append({
                "input": filepath,
                "video_id": video_id,
                "sampled_frames": int(frames),
                "saved_frames": int(faces),
                "status": "ok",
            })

        except Exception as exc:
            logger.error("Error processing %s: %s", filepath, exc)
            record = {
                "input": filepath,
                "video_id": video_id,
                "sampled_frames": 0,
                "saved_frames": 0,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            extraction_records.append(record)
            extraction_errors.append(record)

    summary_path = Path(args.output_dir) / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(extraction_records, handle, indent=2, ensure_ascii=False)
    logger.info("Processing complete")
    logger.info("Total frames processed: %d", total_frames)
    logger.info("Total faces detected: %d", total_faces)
    logger.info("Faces saved to: %s", args.output_dir)
    logger.info("Summary: %s", summary_path)
    if extraction_errors:
        raise RuntimeError(
            f"Extraction failed for {len(extraction_errors)} file(s); see {summary_path}"
        )


if __name__ == "__main__":
    main()
