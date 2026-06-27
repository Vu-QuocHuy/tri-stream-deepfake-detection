#!/usr/bin/env python3
"""Extract face crops from videos listed in a split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List

import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from extract_faces import FastMTCNN, process_video


def _load_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_quota(class_name: str, real_class: str, real_quota: int, fake_quota: int) -> int:
    return real_quota if class_name == real_class else fake_quota


def main():
    parser = argparse.ArgumentParser(
        description="Extract face crops using split manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to split manifest JSON")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True, help="Split to extract")
    parser.add_argument("--output-root", type=str, required=True, help="Root output folder for extracted faces")
    parser.add_argument("--real-class", type=str, default="original", help="Name of the real class")
    parser.add_argument("--real-frames-per-video", type=int, default=32, help="Frames per real video")
    parser.add_argument("--fake-frames-per-video", type=int, default=32, help="Frames per fake video")
    parser.add_argument("--sampling-strategy", type=str, default="uniform", choices=["uniform", "stride"], help="Sampling strategy")
    parser.add_argument("--frame-skip", type=int, default=30, help="Stride mode: process every Nth frame")
    parser.add_argument("--stride", type=int, default=1, help="MTCNN detect stride")
    parser.add_argument("--margin", type=int, default=50, help="MTCNN margin")
    parser.add_argument("--min-face-size", type=int, default=100, help="MTCNN minimum face size")
    parser.add_argument("--min-detection-prob", type=float, default=0.90, help="Minimum usable MTCNN confidence")
    parser.add_argument("--square-margin", type=float, default=0.30, help="Square crop margin fraction")
    parser.add_argument("--pad-mode", type=str, default="reflect", choices=["reflect", "constant"], help="Padding mode for square crop")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Output JPEG quality")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    args = parser.parse_args()

    if args.real_frames_per_video <= 0 or args.fake_frames_per_video <= 0:
        parser.error("frame quotas must be > 0")
    if args.frame_skip <= 0:
        parser.error("--frame-skip must be > 0")
    if args.stride != 1:
        parser.error("tracked main-face extraction requires --stride 1")
    if not 0.0 <= args.min_detection_prob <= 1.0:
        parser.error("--min-detection-prob must be in [0, 1]")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    manifest = _load_manifest(manifest_path)
    split_data: Dict[str, List[str]] = manifest["data"][args.split]

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mtcnn = FastMTCNN(
        stride=args.stride,
        margin=args.margin,
        min_face_size=args.min_face_size,
        device=args.device,
        square_crop=True,
        square_margin=args.square_margin,
        pad_mode=args.pad_mode,
        min_detection_prob=args.min_detection_prob,
        jpeg_quality=args.jpeg_quality,
    )

    grand_total_frames = 0
    grand_total_faces = 0
    extraction_records = []
    extraction_errors = []

    classes = sorted(split_data.keys())
    for class_name in classes:
        video_paths = split_data[class_name]
        class_out = output_root / args.split / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        frames_per_video = _resolve_quota(
            class_name,
            real_class=args.real_class,
            real_quota=args.real_frames_per_video,
            fake_quota=args.fake_frames_per_video,
        )
        print(
            f"\n[{args.split}] class={class_name} videos={len(video_paths)} "
            f"frames_per_video={frames_per_video}"
        )

        stem_counts = {}
        for video_path in video_paths:
            stem = Path(video_path).stem
            stem_counts[stem] = stem_counts.get(stem, 0) + 1

        def video_id_for(video_path: str) -> str:
            stem = Path(video_path).stem
            if stem_counts[stem] == 1:
                return stem
            digest = hashlib.sha1(str(Path(video_path).resolve()).encode("utf-8")).hexdigest()[:8]
            return f"{stem}_{digest}"

        for video_path in tqdm(video_paths, desc=f"{args.split}:{class_name}"):
            video_id = video_id_for(video_path)
            try:
                frames, faces = process_video(
                    video_path=video_path,
                    output_dir=str(class_out),
                    mtcnn=mtcnn,
                    frame_skip=args.frame_skip,
                    frames_per_video=frames_per_video,
                    sampling_strategy=args.sampling_strategy,
                    video_id=video_id,
                )
                grand_total_frames += frames
                grand_total_faces += faces
                record = {
                    "video": video_path,
                    "video_id": video_id,
                    "class": class_name,
                    "requested_frames": frames_per_video,
                    "sampled_frames": frames,
                    "saved_frames": faces,
                    "status": "ok",
                }
                extraction_records.append(record)
            except Exception as exc:
                print(f"Error processing {video_path}: {exc}")
                record = {
                    "video": video_path,
                    "video_id": video_id,
                    "class": class_name,
                    "requested_frames": frames_per_video,
                    "sampled_frames": 0,
                    "saved_frames": 0,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                extraction_records.append(record)
                extraction_errors.append(record)

    summary_path = output_root / f"extraction_summary_{args.split}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(extraction_records, handle, indent=2, ensure_ascii=False)

    print("\nExtraction complete.")
    print(f"Total sampled frames: {grand_total_frames}")
    print(f"Total extracted faces: {grand_total_faces}")
    print(f"Saved to: {output_root}")
    print(f"Summary: {summary_path}")

    if extraction_errors:
        raise RuntimeError(
            f"Extraction failed for {len(extraction_errors)} video(s); see {summary_path}"
        )


if __name__ == "__main__":
    main()
