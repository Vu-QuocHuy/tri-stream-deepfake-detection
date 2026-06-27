# Tri-Stream Deepfake Detection

This repository contains the implementation used for a video-level deepfake
detection model. The full model combines RGB, frequency, and SRM streams with
temporal aggregation over a fixed-length frame sequence.

The current training and evaluation code uses the convention:

- `0 = real`
- `1 = fake`
- model score = `P(fake)`

## Model Overview

The primary architecture is `TemporalMultiStreamDetector`:

- Configurable RGB, frequency, and SRM streams via `--active-streams`.
- Temporal Transformer over per-frame features.
- Binary BCE output for video-level classification.

The same training code supports single-, dual-, and tri-stream configurations.
For example, `--active-streams rgb freq` runs a dual-stream RGB+frequency
ablation, while the default `rgb freq srm` runs the full three-stream model.

Single-stream ablations are available through `TemporalSingleStreamDetector`.

## Repository Layout

```text
tri_stream_deepfake_detection/
  data/          Dataset and augmentation code
  models/        Multi-stream, temporal, and wavelet model definitions
  utils/         Metrics, logging, checkpoint, and plotting helpers
scripts/
  data/      Split-manifest and face-extraction entrypoints
  train/     Multi-stream and single-stream training entrypoints
  eval/      Multi-stream and single-stream evaluation entrypoints
```

## Installation

Create a Python environment, install PyTorch for your CUDA version, then install
the project requirements.

```bash
python3 -m venv .venv
source .venv/bin/activate

# Example CUDA 12.1 install. Adjust for your machine if needed.
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

If `facenet-pytorch` tries to replace your installed PyTorch build, install it
without dependencies:

```bash
pip install --no-deps facenet-pytorch==2.6.0
```

For development checks:

```bash
pip install -r requirements-dev.txt
ruff check .
black --check .
isort --check-only .
```

## Data Format

Training and evaluation expect extracted face crops grouped by split and class:

```text
data/extracted/
  train/
    original/
    Deepfakes/
    Face2Face/
    FaceSwap/
  val/
    original/
    Deepfakes/
    Face2Face/
    FaceSwap/
  test/
    original/
    Deepfakes/
    Face2Face/
    FaceSwap/
```

Extracted frame names should follow:

```text
{video_id}-{sample_order:06d}.jpg
```

The extraction scripts in this repository generate that naming format.

## Build a Split Manifest

If your dataset metadata is stored as CSV files with a `File Path` column, create
a reproducible split manifest:

```bash
python scripts/data/create_split_manifest_from_csv.py \
  --dataset-root /path/to/videos \
  --csv-dir /path/to/csvs \
  --output data/splits/split_manifest.json
```

## Extract Faces

Extract tracked main-face crops for a split from the manifest:

```bash
python scripts/data/extract_from_manifest.py \
  --manifest data/splits/split_manifest.json \
  --split train \
  --output-root data/extracted \
  --device cuda
```

Repeat for `val` and `test`.

## Train

Train the temporal multi-stream model:

```bash
python scripts/train/train_multistream.py \
  --config configs/temporal_multistream_b4_t16.json
```

The JSON config is a flat mapping of argparse destination names. CLI flags
provided after `--config` override config values where argparse exposes an
override flag; for one-way boolean flags such as `--amp`, edit the config file
to set the value back to `false`.

A minimal explicit command is:

```bash
python scripts/train/train_multistream.py \
  --train-real data/extracted/train/original \
  --train-fake \
    data/extracted/train/DeepFakeDetection \
    data/extracted/train/Deepfakes \
    data/extracted/train/Face2Face \
    data/extracted/train/FaceShifter \
    data/extracted/train/FaceSwap \
    data/extracted/train/NeuralTextures \
  --val-real data/extracted/val/original \
  --val-fake \
    data/extracted/val/DeepFakeDetection \
    data/extracted/val/Deepfakes \
    data/extracted/val/Face2Face \
    data/extracted/val/FaceShifter \
    data/extracted/val/FaceSwap \
    data/extracted/val/NeuralTextures \
  --output-dir outputs/temporal \
  --model efficientnet-b4 \
  --n-frames 16 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --ema-update-freq 6 \
  --phase1-epochs 8 \
  --phase2-epochs 25 \
  --amp \
  --focal-loss \
  --balanced-sampler \
  --grad-checkpoint
```

For a fuller local 16 GB GPU configuration:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 scripts/train/train_multistream.py \
  --train-real data/extracted/train/original \
  --train-fake \
    data/extracted/train/DeepFakeDetection \
    data/extracted/train/Deepfakes \
    data/extracted/train/Face2Face \
    data/extracted/train/FaceShifter \
    data/extracted/train/FaceSwap \
    data/extracted/train/NeuralTextures \
  --val-real data/extracted/val/original \
  --val-fake \
    data/extracted/val/DeepFakeDetection \
    data/extracted/val/Deepfakes \
    data/extracted/val/Face2Face \
    data/extracted/val/FaceShifter \
    data/extracted/val/FaceSwap \
    data/extracted/val/NeuralTextures \
  --output-dir outputs/temporal_b4_T16 \
  --model efficientnet-b4 \
  --freq-backbone efficientnet-b4 \
  --srm-backbone efficientnet-b4 \
  --active-streams rgb freq srm \
  --n-frames 16 \
  --sampling random \
  --num-heads 8 \
  --num-layers 2 \
  --spatial-token-grid 2 \
  --wavelet-level 3 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --num-workers 8 \
  --persistent-workers \
  --phase1-epochs 8 \
  --phase2-epochs 25 \
  --lr 3e-4 \
  --lr-backbone 5e-5 \
  --augmentation medium \
  --amp \
  --balanced-sampler \
  --focal-loss \
  --focal-gamma 2.0 \
  --label-smoothing 0.05 \
  --ema-decay 0.999 \
  --ema-update-freq 6 \
  --stream-dropout-p 0.10 \
  --aux-rgb 0.1 \
  --aux-freq 0.3 \
  --aux-srm 0.3 \
  --fusion-contrib-loss 0.5 \
  --fusion-min-freq 0.08 \
  --fusion-min-srm 0.12 \
  --temporal-dropout-p 0.10 \
  --max-temporal-drop 2 \
  --clip-jpeg-p 0.35 \
  --clip-jpeg-quality 40 85 \
  --freeze-backbone-bn \
  --best-metric auc \
  --early-stop-patience 5 \
  --early-stop-min-delta 1e-4 \
  --grad-checkpoint
```

## Evaluate

Evaluate a multi-stream checkpoint:

```bash
python scripts/eval/eval_multistream.py \
  --test-real data/extracted/test/original \
  --test-fake \
    data/extracted/test/DeepFakeDetection \
    data/extracted/test/Deepfakes \
    data/extracted/test/Face2Face \
    data/extracted/test/FaceShifter \
    data/extracted/test/FaceSwap \
    data/extracted/test/NeuralTextures \
  --checkpoint outputs/temporal/checkpoints/best_model.pth \
  --output-dir test_results \
  --n-frames 16 \
  --batch-size 4 \
  --save-predictions
```

Single-stream checkpoints should be evaluated with:

```bash
python scripts/eval/eval_single_stream.py \
  --test-real data/extracted/test/original \
  --test-fake \
    data/extracted/test/DeepFakeDetection \
    data/extracted/test/Deepfakes \
    data/extracted/test/Face2Face \
    data/extracted/test/FaceShifter \
    data/extracted/test/FaceSwap \
    data/extracted/test/NeuralTextures \
  --checkpoint outputs/single_stream/checkpoints/best_model.pth \
  --output-dir test_results_single_stream
```

## Reproducibility Notes

- Training supports flat JSON configs through `--config`. The included examples
  live under `configs/`.
- Each training run writes `run_config.json` in `--output-dir`, including argv,
  parsed args, Python/PyTorch versions, and CUDA metadata.
- Checkpoints store model metadata such as frame count, backbone, stream config,
  threshold metadata, and label convention.
- Evaluation defaults to the validation EER threshold stored in the checkpoint
  when available.
- Dependencies in `requirements.txt` are pinned. For CUDA systems, install the
  PyTorch wheel matching the local CUDA runtime before installing the rest.
- Dataset files, extracted frames, checkpoints, and experiment outputs are not
  intended to be committed to this repository.
- For paper reproduction, publish the exact split manifest, config file,
  checkpoint, and commit hash used for each reported result.

## Citation

If this repository accompanies a paper, add the paper citation here once the
metadata is finalized.
