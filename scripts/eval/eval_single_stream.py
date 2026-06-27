#!/usr/bin/env python3
"""Evaluate a single-stream checkpoint baseline."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from tri_stream_deepfake_detection.data import get_val_transforms
from tri_stream_deepfake_detection.data.dataset import CombinedVideoDataset, VideoSequenceDataset
from tri_stream_deepfake_detection.models.multistream import _effnet_input_size
from tri_stream_deepfake_detection.models.temporal_single_stream import TemporalSingleStreamDetector
from tri_stream_deepfake_detection.utils import (
    calculate_comprehensive_metrics,
    DEFAULT_REPORT_KEYS,
    plot_confusion_matrix,
    plot_roc_curve,
    print_metrics,
    setup_logger,
)
from tri_stream_deepfake_detection.utils.checkpoint import torch_load_checkpoint
from tri_stream_deepfake_detection.utils.metrics import get_EER_states


@torch.no_grad()
def evaluate(model: TemporalSingleStreamDetector, loader: DataLoader,
             device: torch.device, disable_tqdm: bool):
    model.eval()
    all_probs, all_labels = [], []
    for seqs, labels in tqdm(loader, desc="Testing", disable=disable_tqdm):
        seqs = seqs.to(device, non_blocking=True)
        logits = model(seqs)
        probs = torch.sigmoid(logits.reshape(-1)).cpu().numpy()
        all_probs.extend(probs.astype(np.float64))
        all_labels.extend(labels.numpy())
    return np.asarray(all_probs), np.asarray(all_labels, dtype=np.int64)


def _build_test_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    sig = inspect.signature(DataLoader.__init__).parameters
    if num_workers > 0 and "prefetch_factor" in sig:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate single-stream baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-real", nargs="+", required=True)
    parser.add_argument("--test-fake", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="test_results_single_stream")
    parser.add_argument("--model", type=str, default="efficientnet-b4")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Disable tqdm progress bar.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "val_eer"],
        default="val_eer",
        help="Use the validation EER threshold by default; fixed is an explicit override.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--single-stream",
        type=str,
        default=None,
        choices=["rgb", "freq", "srm"],
        help="Optional consistency check against the checkpoint stream.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="eval_single_stream",
        log_file=str(out_dir / "eval.log"),
        level="INFO",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ckpt = torch_load_checkpoint(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        logger.error("Checkpoint is not a metadata dictionary; cannot evaluate safely.")
        sys.exit(1)
    label_convention = ckpt.get("label_convention")
    score_target = ckpt.get("score_target")
    if label_convention != "real=0,fake=1" or score_target != "fake":
        logger.error(
            "Checkpoint label metadata is missing or incompatible "
            "(label_convention=%s, score_target=%s). Current code expects real=0, fake=1, score=P(fake).",
            label_convention,
            score_target,
        )
        sys.exit(1)
    backbone = ckpt.get("backbone", args.model)
    n_frames = ckpt.get("n_frames", args.n_frames)
    num_heads = ckpt.get("num_heads", 8)
    num_layers = ckpt.get("num_layers", 2)
    srm_filters = ckpt.get("srm_filters", 30)
    dropout = ckpt.get("dropout", 0.1)
    checkpoint_bce_output = ckpt.get("bce_output")
    if checkpoint_bce_output is not True:
        logger.error("Checkpoint must use BCE output; got bce_output=%s.", checkpoint_bce_output)
        sys.exit(1)
    architecture = ckpt.get("architecture")
    ckpt_stream = ckpt.get("single_stream")
    if architecture != "temporal_single_stream":
        logger.error(
            "Checkpoint architecture '%s' is not supported by this script. "
            "Expected temporal_single_stream.",
            architecture,
        )
        sys.exit(1)

    if ckpt_stream not in ("rgb", "freq", "srm"):
        logger.error("Checkpoint single_stream invalid.")
        sys.exit(1)
    if args.single_stream is not None and args.single_stream != ckpt_stream:
        logger.error("--single-stream does not match checkpoint.")
        sys.exit(1)
    stream = ckpt_stream

    logger.info("backbone=%s n_frames=%s single_stream=%s", backbone, n_frames, stream)

    threshold_mode = args.threshold_mode
    thr_eff = float(args.threshold)
    if threshold_mode == "val_eer":
        vm = ckpt.get("metrics") or {}
        thr_key = "val_eer_threshold"
        vo = vm.get(thr_key)
        if vo is not None:
            thr_eff = float(vo)
            logger.info("decision_threshold=%.4f (checkpoint %s)", thr_eff, thr_key)
        else:
            logger.warning(
                "Checkpoint has no EER threshold; using --threshold=%s",
                args.threshold,
            )

    image_size = _effnet_input_size(backbone)
    tfm = get_val_transforms(image_size)
    test_ds = CombinedVideoDataset(
        VideoSequenceDataset(
            [(p, -1) for p in args.test_real], is_real=True, transform=tfm,
            n_frames=n_frames, sampling="uniform",
        ),
        VideoSequenceDataset(
            [(p, -1) for p in args.test_fake], is_real=False, transform=tfm,
            n_frames=n_frames, sampling="uniform",
        ),
    )
    logger.info("Test set: %d videos", len(test_ds))
    if len(test_ds) == 0:
        logger.error("No test videos found. Check --test-real / --test-fake and extracted frame names.")
        sys.exit(1)

    loader = _build_test_dataloader(
        dataset=test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TemporalSingleStreamDetector(
        backbone=backbone,
        single_stream=stream,
        n_frames=n_frames,
        num_heads=num_heads,
        num_layers=num_layers,
        srm_filters=srm_filters,
        spatial_token_grid=ckpt.get("spatial_token_grid", 2),
        bce_output=True,
        dropout=dropout,
        pretrained=False,
        freq_mode=ckpt.get("freq_mode", "wavelet_ml"),
        wavelet_level=ckpt.get("wavelet_level", 1),
        wavelet_type=ckpt.get("wavelet_type", "db4"),
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model = model.to(device)
    model.set_phase(2)

    probs_fake, labels = evaluate(model, loader, device, disable_tqdm=args.no_tqdm)
    pred_labels = (probs_fake >= thr_eff).astype(np.int64)
    metrics = calculate_comprehensive_metrics(
        probs=probs_fake,
        labels=labels,
        fixed_decision_threshold=thr_eff,
    )
    print_metrics(metrics, title=f"Single-stream ({stream})", keys=DEFAULT_REPORT_KEYS)

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"single_stream: {stream}\n")
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"backbone: {backbone}\n")
        f.write(f"n_frames: {n_frames}\n")
        f.write(f"test videos: {len(test_ds)}\n")
        f.write(f"threshold_used: {thr_eff}\n")
        f.write(f"threshold_mode: {threshold_mode}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    plot_confusion_matrix(
        confusion_matrix(labels, pred_labels),
        class_names=["Real", "Fake"],
        title=f"Confusion ({stream})",
        save_path=str(out_dir / "confusion_matrix.png"),
        show=False,
    )
    EER, optimal_thr, FRR_list, FAR_list = get_EER_states(probs_fake, labels)
    plot_roc_curve(
        FRR_list, FAR_list, EER,
        title=f"ROC EER={EER:.4f}",
        save_path=str(out_dir / "roc_curve.png"),
        show=False,
    )
    if args.save_predictions:
        predictions_path = out_dir / "predictions.csv"
        pd.DataFrame({
            "true_label": labels,
            "pred_label": pred_labels,
            "prob_real": 1.0 - probs_fake,
            "prob_fake": probs_fake,
        }).to_csv(predictions_path, index=False)
        logger.info("Predictions saved to: %s", predictions_path)

    logger.info("Evaluation complete. Results: %s", out_dir)


if __name__ == "__main__":
    main()
