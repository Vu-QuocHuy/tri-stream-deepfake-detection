#!/usr/bin/env python3
"""Evaluate a temporal multi-stream checkpoint."""

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
from tri_stream_deepfake_detection.models.temporal import TemporalMultiStreamDetector
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
def evaluate(model: TemporalMultiStreamDetector, loader: DataLoader,
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
        description="Evaluate TemporalMultiStreamDetector on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--test-real",   nargs="+", required=True)
    parser.add_argument("--test-fake",   nargs="+", required=True)
    parser.add_argument("--checkpoint",  type=str,  required=True)
    parser.add_argument("--output-dir",  type=str,  default="test_results")
    parser.add_argument("--model",       type=str,  default="efficientnet-b4")
    parser.add_argument("--n-frames",    type=int,  default=16)
    parser.add_argument("--batch-size",  type=int,  default=4)
    parser.add_argument("--num-workers", type=int,  default=4)
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Disable tqdm progress bar (useful for nohup/screen logs).")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Decision threshold on prob(fake) when --threshold-mode=fixed")
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="val_eer",
        choices=["fixed", "val_eer"],
        help="Use the validation EER threshold by default; fixed is an explicit override.",
    )
    parser.add_argument("--save-predictions", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="eval_multistream",
        log_file=str(out_dir / "eval.log"),
        level="INFO",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch_load_checkpoint(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        logger.error("Checkpoint is not a metadata dictionary; cannot evaluate safely.")
        sys.exit(1)
    architecture = ckpt.get("architecture")
    if (
        architecture != "temporal_multi_stream"
        or ckpt.get("spatial_forward") == "single_stream"
        or ckpt.get("single_stream") in ("rgb", "freq", "srm")
    ):
        logger.error(
            "Checkpoint architecture '%s' is not compatible with scripts/eval/eval_multistream.py. "
            "Use scripts/eval/eval_single_stream.py for temporal_single_stream checkpoints.",
            architecture,
        )
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
    backbone  = ckpt.get("backbone",  args.model)
    freq_backbone = ckpt.get("freq_backbone", backbone)
    srm_backbone = ckpt.get("srm_backbone", backbone)
    n_frames  = ckpt.get("n_frames",  args.n_frames)
    num_heads = ckpt.get("num_heads", 8)
    num_layers = ckpt.get("num_layers", 2)
    srm_filters = ckpt.get("srm_filters", 30)
    dropout = ckpt.get("dropout", 0.1)
    if ckpt.get("bce_output") is not True:
        logger.error("Checkpoint must use BCE output; got bce_output=%s.", ckpt.get("bce_output"))
        sys.exit(1)
    spatial_token_grid = ckpt.get("spatial_token_grid", 1)
    active_streams = ckpt.get("active_streams", ["rgb", "freq", "srm"])
    logger.info(
        "rgb_backbone=%s freq_backbone=%s srm_backbone=%s n_frames=%s "
        "layers=%s heads=%s srm_filters=%s token_grid=%s active_streams=%s",
        backbone,
        freq_backbone,
        srm_backbone,
        n_frames,
        num_layers,
        num_heads,
        srm_filters,
        spatial_token_grid,
        ",".join(active_streams),
    )

    threshold_mode = args.threshold_mode
    thr_eff = float(args.threshold)
    if threshold_mode == "val_eer":
        vm = ckpt.get("metrics") or {}
        thr_key = "val_eer_threshold"
        vo = vm.get(thr_key)
        if vo is not None:
            thr_eff = float(vo)
            logger.info(
                "decision_threshold=%.4f (mode=%s, from checkpoint metrics.%s)",
                thr_eff, threshold_mode, thr_key
            )
        else:
            logger.warning(
                "threshold mode '%s' requested but checkpoint has no metrics.%s; "
                "using fixed --threshold=%s",
                threshold_mode, thr_key, args.threshold
            )

    image_size = _effnet_input_size(backbone)
    tfm = get_val_transforms(image_size)

    test_real_cfg = [(p, -1) for p in args.test_real]
    test_fake_cfg = [(p, -1) for p in args.test_fake]

    test_ds = CombinedVideoDataset(
        VideoSequenceDataset(test_real_cfg, is_real=True,  transform=tfm,
                             n_frames=n_frames, sampling="uniform"),
        VideoSequenceDataset(test_fake_cfg, is_real=False, transform=tfm,
                             n_frames=n_frames, sampling="uniform"),
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

    freq_mode = ckpt.get("freq_mode", "wavelet_ml")
    wavelet_level = ckpt.get("wavelet_level", 1)
    wavelet_type = ckpt.get("wavelet_type", "db4")
    logger.info(
        "freq_mode=%s wavelet_level=%s wavelet_type=%s",
        freq_mode,
        wavelet_level,
        wavelet_type,
    )

    model = TemporalMultiStreamDetector(
        backbone=backbone,
        freq_backbone=freq_backbone,
        srm_backbone=srm_backbone,
        n_frames=n_frames,
        num_heads=num_heads,
        num_layers=num_layers,
        srm_filters=srm_filters,
        spatial_token_grid=spatial_token_grid,
        bce_output=True,
        dropout=dropout,
        pretrained=False,
        freq_mode=freq_mode,
        wavelet_level=wavelet_level,
        wavelet_type=wavelet_type,
        active_streams=active_streams,
    )
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    probs_fake, labels = evaluate(model, loader, device, disable_tqdm=args.no_tqdm)

    pred_labels = (probs_fake >= thr_eff).astype(np.int64)

    metrics = calculate_comprehensive_metrics(
        probs=probs_fake,
        labels=labels,
        fixed_decision_threshold=thr_eff,
    )
    print_metrics(metrics, title="Test Results", keys=DEFAULT_REPORT_KEYS)

    metrics_file = out_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("TemporalMultiStreamDetector - Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"checkpoint  : {args.checkpoint}\n")
        f.write(f"backbone    : {backbone}\n")
        f.write(f"active_streams : {','.join(active_streams)}\n")
        f.write(f"n_frames    : {n_frames}\n")
        f.write(f"test videos : {len(test_ds)}\n")
        f.write(f"threshold_used : {thr_eff}\n")
        f.write(f"threshold_mode : {threshold_mode}\n")
        f.write("\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Metrics saved to: %s", metrics_file)

    conf_mat = confusion_matrix(labels, pred_labels)
    plot_confusion_matrix(
        conf_mat,
        class_names=["Real", "Fake"],
        title="Test Confusion Matrix",
        save_path=str(out_dir / "confusion_matrix.png"),
        show=False,
    )

    EER, optimal_thr, FRR_list, FAR_list = get_EER_states(probs_fake, labels)
    plot_roc_curve(
        FRR_list, FAR_list, EER,
        title=f"ROC Curve (EER={EER:.4f})",
        save_path=str(out_dir / "roc_curve.png"),
        show=False,
    )
    logger.info("EER=%.4f optimal_threshold=%.4f", EER, optimal_thr)

    if args.save_predictions:
        df = pd.DataFrame({
            "true_label":  labels,
            "pred_label":  pred_labels,
            "prob_real":   1.0 - probs_fake,
            "prob_fake":   probs_fake,
        })
        df.to_csv(out_dir / "predictions.csv", index=False)
        logger.info("Predictions saved to: %s", out_dir / "predictions.csv")

    logger.info("Evaluation complete. Results: %s", out_dir)


if __name__ == "__main__":
    main()
