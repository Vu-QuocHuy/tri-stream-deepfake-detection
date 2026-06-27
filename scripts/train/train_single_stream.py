#!/usr/bin/env python3
"""Train a video-level single-stream temporal baseline."""

import argparse
import logging
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from train_multistream import (
    EMAModel,
    FocalBCELoss,
    _apply_config_defaults,
    _autocast_cuda,
    _build_dataloader,
    _get_preds_probs,
    _infer_checkpoint_phase,
    _make_grad_scaler,
    _set_global_seed,
    _write_run_config,
)
from tri_stream_deepfake_detection.data import get_train_transforms, get_val_transforms
from tri_stream_deepfake_detection.data.dataset import CombinedVideoDataset, VideoSequenceDataset
from tri_stream_deepfake_detection.models.multistream import _effnet_input_size
from tri_stream_deepfake_detection.models.temporal_single_stream import TemporalSingleStreamDetector
from tri_stream_deepfake_detection.utils import calculate_comprehensive_metrics, plot_training_history, setup_logger
from tri_stream_deepfake_detection.utils.checkpoint import torch_load_checkpoint

logger = logging.getLogger(__name__)


def _freeze_backbone_bn_stats(model: nn.Module) -> None:
    encoder = getattr(getattr(model, "frame_encoder", None), "encoder", None)
    if encoder is None:
        return
    for module in encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch: int,
    disable_tqdm: bool,
    grad_clip: float = 0.0,
    ema=None,
    grad_accum_steps: int = 1,
    freeze_backbone_bn: bool = True,
    max_nonfinite_batches: int = 3,
    ema_update_freq: int = 6,
):
    model.train()
    if freeze_backbone_bn:
        _freeze_backbone_bn_stats(model)

    running_loss = 0.0
    all_preds, all_labels = [], []
    grad_accum_steps = max(1, int(grad_accum_steps))
    ema_update_freq = max(1, int(ema_update_freq))

    optimizer.zero_grad(set_to_none=True)
    optimizer_step_idx = 0
    valid_sample_count = 0
    nonfinite_batches = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", disable=disable_tqdm)
    for iter_idx, (seqs, labels) in enumerate(pbar):
        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_target = labels.float()

        with _autocast_cuda(scaler.is_enabled()):
            logits = model(seqs)
            loss = criterion(logits, labels_target)

        loss_raw = loss.detach()
        if not torch.isfinite(loss_raw):
            nonfinite_batches += 1
            logger.warning(
                "Skipping batch %d with non-finite loss: %s (%d/%d tolerated)",
                iter_idx,
                loss_raw.item(),
                nonfinite_batches,
                max_nonfinite_batches,
            )
            optimizer.zero_grad(set_to_none=True)
            if nonfinite_batches > max_nonfinite_batches:
                raise RuntimeError("Too many non-finite training batches in one epoch.")
            continue
        loss = loss / grad_accum_steps
        should_step = ((iter_idx + 1) % grad_accum_steps == 0) or ((iter_idx + 1) == len(loader))

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if should_step:
                prev_scale = scaler.get_scale()
                scaler.unscale_(optimizer)
                max_grad_norm = grad_clip if grad_clip > 0 else float("inf")
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                if not torch.isfinite(grad_norm):
                    nonfinite_batches += 1
                    logger.warning(
                        "Skipping optimizer step at batch %d with non-finite grad norm: %s "
                        "(%d/%d tolerated)",
                        iter_idx,
                        grad_norm.item(),
                        nonfinite_batches,
                        max_nonfinite_batches,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    if nonfinite_batches > max_nonfinite_batches:
                        raise RuntimeError("Too many non-finite training batches in one epoch.")
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= prev_scale:
                    scheduler.step()
                    optimizer_step_idx += 1
                    if ema is not None and optimizer_step_idx % ema_update_freq == 0:
                        ema.update(model)
        else:
            loss.backward()
            if should_step:
                max_grad_norm = grad_clip if grad_clip > 0 else float("inf")
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                if not torch.isfinite(grad_norm):
                    nonfinite_batches += 1
                    logger.warning(
                        "Skipping optimizer step at batch %d with non-finite grad norm: %s "
                        "(%d/%d tolerated)",
                        iter_idx,
                        grad_norm.item(),
                        nonfinite_batches,
                        max_nonfinite_batches,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if nonfinite_batches > max_nonfinite_batches:
                        raise RuntimeError("Too many non-finite training batches in one epoch.")
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step_idx += 1
                if ema is not None and optimizer_step_idx % ema_update_freq == 0:
                    ema.update(model)

        running_loss += loss_raw.item() * seqs.size(0)
        valid_sample_count += seqs.size(0)
        preds, _ = _get_preds_probs(logits.detach())
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
        pbar.set_postfix({"loss": f"{loss_raw.item():.4f}"})

    if ema is not None:
        ema.update(model)

    if valid_sample_count == 0:
        raise RuntimeError("Training produced no finite batches.")
    epoch_loss = running_loss / valid_sample_count
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    epoch_rec = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch: int, disable_tqdm: bool):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", disable=disable_tqdm)
    for seqs, labels in pbar:
        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_target = labels.float()

        logits = model(seqs)
        loss = criterion(logits, labels_target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Validation produced non-finite loss: {loss.item()}")

        running_loss += loss.item() * seqs.size(0)
        _, probs = _get_preds_probs(logits)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    labels_arr = np.asarray(all_labels, dtype=np.int64)
    if labels_arr.size == 0:
        raise RuntimeError("Validation produced no samples.")
    epoch_loss = running_loss / labels_arr.size
    probs_arr = np.asarray(all_probs, dtype=np.float64)
    metrics = calculate_comprehensive_metrics(probs=probs_arr, labels=labels_arr)
    return epoch_loss, metrics, probs_arr, labels_arr


def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Train video-level single-stream temporal baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Optional flat JSON config. CLI flags override config values.")
    parser.add_argument("--train-real", nargs="+", default=None)
    parser.add_argument("--train-fake", nargs="+", default=None)
    parser.add_argument("--val-real", nargs="+", default=None)
    parser.add_argument("--val-fake", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/single_stream")

    parser.add_argument("--single-stream", type=str, default=None, choices=["rgb", "freq", "srm"])
    parser.add_argument("--model", type=str, default="efficientnet-b4")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--augmentation", type=str, default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--sampling", type=str, default="uniform", choices=["uniform", "random"])
    parser.add_argument("--temporal-dropout-p", type=float, default=0.10)
    parser.add_argument("--max-temporal-drop", type=int, default=2)
    parser.add_argument("--frame-shuffle-p", type=float, default=0.0)
    parser.add_argument("--clip-jpeg-p", type=float, default=0.35)
    parser.add_argument("--clip-jpeg-quality", type=int, nargs=2, default=(25, 85), metavar=("MIN", "MAX"))
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--srm-filters", type=int, default=30)
    parser.add_argument("--spatial-token-grid", type=int, default=2)
    parser.add_argument("--freq-mode", type=str, default="wavelet_ml", choices=["wavelet_ml"])
    parser.add_argument("--wavelet-level", type=int, default=3)
    parser.add_argument("--wavelet-type", type=str, default="db4", choices=["db4"])

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-update-freq", type=int, default=6)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-nonfinite-batches", type=int, default=3)
    parser.add_argument("--freeze-backbone-bn", dest="freeze_backbone_bn", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone-bn", dest="freeze_backbone_bn", action="store_false")

    parser.add_argument("--phase1-epochs", type=int, default=8)
    parser.add_argument("--phase2-epochs", type=int, default=25)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--best-metric", type=str, default="auc", choices=["acc", "auc", "f1", "bal_acc"])
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()
    _apply_config_defaults(parser, config_args.config)
    args = parser.parse_args()
    for required_name in ("train_real", "train_fake", "val_real", "val_fake", "single_stream"):
        if not getattr(args, required_name):
            parser.error(f"--{required_name.replace('_', '-')} is required unless provided by --config")
    _set_global_seed(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    log_dir = Path(args.output_dir) / "logs"
    results_dir = Path(args.output_dir) / "results"
    for d in (ckpt_dir, log_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name=f"single_stream_{args.single_stream}_training",
        log_file=str(log_dir / "training.log"),
        level="INFO",
    )
    _write_run_config(args, Path(args.output_dir))
    logger.info("Single-stream video-level training start")
    logger.info("Device: %s", device)
    logger.info(
        "optimization: batch=%d grad_accum=%d effective_batch=%d freeze_backbone_bn=%s ema_update_freq=%d",
        int(args.batch_size),
        int(args.grad_accum_steps),
        int(args.batch_size) * max(1, int(args.grad_accum_steps)),
        bool(args.freeze_backbone_bn),
        max(1, int(args.ema_update_freq)),
    )

    if args.resume:
        resume_meta = torch_load_checkpoint(args.resume, map_location="cpu")
        if isinstance(resume_meta, dict):
            ckpt_grid = int(resume_meta.get("spatial_token_grid", args.spatial_token_grid))
            if ckpt_grid != int(args.spatial_token_grid):
                logger.warning(
                    "Resume checkpoint uses spatial_token_grid=%d but args requested %d; using checkpoint value.",
                    ckpt_grid,
                    int(args.spatial_token_grid),
                )
                args.spatial_token_grid = ckpt_grid

    image_size = _effnet_input_size(args.model)
    train_tfm = get_train_transforms(image_size, augmentation_level=args.augmentation)
    val_tfm = get_val_transforms(image_size)

    for p in args.train_real + args.train_fake + args.val_real + args.val_fake:
        if not Path(p).exists():
            raise FileNotFoundError(f"Dataset directory not found: {p}")

    train_ds = CombinedVideoDataset(
        VideoSequenceDataset(
            [(p, -1) for p in args.train_real],
            is_real=True,
            transform=train_tfm,
            n_frames=args.n_frames,
            sampling=args.sampling,
            temporal_dropout_p=args.temporal_dropout_p,
            max_temporal_drop=args.max_temporal_drop,
            frame_shuffle_p=args.frame_shuffle_p,
            clip_jpeg_p=args.clip_jpeg_p,
            clip_jpeg_quality=tuple(args.clip_jpeg_quality),
        ),
        VideoSequenceDataset(
            [(p, -1) for p in args.train_fake],
            is_real=False,
            transform=train_tfm,
            n_frames=args.n_frames,
            sampling=args.sampling,
            temporal_dropout_p=args.temporal_dropout_p,
            max_temporal_drop=args.max_temporal_drop,
            frame_shuffle_p=args.frame_shuffle_p,
            clip_jpeg_p=args.clip_jpeg_p,
            clip_jpeg_quality=tuple(args.clip_jpeg_quality),
        ),
    )
    val_ds = CombinedVideoDataset(
        VideoSequenceDataset(
            [(p, -1) for p in args.val_real],
            is_real=True,
            transform=val_tfm,
            n_frames=args.n_frames,
            sampling="uniform",
        ),
        VideoSequenceDataset(
            [(p, -1) for p in args.val_fake],
            is_real=False,
            transform=val_tfm,
            n_frames=args.n_frames,
            sampling="uniform",
        ),
    )
    logger.info("Train: %d videos | Val: %d videos", len(train_ds), len(val_ds))

    train_sampler = None
    if args.balanced_sampler:
        labels_np = np.asarray(train_ds.labels, dtype=np.int64)
        counts_sampler = np.bincount(labels_np, minlength=2).astype(np.float64)
        counts_sampler[counts_sampler == 0] = 1.0
        weights = (1.0 / counts_sampler)[labels_np]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    pin = device.type == "cuda" and sys.platform != "win32"
    train_loader = _build_dataloader(
        dataset=train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=pin,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    val_loader = _build_dataloader(
        dataset=val_ds,
        batch_size=args.batch_size,
        sampler=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    model = TemporalSingleStreamDetector(
        backbone=args.model,
        single_stream=args.single_stream,
        n_frames=args.n_frames,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        srm_filters=args.srm_filters,
        spatial_token_grid=args.spatial_token_grid,
        freq_mode=args.freq_mode,
        wavelet_level=args.wavelet_level,
        wavelet_type=args.wavelet_type,
        pretrained=bool(args.pretrained and not args.resume),
        bce_output=True,
        use_grad_checkpoint=args.grad_checkpoint,
    ).to(device)

    if args.phase1_epochs > 0:
        model.set_phase(1)
        current_phase = 1
        logger.info("[Phase 1] frame encoder + video weighted pooling")
    else:
        model.set_phase(2)
        current_phase = 2
        logger.info("[Phase 2] temporal transformer from epoch 1")

    total, trainable = model.count_parameters()
    logger.info(
        "model=single_stream stream=%s backbone=%s T=%d token_grid=%d params total=%d trainable=%d",
        args.single_stream,
        args.model,
        int(args.n_frames),
        int(args.spatial_token_grid),
        total,
        trainable,
    )

    labels_np = np.asarray(train_ds.labels, dtype=np.int64)
    counts = np.bincount(labels_np, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    pos_weight_val = 1.0 if args.balanced_sampler else counts[0] / counts[1]
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    logger.info(
        "Class distribution: real=%d fake=%d pos_weight=%.3f",
        int(counts[0]),
        int(counts[1]),
        float(pos_weight_val),
    )

    if args.focal_loss:
        criterion = FocalBCELoss(
            gamma=args.focal_gamma,
            pos_weight=pos_weight_tensor,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    frame_params = list(model.frame_encoder.parameters())
    temporal_params = (
        list(model.temporal.parameters())
        + list(model.temporal_diff.parameters())
        + [model.spatial_token_embed]
        + list(model.frame_scorer.parameters())
        + list(model.classifier.parameters())
    )
    total_epochs = args.phase1_epochs + args.phase2_epochs

    def _build_optimizer_scheduler(phase: int):
        steps_per_epoch = max(
            1,
            (len(train_loader) + max(1, int(args.grad_accum_steps)) - 1)
            // max(1, int(args.grad_accum_steps)),
        )
        if phase == 2:
            opt = AdamW(
                [
                    {"params": frame_params, "lr": args.lr_backbone},
                    {"params": temporal_params, "lr": args.lr},
                ],
                weight_decay=1e-3,
            )
            num_steps = steps_per_epoch * max(args.phase2_epochs, 1)
            warmup = steps_per_epoch * min(3, max(args.phase2_epochs // 6, 1))
        else:
            opt = AdamW(
                [
                    {"params": [p for p in frame_params if p.requires_grad], "lr": args.lr},
                    {"params": [p for p in temporal_params if p.requires_grad], "lr": args.lr},
                ],
                weight_decay=1e-3,
            )
            num_steps = steps_per_epoch * max(args.phase1_epochs, 1)
            warmup = steps_per_epoch * min(3, max(args.phase1_epochs // 3, 1))

        sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=num_steps)
        return opt, sch

    optimizer, scheduler = _build_optimizer_scheduler(current_phase)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = _make_grad_scaler(use_amp)

    start_epoch = 0
    best_score = 0.0
    best_early_score = float("-inf")
    no_improve_epochs = 0
    if args.resume:
        logger.info("Resuming from: %s", args.resume)
        ckpt = model.load_checkpoint(args.resume, device=str(device))
        if ckpt.get("label_convention") != "real=0,fake=1" or ckpt.get("score_target") != "fake":
            raise ValueError("Checkpoint label metadata is incompatible; expected real=0,fake=1 and score_target=fake.")
        if ckpt.get("architecture") != "temporal_single_stream":
            raise ValueError(f"Checkpoint architecture is not temporal_single_stream: {ckpt.get('architecture')}")
        if ckpt.get("single_stream") and ckpt.get("single_stream") != args.single_stream:
            raise ValueError("Checkpoint single_stream does not match --single-stream.")

        resume_phase = _infer_checkpoint_phase(ckpt)
        if resume_phase != current_phase:
            model.set_phase(resume_phase)
            current_phase = resume_phase
            optimizer, scheduler = _build_optimizer_scheduler(current_phase)

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as e:
                logger.warning("Could not load optimizer state; continuing fresh. Details: %s", e)
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                logger.warning("Could not load scheduler state; continuing fresh. Details: %s", e)
        start_epoch = int(ckpt.get("epoch", 0))
        metrics = ckpt.get("metrics") or {}
        score_key = {"acc": "val_acc", "auc": "val_auc", "f1": "val_f1", "bal_acc": "val_bal_acc"}.get(
            args.best_metric,
            "val_auc",
        )
        if metrics.get(score_key) is not None:
            restored_score = float(metrics[score_key])
            restored_phase = int(metrics.get("phase", resume_phase))
            if restored_phase == 2:
                best_score = restored_score
                best_early_score = restored_score

    ema = EMAModel(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        logger.info("EMA enabled (decay=%.6f)", float(args.ema_decay))

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
    }

    for epoch in range(start_epoch + 1, total_epochs + 1):
        logger.info("Epoch %d/%d", epoch, total_epochs)

        if current_phase == 1 and epoch > args.phase1_epochs:
            model.set_phase(2)
            current_phase = 2
            optimizer, scheduler = _build_optimizer_scheduler(current_phase)
            scaler = _make_grad_scaler(use_amp)
            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                "[Phase 2 START] epoch=%d lr_backbone=%.6g lr_temporal=%.6g trainable=%d",
                epoch,
                float(args.lr_backbone),
                float(args.lr),
                trainable_now,
            )
            best_early_score = float("-inf")
            no_improve_epochs = 0

        train_loss, train_acc, train_prec, train_rec = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            disable_tqdm=args.no_tqdm,
            grad_clip=args.grad_clip,
            ema=ema,
            grad_accum_steps=args.grad_accum_steps,
            freeze_backbone_bn=args.freeze_backbone_bn,
            max_nonfinite_batches=max(0, args.max_nonfinite_batches),
            ema_update_freq=args.ema_update_freq,
        )

        if ema is not None:
            with ema.average_parameters(model):
                val_loss, val_metrics, probs, labels_arr = validate_one_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    epoch,
                    disable_tqdm=args.no_tqdm,
                )
        else:
            val_loss, val_metrics, probs, labels_arr = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                disable_tqdm=args.no_tqdm,
            )

        val_auc = float(val_metrics.get("auc_roc", 0.0))
        val_eer = float(val_metrics.get("eer", 1.0))
        eer_thr = float(val_metrics.get("optimal_threshold", 0.5))
        preds_eer = (probs >= eer_thr).astype(np.int64)
        val_acc = float(accuracy_score(labels_arr, preds_eer))
        val_f1 = float(f1_score(labels_arr, preds_eer, pos_label=1, zero_division=0))
        val_prec = float(precision_score(labels_arr, preds_eer, pos_label=1, zero_division=0))
        val_rec = float(recall_score(labels_arr, preds_eer, pos_label=1, zero_division=0))
        val_bal_acc = float(balanced_accuracy_score(labels_arr, preds_eer))

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_precision"].append(train_prec)
        history["train_recall"].append(train_rec)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        metrics = {
            "architecture": "temporal_single_stream",
            "single_stream": args.single_stream,
            "phase": current_phase,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_precision": float(train_prec),
            "train_recall": float(train_rec),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_auc": float(val_auc),
            "val_f1": float(val_f1),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "val_bal_acc": float(val_bal_acc),
            "val_eer": float(val_eer),
            "val_eer_threshold": float(eer_thr),
        }

        score = {
            "acc": val_acc,
            "auc": val_auc,
            "f1": val_f1,
            "bal_acc": val_bal_acc,
        }[args.best_metric]
        score_is_finite = bool(
            np.isfinite(float(score))
            and np.isfinite(float(train_loss))
            and np.isfinite(float(val_loss))
        )
        early_score = score
        early_improved = early_score > best_early_score + float(args.early_stop_min_delta)
        if early_improved:
            best_early_score = float(early_score)

        logger.info(
            "epoch=%d phase=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_auc=%.4f val_f1=%.4f",
            epoch,
            current_phase,
            float(train_loss),
            float(train_acc),
            float(val_loss),
            float(val_acc),
            float(val_auc),
            float(val_f1),
        )

        phase_tag = f"P{current_phase}"
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}_{phase_tag}.pth"
        model.save_checkpoint(
            str(ckpt_path),
            epoch=epoch,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            metrics=metrics,
        )

        if not score_is_finite:
            logger.warning(
                "Skipping best checkpoint update because metric/loss is non-finite "
                "(score=%s, train_loss=%s, val_loss=%s).",
                score,
                train_loss,
                val_loss,
            )

        if score_is_finite and current_phase == 2 and score > (best_score + args.early_stop_min_delta):
            best_score = float(score)
            metrics["best_metric"] = args.best_metric
            metrics["best_metric_value"] = best_score
            best_path = ckpt_dir / "best_model.pth"
            if ema is not None:
                with ema.average_parameters(model):
                    model.save_checkpoint(
                        str(best_path),
                        epoch=epoch,
                        optimizer_state=optimizer.state_dict(),
                        scheduler_state=scheduler.state_dict(),
                        metrics=metrics,
                    )
            else:
                model.save_checkpoint(
                    str(best_path),
                    epoch=epoch,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    metrics=metrics,
                )
            logger.info(
                "Saved best checkpoint: %.4f (%s, phase=2)",
                best_score,
                args.best_metric,
            )
        if current_phase == 2 and args.early_stop_patience > 0:
            if early_improved:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                logger.info(
                    "Early-stop counter: %d/%d",
                    no_improve_epochs,
                    int(args.early_stop_patience),
                )
                if no_improve_epochs >= args.early_stop_patience:
                    logger.info("Early stopping triggered.")
                    break

    try:
        plot_training_history(history, save_path=str(results_dir / "training_history.png"), show=False)
    except Exception as e:
        logger.warning("Could not plot training history: %s", e)
    logger.info("Training complete. Best %s=%.4f (Phase 2)", args.best_metric, best_score)


if __name__ == "__main__":
    main()
