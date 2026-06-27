#!/usr/bin/env python3
"""Train a video-level temporal multi-stream detector."""

import argparse
from contextlib import contextmanager, nullcontext
import inspect
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from tri_stream_deepfake_detection.data import get_train_transforms, get_val_transforms
from tri_stream_deepfake_detection.data.dataset import (
    CombinedVideoDataset,
    VideoSequenceDataset,
)
from tri_stream_deepfake_detection.models.multistream import (
    VALID_STREAMS,
    _effnet_input_size,
    normalize_active_streams,
)
from tri_stream_deepfake_detection.models.temporal import TemporalMultiStreamDetector
from tri_stream_deepfake_detection.utils import (
    calculate_comprehensive_metrics,
    plot_training_history,
    setup_logger,
)
from tri_stream_deepfake_detection.utils.checkpoint import torch_load_checkpoint

logger = logging.getLogger(__name__)


def _load_json_config(path: str) -> dict:
    """Load a flat JSON config whose keys match argparse destinations."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return config


def _apply_config_defaults(parser: argparse.ArgumentParser, config_path: Optional[str]) -> None:
    if not config_path:
        return
    config = _load_json_config(config_path)
    valid_dests = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - valid_dests)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown}")
    parser.set_defaults(**config)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    """Write a reproducibility snapshot for the current run."""
    payload = {
        "argv": sys.argv,
        "args": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
    }
    path = output_dir / "run_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.float().reshape(-1)
        targets_smooth = targets
        if self.label_smoothing > 0:
            targets_smooth = targets_smooth * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets_smooth,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        return ((1.0 - pt) ** self.gamma * bce).mean()


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow:
                    continue
                if not self.shadow[k].is_floating_point():
                    self.shadow[k].copy_(v)
                else:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        if self.backup is not None:
            raise RuntimeError("EMA weights are already applied")
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            raise RuntimeError("EMA weights are not currently applied")
        model.load_state_dict(self.backup)
        self.backup = None

    @contextmanager
    def average_parameters(self, model: nn.Module):
        self.apply(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.clone().detach() for k, v in state["shadow"].items()}
        self.decay = state.get("decay", self.decay)


def _get_preds_probs(logits: torch.Tensor):
    logits = logits.reshape(-1)
    probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    preds = (probs >= 0.5).astype(np.int64)
    return preds, probs


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_cuda(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def _infer_checkpoint_phase(ckpt: dict) -> int:
    metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
    phase = metrics.get("phase") if isinstance(metrics, dict) else None
    if phase in (1, 2):
        return int(phase)
    raise ValueError("Checkpoint metrics.phase is missing or invalid.")


def _set_global_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    sampler=None,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False

    sig = inspect.signature(DataLoader.__init__).parameters
    if num_workers > 0 and "prefetch_factor" in sig:
        kwargs["prefetch_factor"] = prefetch_factor
    if num_workers > 0 and "persistent_workers" in sig:
        kwargs["persistent_workers"] = bool(persistent_workers)

    return DataLoader(**kwargs)


def _fusion_contribution_penalty(
    fusion_w: torch.Tensor,
    min_freq: float,
    min_srm: float,
    active_streams,
) -> torch.Tensor:
    penalties = []
    active = set(active_streams)
    stream_index = {name: idx for idx, name in enumerate(VALID_STREAMS)}
    if "freq" in active:
        penalties.append(F.relu(float(min_freq) - fusion_w[:, stream_index["freq"]]))
    if "srm" in active:
        penalties.append(F.relu(float(min_srm) - fusion_w[:, stream_index["srm"]]))
    if not penalties:
        return fusion_w.new_zeros(())
    return sum(penalties).mean()


def _optimizer_param_groups(
    model: nn.Module,
    spatial_params,
    temporal_params,
    spatial_lr: float,
    temporal_lr: float,
    weight_decay: float = 1e-3,
) -> list[dict]:
    norm_types = (
        nn.modules.batchnorm._BatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    no_decay_ids = set()
    for module in model.modules():
        if isinstance(module, norm_types):
            no_decay_ids.update(id(p) for p in module.parameters(recurse=False))

    named_params = dict(model.named_parameters())
    for name, param in named_params.items():
        if (
            name.endswith(".bias")
            or name in {"spatial_token_embed", "temporal.cls_token"}
            or param.ndim == 0
            or name.endswith(("freq_scale_raw", "srm_scale_raw", "_freq_mix"))
        ):
            no_decay_ids.add(id(param))

    groups = []
    seen = set()
    for params, lr in ((spatial_params, spatial_lr), (temporal_params, temporal_lr)):
        decay, no_decay = [], []
        for param in params:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            (no_decay if id(param) in no_decay_ids else decay).append(param)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def _freeze_backbone_bn_stats(model: nn.Module) -> None:
    spatial = getattr(model, "spatial", None)
    if spatial is None:
        return
    for encoder_name in ("rgb_encoder", "freq_encoder", "srm_encoder"):
        encoder = getattr(spatial, encoder_name, None)
        if encoder is None:
            continue
        for module in encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, device, epoch, disable_tqdm: bool,
                    aux_weights: dict, fusion_reg_cfg: dict,
                    grad_clip: float = 0.0, ema=None,
                    grad_accum_steps: int = 1,
                    freeze_backbone_bn: bool = True,
                    ema_update_freq: int = 6):
    model.train()
    if freeze_backbone_bn:
        _freeze_backbone_bn_stats(model)
    running_loss = 0.0
    all_preds, all_labels = [], []
    valid_sample_count = 0

    use_aux = any(v > 0.0 for v in aux_weights.values())
    use_fusion_reg = float(fusion_reg_cfg.get("weight", 0.0)) > 0.0
    grad_accum_steps = max(1, int(grad_accum_steps))
    
    ema_update_freq = max(1, int(ema_update_freq))

    optimizer.zero_grad(set_to_none=True)
    optimizer_step_idx = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", disable=disable_tqdm)
    for iter_idx, (seqs, labels) in enumerate(pbar):
        seqs   = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        labels_target = labels.float()

        with _autocast_cuda(scaler.is_enabled()):
            if use_aux or use_fusion_reg:
                logits, extras = model(
                    seqs,
                    return_aux=use_aux,
                    return_fusion_w=use_fusion_reg,
                )
                loss = criterion(logits, labels_target)
                if use_aux:
                    if aux_weights['rgb'] > 0.0:
                        loss = loss + aux_weights['rgb']  * criterion(extras['aux_rgb'],  labels_target)
                    if aux_weights['freq'] > 0.0:
                        loss = loss + aux_weights['freq'] * criterion(extras['aux_freq'], labels_target)
                    if aux_weights['srm'] > 0.0:
                        loss = loss + aux_weights['srm']  * criterion(extras['aux_srm'],  labels_target)
                if use_fusion_reg:
                    fusion_pen = _fusion_contribution_penalty(
                        extras["fusion_w"],
                        min_freq=float(fusion_reg_cfg.get("min_freq", 0.08)),
                        min_srm=float(fusion_reg_cfg.get("min_srm", 0.12)),
                        active_streams=fusion_reg_cfg.get("active_streams", ("rgb", "freq", "srm")),
                    )
                    loss = loss + float(fusion_reg_cfg.get("weight", 0.0)) * fusion_pen
            else:
                logits = model(seqs)
                loss = criterion(logits, labels_target)

        loss_raw_for_log = loss.detach()
        if not torch.isfinite(loss_raw_for_log):
            logger.warning("Skipping batch with non-finite loss: %s", loss_raw_for_log.item())
            optimizer.zero_grad(set_to_none=True)
            continue
        loss = loss / grad_accum_steps
        should_step = (
            ((iter_idx + 1) % grad_accum_steps == 0)
            or ((iter_idx + 1) == len(loader))
        )

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if should_step:
                prev_scale = scaler.get_scale()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    if not torch.isfinite(grad_norm):
                        logger.warning("Skipping optimizer step with non-finite grad norm: %s", grad_norm.item())
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
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
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    if not torch.isfinite(grad_norm):
                        logger.warning("Skipping optimizer step with non-finite grad norm: %s", grad_norm.item())
                        optimizer.zero_grad(set_to_none=True)
                        continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step_idx += 1
                if ema is not None and optimizer_step_idx % ema_update_freq == 0:
                    ema.update(model)

        running_loss += loss_raw_for_log.item() * seqs.size(0)
        valid_sample_count += seqs.size(0)
        preds, _ = _get_preds_probs(logits.detach())
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
        pbar.set_postfix({'loss': f'{loss_raw_for_log.item():.4f}'})

    if ema is not None:
        ema.update(model)

    if valid_sample_count == 0:
        raise RuntimeError("Training produced no finite batches.")
    epoch_loss = running_loss / valid_sample_count
    epoch_acc  = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    epoch_rec  = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch, disable_tqdm: bool):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []
    stream_index = {name: idx for idx, name in enumerate(VALID_STREAMS)}
    fusion_w_sum = np.zeros(len(VALID_STREAMS), dtype=np.float64)
    fusion_w_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", disable=disable_tqdm)
    for seqs, labels in pbar:
        seqs   = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        labels_target = labels.float()

        logits, extras = model(seqs, return_fusion_w=True)
        loss = criterion(logits, labels_target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Validation produced non-finite loss: {loss.item()}")

        running_loss += loss.item() * seqs.size(0)
        _, probs = _get_preds_probs(logits)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        w = extras["fusion_w"].detach().cpu().numpy()
        fusion_w_sum += w.sum(axis=0)
        fusion_w_count += w.shape[0]

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    labels_arr = np.asarray(all_labels, dtype=np.int64)
    if labels_arr.size == 0:
        raise RuntimeError("Validation produced no finite batches.")
    epoch_loss = running_loss / labels_arr.size
    probs_arr  = np.asarray(all_probs, dtype=np.float64)
    metrics    = calculate_comprehensive_metrics(probs=probs_arr, labels=labels_arr)

    fusion_w_mean = fusion_w_sum / max(fusion_w_count, 1)
    metrics["fusion_w_rgb"] = float(fusion_w_mean[stream_index["rgb"]])
    metrics["fusion_w_freq"] = float(fusion_w_mean[stream_index["freq"]])
    metrics["fusion_w_srm"] = float(fusion_w_mean[stream_index["srm"]])

    return epoch_loss, metrics, probs_arr, labels_arr


def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Train Temporal Multi-Stream DeepFake Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Optional flat JSON config. CLI flags override config values.")
    parser.add_argument("--train-real", nargs="+", default=None)
    parser.add_argument("--train-fake", nargs="+", default=None)
    parser.add_argument("--val-real",   nargs="+", default=None)
    parser.add_argument("--val-fake",   nargs="+", default=None)

    parser.add_argument("--output-dir", type=str, default="outputs/temporal")

    parser.add_argument("--model",       type=str, default="efficientnet-b4",
                        help="RGB stream backbone. B4 typically needs --grad-checkpoint "
                             "for full fine-tuning on 16GB VRAM.")
    parser.add_argument("--freq-backbone", type=str, default="efficientnet-b4",
                        help="Frequency/Wavelet stream backbone.")
    parser.add_argument("--srm-backbone",  type=str, default="efficientnet-b4",
                        help="SRM stream backbone.")
    parser.add_argument(
        "--active-streams",
        nargs="+",
        default=["rgb", "freq", "srm"],
        choices=["rgb", "freq", "srm"],
        help="Spatial streams used by fusion. Default uses the full RGB+freq+SRM model.",
    )
    parser.add_argument("--augmentation", type=str, default="medium",
                        choices=["light", "medium", "heavy"],
                        help="Augmentation level. 'medium' adds JPEG compression+blur "
                             "for robustness.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing on backbone. "
                             "Required for B4 fine-tuning on ≤16GB VRAM. "
                             "Reduces activation memory ~6x, ~30%% slower.")
    parser.add_argument("--n-frames",    type=int, default=16,
                        help="Fixed T: number of frames per video sequence")
    parser.add_argument("--sampling",    type=str, default="uniform",
                        choices=["uniform", "random"],
                        help="Frame selection strategy (uniform or random for training aug)")
    parser.add_argument("--temporal-dropout-p", type=float, default=0.10,
                        help="Probability of replacing 1-N selected frames with neighboring "
                             "frames while keeping sequence length fixed.")
    parser.add_argument("--max-temporal-drop", type=int, default=2,
                        help="Maximum frames replaced when temporal dropout is applied.")
    parser.add_argument("--frame-shuffle-p", type=float, default=0.0,
                        help="Probability of shuffling selected frames. Use mostly as a "
                             "temporal-reasoning diagnostic; default keeps order intact.")
    parser.add_argument("--clip-jpeg-p", type=float, default=0.35,
                        help="Probability of applying one JPEG quality to all frames in a "
                             "training clip before per-frame transforms.")
    parser.add_argument("--clip-jpeg-quality", type=int, nargs=2, default=(40, 85),
                        metavar=("MIN", "MAX"),
                        help="Quality range for clip-level JPEG augmentation.")
    parser.add_argument("--num-heads",   type=int, default=8,
                        help="Temporal Transformer attention heads")
    parser.add_argument("--num-layers",  type=int, default=2,
                        help="Temporal Transformer encoder layers")
    parser.add_argument("--srm-filters", type=int, default=30)
    parser.add_argument(
        "--spatial-token-grid",
        type=int,
        default=2,
        help="Coarse spatial grid per frame before temporal Transformer. "
             "2 keeps 4 local tokens/frame; 1 uses global pooled temporal input.",
    )
    parser.add_argument(
        "--stream-dropout-p",
        type=float,
        default=0.1,
        help="Drop probability for each stream in channel-attention fusion during training.",
    )
    parser.add_argument(
        "--freq-logit-bias",
        type=float,
        default=0.08,
        help="Initial additive bias on frequency-stream fusion logit (learnable; "
             "optimizer can shrink it if freq hurts val/OOD).",
    )
    parser.add_argument(
        "--srm-logit-bias",
        type=float,
        default=0.0,
        help="Initial additive bias on SRM-stream fusion logit (learnable).",
    )
    parser.add_argument(
        "--freq-mode",
        type=str,
        default="wavelet_ml",
        choices=["wavelet_ml"],
        help="Frequency-stream representation. Only multi-level db4 DWT is supported.",
    )
    parser.add_argument(
        "--wavelet-level",
        type=int,
        default=3,
        help="Number of db4 DWT levels (>=1).",
    )
    parser.add_argument(
        "--wavelet-type",
        type=str,
        default="db4",
        choices=["db4"],
        help="Wavelet kernel. Only 'db4' is supported.",
    )

    parser.add_argument("--aux-rgb",  type=float, default=0.1,
                        help="Weight for RGB-stream auxiliary loss.")
    parser.add_argument("--aux-freq", type=float, default=0.3,
                        help="Weight for Frequency-stream auxiliary loss.")
    parser.add_argument("--aux-srm",  type=float, default=0.3,
                        help="Weight for SRM-stream auxiliary loss.")
    parser.add_argument(
        "--fusion-contrib-loss",
        type=float,
        default=0.5,
        help="Weight for fusion contribution regularization (0 disables). "
             "This penalizes low freq/srm fusion weights to reduce RGB collapse.",
    )
    parser.add_argument(
        "--fusion-reg-schedule",
        type=str,
        default="constant",
        choices=["constant", "decay"],
        help="Fusion regularization weight schedule. 'constant' keeps weight fixed, "
             "'decay' starts at 1.0 and decays to 0.3 over training.",
    )
    parser.add_argument(
        "--fusion-min-freq",
        type=float,
        default=0.08,
        help="Minimum target for mean freq fusion weight in contribution regularization.",
    )
    parser.add_argument(
        "--fusion-min-srm",
        type=float,
        default=0.12,
        help="Minimum target for mean SRM fusion weight in contribution regularization.",
    )
    parser.add_argument("--batch-size",  type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="DataLoader prefetch factor (only when num_workers > 0)")
    parser.add_argument("--persistent-workers", action="store_true",
                        help="Keep DataLoader workers alive between epochs (good for server training)")
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Disable tqdm progress bars (cleaner logs for nohup/screen)")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--amp",         action="store_true",
                        help="Mixed precision (recommended on GPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic training (slower, but reproducible)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 disables). "
                             "Stabilizes training with focal loss + multi-stream.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0-0.1). Smooth targets towards 0.5 "
                             "for better generalization. Recommended: 0.05.")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay factor (0 disables). Maintains smoothed model weights "
                             "for more stable validation. Recommended: 0.999.")
    parser.add_argument("--ema-update-freq", type=int, default=6,
                        help="Update EMA weights every N successful optimizer steps.")
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                        help="Accumulate gradients over N micro-batches. Effective video "
                             "batch equals batch_size * grad_accum_steps.")
    parser.add_argument("--freeze-backbone-bn", dest="freeze_backbone_bn",
                        action="store_true", default=True,
                        help="Keep EfficientNet BatchNorm layers in eval mode during training. "
                             "Recommended for tiny video batches.")
    parser.add_argument("--no-freeze-backbone-bn", dest="freeze_backbone_bn",
                        action="store_false",
                        help="Allow EfficientNet BatchNorm running stats to update.")

    parser.add_argument("--phase1-epochs", type=int, default=8,
                        help="Phase 1: spatial-only training (Transformer frozen). "
                             "Forward uses learned frame-weighted pooling for clean "
                             "gradients to spatial encoder.")
    parser.add_argument("--phase2-epochs", type=int, default=25,
                        help="Phase 2: full temporal training (all params active). "
                             "Transformer learns on stable spatial features.")
    parser.add_argument("--lr-backbone",   type=float, default=5e-5,
                        help="LR for spatial backbone in Phase 2 (lower than temporal head).")

    parser.add_argument("--balanced-sampler",  action="store_true")
    parser.add_argument("--focal-loss",        action="store_true")
    parser.add_argument("--focal-gamma",       type=float, default=2.0)

    parser.add_argument("--best-metric",  type=str, default="auc",
                        choices=["acc", "auc", "f1", "bal_acc"])
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early stop after N consecutive non-improving Phase-2 epochs (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement required to reset early-stop counter.",
    )

    parser.add_argument("--resume",              type=str, default=None,
                        help="Resume full temporal checkpoint")
    parser.add_argument("--pretrained-spatial",  type=str, default=None,
                        help="Bootstrap from a frame-level multi-stream checkpoint")
    parser.add_argument(
        "--freq-srm-warmup-epochs",
        type=int,
        default=0,
        help="Freeze all RGB-specific trainable modules for N epochs while RGB remains "
             "a fixed reference in fusion (0 disables).",
    )

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()
    _apply_config_defaults(parser, config_args.config)
    args = parser.parse_args()
    for required_name in ("train_real", "train_fake", "val_real", "val_fake"):
        if not getattr(args, required_name):
            parser.error(f"--{required_name.replace('_', '-')} is required unless provided by --config")
    args.active_streams = list(normalize_active_streams(args.active_streams))
    _set_global_seed(args.seed, deterministic=args.deterministic)
    if sys.platform == "win32":
        torch.multiprocessing.set_sharing_strategy("file_system")

    ckpt_dir    = Path(args.output_dir) / "checkpoints"
    log_dir     = Path(args.output_dir) / "logs"
    results_dir = Path(args.output_dir) / "results"
    for d in [ckpt_dir, log_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="temporal_training",
        log_file=str(log_dir / "training.log"),
        level="INFO",
    )

    _write_run_config(args, Path(args.output_dir))
    logger.info("Temporal training start")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.resume:
        resume_meta = torch_load_checkpoint(args.resume, map_location="cpu")
        if isinstance(resume_meta, dict):
            ckpt_streams = list(resume_meta.get("active_streams", ["rgb", "freq", "srm"]))
            if ckpt_streams != list(args.active_streams):
                raise ValueError(
                    "Resume checkpoint active_streams does not match --active-streams "
                    f"(checkpoint={ckpt_streams}, args={list(args.active_streams)})."
                )
            ckpt_grid = int(resume_meta.get("spatial_token_grid", 1))
            if ckpt_grid != int(args.spatial_token_grid):
                logger.warning(
                    "Resume checkpoint uses spatial_token_grid=%d but args requested %d; "
                    "using checkpoint value to avoid positional-encoding mismatch.",
                    ckpt_grid,
                    int(args.spatial_token_grid),
                )
                args.spatial_token_grid = ckpt_grid

    image_size = _effnet_input_size(args.model)
    train_tfm  = get_train_transforms(image_size, augmentation_level=args.augmentation)
    val_tfm    = get_val_transforms(image_size)
    logger.info(
        "device=%s rgb_backbone=%s freq_backbone=%s srm_backbone=%s "
        "active_streams=%s freq_mode=%s wavelet_level=%d T=%d token_grid=%d epochs=%d+%d",
        device,
        args.model,
        args.freq_backbone,
        args.srm_backbone,
        ",".join(args.active_streams),
        args.freq_mode,
        int(args.wavelet_level),
        int(args.n_frames),
        int(args.spatial_token_grid),
        int(args.phase1_epochs),
        int(args.phase2_epochs),
    )
    logger.info(
        "sequence_aug: temporal_dropout=%.3f max_drop=%d frame_shuffle=%.3f "
        "clip_jpeg=%.3f quality=%s",
        float(args.temporal_dropout_p),
        int(args.max_temporal_drop),
        float(args.frame_shuffle_p),
        float(args.clip_jpeg_p),
        tuple(args.clip_jpeg_quality),
    )
    logger.info(
        "optimization: batch=%d grad_accum=%d effective_batch=%d freeze_backbone_bn=%s ema_update_freq=%d",
        int(args.batch_size),
        int(args.grad_accum_steps),
        int(args.batch_size) * max(1, int(args.grad_accum_steps)),
        bool(args.freeze_backbone_bn),
        max(1, int(args.ema_update_freq)),
    )
    logger.info(
        "fusion_reg: weight=%.4f schedule=%s min_freq=%.3f min_srm=%.3f",
        float(args.fusion_contrib_loss),
        args.fusion_reg_schedule,
        float(args.fusion_min_freq),
        float(args.fusion_min_srm),
    )
    logger.info(
        "aux_losses: rgb=%.2f freq=%.2f srm=%.2f",
        float(args.aux_rgb),
        float(args.aux_freq),
        float(args.aux_srm),
    )
    if args.freq_srm_warmup_epochs > 0:
        logger.info(
            "warmup: freq/SRM warmup for %d epochs (RGB frozen)",
            int(args.freq_srm_warmup_epochs),
        )

    for p in args.train_real + args.train_fake + args.val_real + args.val_fake:
        if not os.path.exists(p):
            logger.error(f"Dataset directory not found: {p}")
            raise FileNotFoundError(f"Dataset directory not found: {p}")

    train_real_cfg = [(p, -1) for p in args.train_real]
    train_fake_cfg = [(p, -1) for p in args.train_fake]
    val_real_cfg   = [(p, -1) for p in args.val_real]
    val_fake_cfg   = [(p, -1) for p in args.val_fake]

    train_ds = CombinedVideoDataset(
        VideoSequenceDataset(train_real_cfg, is_real=True,  transform=train_tfm,
                             n_frames=args.n_frames, sampling=args.sampling,
                             temporal_dropout_p=args.temporal_dropout_p,
                             max_temporal_drop=args.max_temporal_drop,
                             frame_shuffle_p=args.frame_shuffle_p,
                             clip_jpeg_p=args.clip_jpeg_p,
                             clip_jpeg_quality=tuple(args.clip_jpeg_quality)),
        VideoSequenceDataset(train_fake_cfg, is_real=False, transform=train_tfm,
                             n_frames=args.n_frames, sampling=args.sampling,
                             temporal_dropout_p=args.temporal_dropout_p,
                             max_temporal_drop=args.max_temporal_drop,
                             frame_shuffle_p=args.frame_shuffle_p,
                             clip_jpeg_p=args.clip_jpeg_p,
                             clip_jpeg_quality=tuple(args.clip_jpeg_quality)),
    )
    val_ds = CombinedVideoDataset(
        VideoSequenceDataset(val_real_cfg, is_real=True,  transform=val_tfm,
                             n_frames=args.n_frames, sampling="uniform"),
        VideoSequenceDataset(val_fake_cfg, is_real=False, transform=val_tfm,
                             n_frames=args.n_frames, sampling="uniform"),
    )

    logger.info(f"Train: {len(train_ds)} videos  |  Val: {len(val_ds)} videos")

    train_sampler = None
    if args.balanced_sampler:
        labels_np = np.asarray(train_ds.labels, dtype=np.int64)
        counts    = np.bincount(labels_np, minlength=2).astype(np.float64)
        counts[counts == 0] = 1.0
        weights   = (1.0 / counts)[labels_np]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        logger.info(
            f"WeightedRandomSampler: real={int(counts[0])}, fake={int(counts[1])} videos"
        )

    _pin = device.type == "cuda" and sys.platform != "win32"

    train_loader = _build_dataloader(
        dataset=train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=_pin,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    val_loader = _build_dataloader(
        dataset=val_ds,
        batch_size=args.batch_size,
        sampler=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=_pin,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    model = TemporalMultiStreamDetector(
        backbone=args.model,
        freq_backbone=args.freq_backbone,
        srm_backbone=args.srm_backbone,
        n_frames=args.n_frames,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        srm_filters=args.srm_filters,
        stream_dropout_p=args.stream_dropout_p,
        freq_logit_bias=args.freq_logit_bias,
        srm_logit_bias=args.srm_logit_bias,
        spatial_token_grid=args.spatial_token_grid,
        freq_mode=args.freq_mode,
        wavelet_level=args.wavelet_level,
        wavelet_type=args.wavelet_type,
        pretrained=True,
        bce_output=True,
        use_grad_checkpoint=args.grad_checkpoint,
        active_streams=args.active_streams,
    ).to(device)

    if args.pretrained_spatial:
        logger.info(f"Loading spatial weights from: {args.pretrained_spatial}")
        model.load_spatial_weights(args.pretrained_spatial, device=str(device))

    if args.phase1_epochs > 0:
        model.set_phase(1)
        current_phase = 1
        logger.info("[Phase 1] spatial-only")
    else:
        model.set_phase(2)
        current_phase = 2
        logger.info("[Phase 2] full temporal from epoch 1")

    total, trainable = model.count_parameters()
    logger.info(f"params total={total:,} trainable={trainable:,}")

    labels_np = np.asarray(train_ds.labels, dtype=np.int64)
    counts    = np.bincount(labels_np, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    
    logger.info(
        f"Class distribution: real={int(counts[0])}, fake={int(counts[1])} "
        f"(real/fake={counts[0]/counts[1]:.2f}:1)"
    )
    
    if args.balanced_sampler:
        pos_weight_val = 1.0
        logger.info(
            f"pos_weight=1.0 (balanced_sampler already balances batches, "
            f"no additional weighting needed)"
        )
    else:
        pos_weight_val = counts[0] / counts[1]
        logger.info(
            f"pos_weight={pos_weight_val:.3f} "
            f"(BCE positive class=fake; real/fake={counts[0]/counts[1]:.2f}:1)"
        )
    
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    if args.focal_loss:
        criterion = FocalBCELoss(
            gamma=args.focal_gamma,
            pos_weight=pos_weight_tensor,
            label_smoothing=args.label_smoothing,
        )
        logger.info(
            f"loss=FocalBCELoss(gamma={args.focal_gamma:.1f}, "
            f"pos_weight={pos_weight_val:.3f}, "
            f"label_smoothing={args.label_smoothing:.3f})"
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        logger.info(f"loss=BCEWithLogitsLoss(pos_weight={pos_weight_val:.3f})")

    spatial_params  = list(model.spatial.parameters())
    aux_params      = (
        list(model.aux_rgb.parameters())
        + list(model.aux_freq.parameters())
        + list(model.aux_srm.parameters())
    )
    temporal_params = (
        list(model.temporal.parameters())
        + list(model.temporal_diff.parameters())
        + [model.spatial_token_embed]
        + list(model.frame_scorer.parameters())
        + list(model.classifier.parameters())
        + aux_params
    )

    total_epochs = args.phase1_epochs + args.phase2_epochs

    def _build_optimizer_scheduler(phase: int):
        steps_per_epoch = max(
            1,
            (len(train_loader) + max(1, int(args.grad_accum_steps)) - 1)
            // max(1, int(args.grad_accum_steps)),
        )
        spatial_lr = args.lr_backbone if phase == 2 else args.lr
        temporal_lr = args.lr
        if phase == 2:
            param_groups = _optimizer_param_groups(
                model, spatial_params, temporal_params,
                spatial_lr=args.lr_backbone, temporal_lr=args.lr,
            )
        else:
            param_groups = _optimizer_param_groups(
                model, spatial_params, temporal_params,
                spatial_lr=args.lr, temporal_lr=args.lr,
            )

        if phase == 2:
            num_steps = steps_per_epoch * max(args.phase2_epochs, 1)
            warmup = steps_per_epoch * min(3, max(args.phase2_epochs // 6, 1))
        else:
            num_steps = steps_per_epoch * max(args.phase1_epochs, 1)
            warmup = steps_per_epoch * min(3, max(args.phase1_epochs // 3, 1))

        opt = AdamW(param_groups, weight_decay=1e-3)
        sch = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=warmup, num_training_steps=num_steps
        )
        return opt, sch

    optimizer, scheduler = _build_optimizer_scheduler(current_phase)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler  = _make_grad_scaler(use_amp)

    start_epoch = 0
    best_score  = 0.0
    no_improve_epochs = 0
    resume_ckpt = None

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        ckpt = model.load_checkpoint(args.resume, device=str(device))
        resume_ckpt = ckpt
        if isinstance(ckpt, dict):
            if ckpt.get("label_convention") != "real=0,fake=1" or ckpt.get("score_target") != "fake":
                raise ValueError(
                    "Cannot safely resume training from a checkpoint without "
                    "label_convention='real=0,fake=1' and score_target='fake'."
                )
            if ckpt.get("architecture") != "temporal_multi_stream":
                raise ValueError(f"Checkpoint architecture is not temporal_multi_stream: {ckpt.get('architecture')}")
            resume_phase = _infer_checkpoint_phase(ckpt)
            if resume_phase != current_phase:
                model.set_phase(resume_phase)
                current_phase = resume_phase
                optimizer, scheduler = _build_optimizer_scheduler(current_phase)
                logger.info(
                    f"Resume checkpoint indicates Phase {resume_phase}; "
                    f"optimizer/scheduler rebuilt to match checkpoint phase."
                )

            optimizer_state_loaded = False
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    optimizer_state_loaded = True
                except ValueError as e:
                    logger.warning(
                        "Could not load optimizer state (likely parameter-group mismatch). "
                        "Continuing with fresh optimizer state. Details: %s", e
                    )
            if "scheduler_state_dict" in ckpt and (
                "optimizer_state_dict" not in ckpt or optimizer_state_loaded
            ):
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception as e:
                    logger.warning(
                        "Could not load scheduler state. Continuing with fresh scheduler. "
                        "Details: %s", e
                    )
            if "scaler_state_dict" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                except Exception as e:
                    logger.warning(
                        "Could not load AMP scaler state. Continuing with a fresh scaler. "
                        "Details: %s", e
                    )
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"])

            training_state = ckpt.get("training_state") or {}
            if isinstance(training_state, dict):
                restored_best = training_state.get("best_score")
                if restored_best is not None:
                    best_score = float(restored_best)
                no_improve_epochs = int(training_state.get("no_improve_epochs", 0))

    ema = None
    if args.ema_decay > 0:
        ema = EMAModel(model, decay=args.ema_decay)
        if isinstance(resume_ckpt, dict) and "ema_state_dict" in resume_ckpt:
            ema.load_state_dict(resume_ckpt["ema_state_dict"])
            logger.info("Restored EMA state from checkpoint (decay=%.6f)", ema.decay)
        else:
            logger.info(f"EMA enabled (decay={args.ema_decay})")
    elif isinstance(resume_ckpt, dict) and "ema_state_dict" in resume_ckpt:
        logger.warning("Checkpoint contains EMA state, but EMA is disabled by --ema-decay=0")

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
        phase2_started = False
        if current_phase == 1 and epoch > args.phase1_epochs:
            model.set_phase(2)
            current_phase = 2
            optimizer, scheduler = _build_optimizer_scheduler(phase=2)
            scaler = _make_grad_scaler(use_amp)
            phase2_started = True

        rgb_warmup_active = bool(
            args.freq_srm_warmup_epochs > 0
            and "rgb" in model.active_streams
            and epoch <= args.freq_srm_warmup_epochs
        )
        if rgb_warmup_active:
            model.set_stream_trainable("rgb", False)
            if epoch == start_epoch + 1:
                logger.info(
                    "[Warmup] Freezing all RGB-specific modules through epoch %d; "
                    "RGB remains a fixed reference in fusion.",
                    args.freq_srm_warmup_epochs,
                )
        elif (
            args.freq_srm_warmup_epochs > 0
            and "rgb" in model.active_streams
            and epoch == args.freq_srm_warmup_epochs + 1
        ):
            model.set_stream_trainable("rgb", True)
            logger.info("[Warmup END] Unfreezing all RGB-specific modules")

        if phase2_started:
            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                f"[Phase 2 START] epoch={epoch} "
                f"lr_backbone={args.lr_backbone} lr_temporal={args.lr} "
                f"trainable={trainable_now:,}"
            )

        aux_weights = {
            "rgb":  float(args.aux_rgb) if "rgb" in model.active_streams and not rgb_warmup_active else 0.0,
            "freq": float(args.aux_freq) if "freq" in model.active_streams else 0.0,
            "srm":  float(args.aux_srm) if "srm" in model.active_streams else 0.0,
        }
        
        fusion_reg_weight = float(args.fusion_contrib_loss)
        if args.fusion_reg_schedule == "decay":
            progress = (epoch - 1) / max(total_epochs - 1, 1)
            fusion_reg_weight = 1.0 - progress * 0.7
            if epoch == 1 or epoch % 5 == 0:
                logger.info(f"Fusion reg weight (decay schedule): {fusion_reg_weight:.4f}")
        
        fusion_reg_cfg = {
            "weight": fusion_reg_weight,
            "min_freq": float(args.fusion_min_freq),
            "min_srm": float(args.fusion_min_srm),
            "active_streams": model.active_streams,
        }
        train_loss, train_acc, train_prec, train_rec = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, disable_tqdm=args.no_tqdm,
            aux_weights=aux_weights,
            fusion_reg_cfg=fusion_reg_cfg,
            grad_clip=args.grad_clip,
            ema=ema,
            grad_accum_steps=args.grad_accum_steps,
            freeze_backbone_bn=args.freeze_backbone_bn,
            ema_update_freq=args.ema_update_freq,
        )

        if ema is not None:
            with ema.average_parameters(model):
                val_loss, val_metrics, probs, labels_arr = validate_one_epoch(
                    model, val_loader, criterion, device, epoch, disable_tqdm=args.no_tqdm
                )
        else:
            val_loss, val_metrics, probs, labels_arr = validate_one_epoch(
                model, val_loader, criterion, device, epoch, disable_tqdm=args.no_tqdm
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
        val_rec_real = float(recall_score(labels_arr, preds_eer, pos_label=0, zero_division=0))

        fw_rgb  = float(val_metrics.get("fusion_w_rgb",  0.0))
        fw_freq = float(val_metrics.get("fusion_w_freq", 0.0))
        fw_srm  = float(val_metrics.get("fusion_w_srm",  0.0))
        
        freq_scale = float(
            model.spatial.fusion.min_aux_scale
            + F.softplus(model.spatial.fusion.freq_scale_raw).item()
        )
        srm_scale = float(
            model.spatial.fusion.min_aux_scale
            + F.softplus(model.spatial.fusion.srm_scale_raw).item()
        )
        logger.info(
            "Epoch %d/%d phase=%d train_loss=%.4f train_acc=%.4f "
            "val_loss=%.4f val_auc=%.4f val_f1=%.4f val_bal_acc=%.4f",
            epoch,
            total_epochs,
            current_phase,
            train_loss,
            train_acc,
            val_loss,
            val_auc,
            val_f1,
            val_bal_acc,
        )
        logger.info(
            "EER=%.4f threshold=%.4f | "
            "fusion rgb=%.3f freq=%.3f srm=%.3f | scales freq=%.3f srm=%.3f",
            val_eer,
            eer_thr,
            fw_rgb,
            fw_freq,
            fw_srm,
            freq_scale,
            srm_scale,
        )
        
        if "freq" in model.active_streams and fw_freq < 0.05:
            logger.warning(f"WARNING: Frequency fusion weight collapsed below 0.05 (current: {fw_freq:.3f})")
        if "srm" in model.active_streams and fw_srm < 0.08:
            logger.warning(f"WARNING: SRM fusion weight collapsed below 0.08 (current: {fw_srm:.3f})")

        val_ckpt_metrics = {
            "active_streams": list(model.active_streams),
            "phase": current_phase,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_bal_acc": val_bal_acc,
            "val_eer": val_eer,
            "val_eer_threshold": eer_thr,
            "val_recall_real": val_rec_real,
            "val_fusion_w_rgb":  fw_rgb,
            "val_fusion_w_freq": fw_freq,
            "val_fusion_w_srm":  fw_srm,
        }

        phase_tag = f"P{current_phase}"
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_precision"].append(train_prec)
        history["train_recall"].append(train_rec)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        should_stop = False
        if current_phase == 2:
            score = {
                "acc": val_acc,
                "auc": val_auc,
                "f1": val_f1,
                "bal_acc": val_bal_acc,
            }[args.best_metric]
            score_name = args.best_metric

            score_is_finite = bool(
                np.isfinite(float(score))
                and np.isfinite(float(train_loss))
                and np.isfinite(float(val_loss))
            )
            if not score_is_finite:
                logger.warning(
                    "Skipping best-checkpoint and early-stop updates because metric/loss "
                    "is non-finite (score=%s, train_loss=%s, val_loss=%s).",
                    score,
                    train_loss,
                    val_loss,
                )
            elif score > (best_score + args.early_stop_min_delta):
                best_score = score
                no_improve_epochs = 0
                best_path = ckpt_dir / "best_model.pth"
                best_ckpt_metrics = {
                    **val_ckpt_metrics,
                    "best_metric_name": score_name,
                    "best_metric_value": float(score),
                }
                best_training_state = {
                    "best_score": best_score,
                    "no_improve_epochs": no_improve_epochs,
                }
                if ema is not None:
                    with ema.average_parameters(model):
                        model.save_checkpoint(
                            str(best_path),
                            epoch=epoch,
                            metrics=best_ckpt_metrics,
                            ema_state=ema.state_dict(),
                            training_state=best_training_state,
                            model_weights="ema",
                        )
                else:
                    model.save_checkpoint(
                        str(best_path),
                        epoch=epoch,
                        metrics=best_ckpt_metrics,
                        training_state=best_training_state,
                        model_weights="raw",
                    )
                logger.info(
                    f"Best model saved [{score_name}={best_score:.4f}] "
                    f"val_eer_threshold={eer_thr:.4f}"
                )
            else:
                no_improve_epochs += 1
                if args.early_stop_patience > 0:
                    logger.info(
                        "No improvement on %s (best=%.4f, current=%.4f). "
                        "Early-stop counter: %d/%d",
                        score_name,
                        best_score,
                        score,
                        no_improve_epochs,
                        args.early_stop_patience,
                    )

                if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                    logger.info(
                        "Early stopping triggered at epoch %d (patience=%d, min_delta=%.6f).",
                        epoch,
                        args.early_stop_patience,
                        args.early_stop_min_delta,
                    )
                    should_stop = True

        training_state = {
            "best_score": best_score,
            "no_improve_epochs": no_improve_epochs,
        }
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}_{phase_tag}.pth"
        model.save_checkpoint(
            str(ckpt_path),
            epoch=epoch,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            metrics=val_ckpt_metrics,
            ema_state=ema.state_dict() if ema is not None else None,
            scaler_state=scaler.state_dict(),
            training_state=training_state,
            model_weights="raw",
        )

        if should_stop:
            break
    plot_training_history(
        history,
        metrics=["loss", "accuracy", "precision", "recall"],
        save_path=str(results_dir / "training_history.png"),
        show=False,
    )

    logger.info("Training complete!")
    logger.info(f"Best {args.best_metric} (Phase 2): {best_score:.4f}")


if __name__ == "__main__":
    main()
