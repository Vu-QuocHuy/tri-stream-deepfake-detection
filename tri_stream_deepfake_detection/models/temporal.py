"""Temporal multi-stream video detector."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tri_stream_deepfake_detection.models.multistream import (
    VALID_STREAMS,
    MultiStreamDeepFakeDetector,
    normalize_active_streams,
)
from tri_stream_deepfake_detection.utils.checkpoint import torch_load_checkpoint


logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding max_len {self.pe.size(1)}"
            )
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TemporalTransformer(nn.Module):
    """Pre-LN Transformer encoder with a CLS aggregation token."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_frames: int = 64,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=max_frames + 1, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        B = frame_features.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, frame_features], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x[:, 0]


class TemporalDifferenceModule(nn.Module):
    """Gated residual module for frame-to-frame feature changes."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.diff_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        diff = torch.zeros_like(features)
        diff[:, 1:] = features[:, 1:] - features[:, :-1]
        diff_proj = self.diff_proj(diff)
        gate = self.gate(torch.cat([features, diff_proj], dim=-1))
        return features + gate * diff_proj

class TemporalMultiStreamDetector(nn.Module):
    """Video-level detector using multi-stream spatial features and temporal aggregation."""

    def __init__(
        self,
        backbone: str = 'efficientnet-b4',
        freq_backbone: Optional[str] = None,
        srm_backbone: Optional[str] = None,
        n_frames: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        srm_filters: int = 30,
        stream_dropout_p: float = 0.0,
        freq_logit_bias: float = 0.0,
        srm_logit_bias: float = 0.0,
        spatial_token_grid: int = 2,
        freq_mode: str = 'wavelet_ml',
        wavelet_level: int = 1,
        wavelet_type: str = 'db4',
        pretrained: bool = True,
        bce_output: bool = True,
        use_grad_checkpoint: bool = False,
        dropout: float = 0.1,
        active_streams: Optional[List[str]] = None,
    ):
        super().__init__()

        self.active_streams = normalize_active_streams(active_streams)
        self.backbone_name = backbone
        self.freq_backbone_name = freq_backbone or backbone
        self.srm_backbone_name = srm_backbone or backbone
        self.n_frames = n_frames
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.srm_filters = srm_filters
        self.stream_dropout_p = float(stream_dropout_p)
        self.freq_logit_bias_init = float(freq_logit_bias)
        self.srm_logit_bias_init = float(srm_logit_bias)
        self.spatial_token_grid = max(1, int(spatial_token_grid))
        self.dropout = float(dropout)
        self.bce_output = bce_output
        self.use_grad_checkpoint = use_grad_checkpoint

        self._phase: int = 2

        self.spatial = MultiStreamDeepFakeDetector(
            rgb_model=backbone,
            freq_model=self.freq_backbone_name,
            srm_model=self.srm_backbone_name,
            srm_filters=srm_filters,
            stream_dropout_p=stream_dropout_p,
            freq_logit_bias=freq_logit_bias,
            srm_logit_bias=srm_logit_bias,
            freq_mode=freq_mode,
            wavelet_level=wavelet_level,
            wavelet_type=wavelet_type,
            pretrained=pretrained,
            num_classes=2,
            active_streams=list(self.active_streams),
        )
        feat_dim = self.spatial._feat_dim

        if use_grad_checkpoint:
            self._enable_grad_checkpointing()

        self.temporal = TemporalTransformer(
            d_model=feat_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            max_frames=(n_frames * self.spatial_token_grid * self.spatial_token_grid) + 4,
        )

        self.temporal_diff = TemporalDifferenceModule(feat_dim)
        token_count = self.spatial_token_grid * self.spatial_token_grid
        self.spatial_token_embed = nn.Parameter(torch.zeros(1, 1, token_count, feat_dim))

        out_dim = 1 if bce_output else 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

        self.frame_scorer = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 1),
        )
        nn.init.zeros_(self.frame_scorer[-1].weight)
        nn.init.zeros_(self.frame_scorer[-1].bias)

        self.aux_rgb  = nn.Linear(feat_dim, out_dim)
        self.aux_freq = nn.Linear(feat_dim, out_dim)
        self.aux_srm  = nn.Linear(feat_dim, out_dim)

        self._feat_dim = feat_dim
        self._freeze_inactive_stream_params()

    def _stream_specific_modules(self) -> Dict[str, List[nn.Module]]:
        modules = {
            "rgb": [
                self.spatial.rgb_encoder,
                self.spatial._proj_rgb,
                self.spatial._spatial_attn["rgb"],
                self.aux_rgb,
            ],
            "freq": [
                self.spatial.freq_encoder,
                self.spatial._proj_freq,
                self.spatial._freq_adapter,
                self.spatial._wavelet_ml_adapter,
                self.spatial._freq_layer_norm,
                self.spatial._spatial_attn["freq"],
                self.aux_freq,
            ],
            "srm": [
                self.spatial.srm_encoder,
                self.spatial._proj_srm,
                self.spatial._srm_conv,
                self.spatial._srm_to3,
                self.spatial._srm_bn,
                self.spatial._spatial_attn["srm"],
                self.aux_srm,
            ],
        }
        for index, stream in enumerate(self.active_streams):
            modules[stream].append(self.spatial.fusion.stream_norms[index])
        return modules

    def _stream_specific_parameters(self) -> Dict[str, List[nn.Parameter]]:
        return {
            "rgb": [],
            "freq": [
                self.spatial._freq_mix,
                self.spatial.fusion.freq_logit_bias,
                self.spatial.fusion.freq_scale_raw,
            ],
            "srm": [
                self.spatial.fusion.srm_logit_bias,
                self.spatial.fusion.srm_scale_raw,
            ],
        }

    def set_stream_trainable(self, stream: str, trainable: bool) -> None:
        if stream not in VALID_STREAMS:
            raise ValueError(f"Unknown stream {stream!r}; expected one of {VALID_STREAMS}")
        enabled = bool(trainable and stream in self.active_streams)
        for module in self._stream_specific_modules()[stream]:
            for param in module.parameters():
                param.requires_grad = enabled
        for param in self._stream_specific_parameters()[stream]:
            param.requires_grad = enabled

    def _freeze_inactive_stream_params(self) -> None:
        active = set(self.active_streams)
        for stream, modules in self._stream_specific_modules().items():
            if stream in active:
                continue
            for module in modules:
                for p in module.parameters():
                    p.requires_grad = False
        for stream, params in self._stream_specific_parameters().items():
            if stream in active:
                continue
            for param in params:
                param.requires_grad = False

    def set_phase(self, phase: int) -> None:
        """Switch between spatial-only phase 1 and full temporal phase 2."""
        assert phase in (1, 2), "phase must be 1 or 2"
        self._phase = phase
        if phase == 1:
            for p in self.temporal.parameters():
                p.requires_grad = False
            for p in self.spatial.parameters():
                p.requires_grad = True
            for p in self.temporal_diff.parameters():
                p.requires_grad = False
            self.spatial_token_embed.requires_grad = True
            for p in self.classifier.parameters():
                p.requires_grad = True
            for p in self.frame_scorer.parameters():
                p.requires_grad = True
            for aux in [self.aux_rgb, self.aux_freq, self.aux_srm]:
                for p in aux.parameters():
                    p.requires_grad = True
            self._freeze_inactive_stream_params()
        else:
            for p in self.parameters():
                p.requires_grad = True
            self._freeze_inactive_stream_params()

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.spatial.parameters():
            p.requires_grad = not freeze

    def freeze_temporal(self, freeze: bool = True) -> None:
        for p in self.temporal.parameters():
            p.requires_grad = not freeze
        for p in self.temporal_diff.parameters():
            p.requires_grad = not freeze

    def _enable_grad_checkpointing(self) -> None:
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        def make_checkpointed(original_fn):
            def _ckpt_forward(*args, **kwargs):
                return ckpt_fn(original_fn, *args, use_reentrant=False, **kwargs)
            return _ckpt_forward
        
        encoder_by_stream = {
            "rgb": self.spatial.rgb_encoder,
            "freq": self.spatial.freq_encoder,
            "srm": self.spatial.srm_encoder,
        }
        for stream in self.active_streams:
            encoder = encoder_by_stream[stream]
            if hasattr(encoder, 'set_grad_checkpointing'):
                encoder.set_grad_checkpointing(True)
            else:
                for block in encoder._blocks:
                    block.forward = make_checkpointed(block.forward)

    def _encode_frames_batched(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        fused_flat = self.spatial.encode_frame(x_flat)
        return fused_flat.view(B, T, -1)

    def _encode_frame_tokens_batched(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        tokens_flat = self.spatial.encode_frame_tokens(
            x_flat, token_grid=self.spatial_token_grid
        )
        K = tokens_flat.shape[1]
        return tokens_flat.view(B, T, K, -1)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_fusion_w: bool = False,
    ):
        T = x.shape[1]

        if T != self.n_frames:
            raise ValueError(
                f"Expected {self.n_frames} frames per video, got {T}. "
                f"Ensure VideoSequenceDataset n_frames matches model n_frames."
            )

        need_streams = bool(return_aux or return_fusion_w)

        stream_feat_lists: Dict[str, List[torch.Tensor]] = {
            stream: [] for stream in self.active_streams
        }
        w_list: List[torch.Tensor] = []
        use_spatial_tokens = self.spatial_token_grid > 1

        if use_spatial_tokens and not need_streams and not self.use_grad_checkpoint:
            features = self._encode_frame_tokens_batched(x)
        elif not use_spatial_tokens and not need_streams and not self.use_grad_checkpoint:
            features = self._encode_frames_batched(x)
        else:
            frame_features = []
            for t in range(T):
                if use_spatial_tokens and need_streams:
                    fused, f_rgb, f_freq, f_srm, w = self.spatial.encode_frame_tokens_full(
                        x[:, t], token_grid=self.spatial_token_grid
                    )
                    for stream, feat in zip(VALID_STREAMS, (f_rgb, f_freq, f_srm)):
                        if feat is not None:
                            stream_feat_lists[stream].append(feat)
                    w_list.append(w)
                elif use_spatial_tokens:
                    fused = self.spatial.encode_frame_tokens(
                        x[:, t], token_grid=self.spatial_token_grid
                    )
                elif need_streams:
                    fused, f_rgb, f_freq, f_srm, w = self.spatial.encode_frame_full(x[:, t])
                    for stream, feat in zip(VALID_STREAMS, (f_rgb, f_freq, f_srm)):
                        if feat is not None:
                            stream_feat_lists[stream].append(feat)
                    w_list.append(w)
                else:
                    fused = self.spatial.encode_frame(x[:, t])
                frame_features.append(fused)
            features = torch.stack(frame_features, dim=1)

        if use_spatial_tokens:
            features = features + self.spatial_token_embed[:, :, :features.shape[2], :]

        if self._phase != 1:
            features = self.temporal_diff(features)

        if self._phase == 1:
            if use_spatial_tokens:
                frame_repr = features.mean(dim=2)
            else:
                frame_repr = features
            frame_w = torch.softmax(self.frame_scorer(frame_repr), dim=1)
            video_repr = (frame_repr * frame_w).sum(dim=1)
        else:
            if use_spatial_tokens:
                Bf, Tf, Kf, Df = features.shape
                features_for_temporal = features.reshape(Bf, Tf * Kf, Df)
                video_repr = self.temporal(features_for_temporal)
            else:
                video_repr = self.temporal(features)

        logits = self.classifier(video_repr)
        if self.bce_output:
            logits = logits.squeeze(-1)

        if not need_streams:
            return logits

        extras: Dict[str, torch.Tensor] = {}
        if return_aux:
            aux_head_by_stream = {
                "rgb": self.aux_rgb,
                "freq": self.aux_freq,
                "srm": self.aux_srm,
            }
            for stream in self.active_streams:
                feats = stream_feat_lists[stream]
                if not feats:
                    continue
                feat_T = sum(feats) / len(feats)
                aux = aux_head_by_stream[stream](feat_T)
                if self.bce_output:
                    aux = aux.squeeze(-1)
                extras[f"aux_{stream}"] = aux
        if return_fusion_w:
            extras["fusion_w"] = torch.stack(w_list, dim=1).mean(dim=1)
        return logits, extras

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
        ema_state: Optional[dict] = None,
        scaler_state: Optional[dict] = None,
        training_state: Optional[dict] = None,
        model_weights: str = "raw",
    ) -> None:
        if model_weights not in ("raw", "ema"):
            raise ValueError("model_weights must be 'raw' or 'ema'")
        ckpt: Dict[str, Any] = {
            'model_state_dict': self.state_dict(),
            'architecture': 'temporal_multi_stream',
            'label_convention': 'real=0,fake=1',
            'score_target': 'fake',
            'backbone': self.backbone_name,
            'freq_backbone': self.freq_backbone_name,
            'srm_backbone': self.srm_backbone_name,
            'active_streams': list(self.active_streams),
            'n_frames': self.n_frames,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'srm_filters': self.srm_filters,
            'stream_dropout_p': self.stream_dropout_p,
            'spatial_token_grid': self.spatial_token_grid,
            'dropout': self.dropout,
            'bce_output': self.bce_output,
            'out_dim': self.classifier[-1].out_features,
            'freq_logit_bias': float(self.spatial.fusion.freq_logit_bias.detach().cpu()),
            'srm_logit_bias': float(self.spatial.fusion.srm_logit_bias.detach().cpu()),
            'freq_scale': float(
                self.spatial.fusion.min_aux_scale
                + F.softplus(self.spatial.fusion.freq_scale_raw.detach()).cpu()
            ),
            'srm_scale': float(
                self.spatial.fusion.min_aux_scale
                + F.softplus(self.spatial.fusion.srm_scale_raw.detach()).cpu()
            ),
            'freq_mode': str(self.spatial._freq_mode),
            'wavelet_level': int(self.spatial._wavelet_level),
            'wavelet_type': str(self.spatial._wavelet_type),
            'model_weights': model_weights,
        }
        if epoch is not None:
            ckpt['epoch'] = epoch
        if optimizer_state is not None:
            ckpt['optimizer_state_dict'] = optimizer_state
        if scheduler_state is not None:
            ckpt['scheduler_state_dict'] = scheduler_state
        if metrics is not None:
            ckpt['metrics'] = metrics
        if ema_state is not None:
            ckpt['ema_state_dict'] = ema_state
        if scaler_state is not None:
            ckpt['scaler_state_dict'] = scaler_state
        if training_state is not None:
            ckpt['training_state'] = training_state
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, device: str = 'cpu') -> dict:
        ckpt = torch_load_checkpoint(path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)

        adapter_key = "spatial._wavelet_ml_adapter.0.weight"
        if adapter_key in state:
            ckpt_channels = int(state[adapter_key].shape[1])
            if ckpt_channels != int(self.spatial._wavelet_level):
                self.spatial._wavelet_level = ckpt_channels
                self.spatial._wavelet_ml_transform.level = ckpt_channels
                self.spatial._wavelet_ml_adapter[0] = nn.Conv2d(
                    ckpt_channels, 3, kernel_size=1, bias=False
                )
                self.spatial._wavelet_ml_adapter[0].to(next(self.parameters()).device)

        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            logger.warning(
                f"[load_checkpoint] {len(missing)} missing keys "
                f"(e.g. {missing[:3]}) — left at default init."
            )
        if unexpected:
            logger.warning(
                f"[load_checkpoint] {len(unexpected)} unexpected keys "
                f"(e.g. {unexpected[:3]}) — ignored."
            )
        return ckpt

    def load_spatial_weights(self, tristream_checkpoint: str, device: str = 'cpu') -> None:
        """Bootstrap shape-compatible spatial weights from a frame-level checkpoint."""
        ckpt = torch_load_checkpoint(tristream_checkpoint, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        own = self.spatial.state_dict()
        loaded, skipped = 0, 0
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
            else:
                skipped += 1
        self.spatial.load_state_dict(own)
        logger.info(
            "Loaded %d spatial weights, skipped %d shape-mismatched or temporal-head weights.",
            loaded,
            skipped,
        )
