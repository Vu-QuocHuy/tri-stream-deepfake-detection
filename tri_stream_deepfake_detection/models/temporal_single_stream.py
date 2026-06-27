"""
Video-level single-stream temporal detector.

This is a baseline model: one spatial stream (rgb | freq | srm) encodes each
frame, then a temporal module aggregates the frame sequence into one video
prediction. It intentionally has no multi-stream fusion or auxiliary stream loss.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from tri_stream_deepfake_detection.utils.checkpoint import torch_load_checkpoint
from tri_stream_deepfake_detection.models.multistream import (
    SpatialAttentionGate,
    _effnet_feat_dim,
    _effnet_input_size,
    _rgb_to_gray,
)
from tri_stream_deepfake_detection.models.temporal import (
    TemporalDifferenceModule,
    TemporalTransformer,
)
from tri_stream_deepfake_detection.models.wavelet_stream import MultiLevelWaveletTransform


logger = logging.getLogger(__name__)


class SingleStreamFrameEncoder(nn.Module):
    """Encode one frame with exactly one stream: rgb, freq, or srm."""

    def __init__(
        self,
        stream: str = "rgb",
        backbone: str = "efficientnet-b4",
        srm_filters: int = 30,
        freq_mode: str = "wavelet_ml",
        wavelet_level: int = 3,
        wavelet_type: str = "db4",
        pretrained: bool = True,
    ):
        super().__init__()
        if stream not in ("rgb", "freq", "srm"):
            raise ValueError("stream must be 'rgb', 'freq', or 'srm'")
        mode = str(freq_mode).lower()
        if mode != "wavelet_ml":
            raise ValueError("freq_mode must be 'wavelet_ml'")

        self.stream = stream
        self.backbone_name = str(backbone)
        self.input_size = _effnet_input_size(self.backbone_name)
        self.feat_dim = _effnet_feat_dim(self.backbone_name)
        self.srm_filters = int(srm_filters)
        self.freq_mode = mode
        self.wavelet_level = max(1, int(wavelet_level))
        self.wavelet_type = str(wavelet_type).lower()
        if self.wavelet_type != "db4":
            raise ValueError("wavelet_type must be 'db4'")

        self.encoder = self._build_encoder(self.backbone_name, pretrained)
        self.spatial_attn = SpatialAttentionGate()

        self._freq_adapter = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self._freq_adapter[-1].weight)
        self._freq_mix = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))

        self._wavelet_ml_transform = MultiLevelWaveletTransform(
            wavelet=self.wavelet_type,
            level=self.wavelet_level,
        )
        self._wavelet_ml_adapter = nn.Sequential(
            nn.Conv2d(self.wavelet_level, 3, kernel_size=1, bias=False),
            nn.GroupNorm(1, 3),
        )
        self._freq_layer_norm = nn.LayerNorm(self.feat_dim)

        self._srm_conv = nn.Conv2d(1, self.srm_filters, kernel_size=5, padding=2, bias=False)
        self._srm_to3 = nn.Conv2d(self.srm_filters, 3, kernel_size=1, bias=False)
        self._srm_bn = nn.GroupNorm(1, 3)
        self._init_srm_kernels()

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    @staticmethod
    def _build_encoder(model_name: str, pretrained: bool) -> EfficientNet:
        if pretrained:
            return EfficientNet.from_pretrained(model_name)
        return EfficientNet.from_name(model_name)

    def _init_srm_kernels(self) -> None:
        filters: List[torch.Tensor] = []
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            k = torch.zeros(5, 5)
            k[2, 2] = -1.0
            k[2 + dx, 2 + dy] = 1.0
            filters.append(k)
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            k = torch.zeros(5, 5)
            k[2 - dx, 2 - dy] = 1.0
            k[2, 2] = -2.0
            k[2 + dx, 2 + dy] = 1.0
            filters.append(k)
        lap3 = torch.zeros(5, 5)
        lap3[1:4, 1:4] = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        )
        filters.append(lap3)
        lap5 = torch.tensor(
            [
                [0, 0, -1, 0, 0],
                [0, 0, -2, 0, 0],
                [-1, -2, 16, -2, -1],
                [0, 0, -2, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=torch.float32,
        )
        filters.append(lap5)
        sq3 = torch.zeros(5, 5)
        sq3[1:4, 1:4] = torch.tensor(
            [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=torch.float32
        )
        filters.append(sq3)
        diag = torch.zeros(5, 5)
        diag[0, 0] = 1.0
        diag[1, 1] = -2.0
        diag[2, 2] = 1.0
        filters.append(diag)

        base_count = len(filters)
        noise_gen = torch.Generator()
        noise_gen.manual_seed(42)
        while len(filters) < self.srm_filters:
            src = filters[len(filters) % base_count]
            scale = 0.5 + (len(filters) % 5) * 0.3
            filters.append(src * scale + 0.01 * torch.randn(5, 5, generator=noise_gen))

        with torch.no_grad():
            w = self._srm_conv.weight
            for k in range(min(w.shape[0], len(filters))):
                f = filters[k]
                f = f / (f.abs().sum() + 1e-6)
                w[k, 0].copy_(f)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self._std + self._mean).clamp(0.0, 1.0)

    def _renorm(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self._mean) / (self._std + 1e-6)

    def _resize_rgb(self, x_rgb: torch.Tensor) -> torch.Tensor:
        if x_rgb.shape[-2:] == (self.input_size, self.input_size):
            return x_rgb
        return F.interpolate(
            x_rgb,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

    def _compute_freq_channels(self, x_rgb: torch.Tensor) -> torch.Tensor:
        return self._compute_freq_channels_wavelet_ml(x_rgb)

    def _compute_freq_channels_wavelet_ml(self, x_rgb: torch.Tensor) -> torch.Tensor:
        x01 = self._denorm(x_rgb)
        wav = self._wavelet_ml_transform(x01)
        wav_3ch = self._wavelet_ml_adapter(wav)
        freq01 = torch.sigmoid(wav_3ch)
        adapt = torch.tanh(self._freq_adapter(freq01))
        mix = torch.clamp(self._freq_mix, 0.0, 0.35)
        return self._renorm((freq01 + mix * adapt).clamp(0.0, 1.0))

    def _compute_srm_channels(self, x_rgb: torch.Tensor) -> torch.Tensor:
        x01 = self._denorm(x_rgb)
        gray = _rgb_to_gray(x01)
        noise = torch.abs(self._srm_conv(gray))
        out3 = self._srm_bn(self._srm_to3(noise))
        return self._renorm(torch.sigmoid(out3))

    def _stream_input(self, x_rgb: torch.Tensor) -> torch.Tensor:
        x_rgb = self._resize_rgb(x_rgb)
        if self.stream == "rgb":
            return x_rgb
        if self.stream == "freq":
            return self._compute_freq_channels(x_rgb)
        return self._compute_srm_channels(x_rgb)

    def _encode_feature_map(self, x_stream: torch.Tensor) -> torch.Tensor:
        fmap = self.encoder.extract_features(x_stream)
        return self.spatial_attn(fmap)

    def encode_frame(self, x_rgb: torch.Tensor) -> torch.Tensor:
        x_stream = self._stream_input(x_rgb)
        fmap = self._encode_feature_map(x_stream)
        feat = self.encoder._avg_pooling(fmap).flatten(start_dim=1)
        feat = self.encoder._dropout(feat)
        if self.stream == "freq":
            feat = self._freq_layer_norm(feat)
        return feat

    def encode_frame_tokens(self, x_rgb: torch.Tensor, token_grid: int = 2) -> torch.Tensor:
        x_stream = self._stream_input(x_rgb)
        fmap = self._encode_feature_map(x_stream)
        g = max(1, int(token_grid))
        tokens = F.adaptive_avg_pool2d(fmap, output_size=(g, g))
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.encoder._dropout(tokens)
        if self.stream == "freq":
            tokens = self._freq_layer_norm(tokens)
        return tokens


class TemporalSingleStreamDetector(nn.Module):
    """Video-level detector with one spatial stream and temporal aggregation."""

    def __init__(
        self,
        backbone: str = "efficientnet-b4",
        single_stream: str = "rgb",
        n_frames: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        srm_filters: int = 30,
        spatial_token_grid: int = 2,
        freq_mode: str = "wavelet_ml",
        wavelet_level: int = 3,
        wavelet_type: str = "db4",
        pretrained: bool = True,
        bce_output: bool = True,
        use_grad_checkpoint: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = str(backbone)
        self.single_stream = single_stream
        self.n_frames = int(n_frames)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.srm_filters = int(srm_filters)
        self.spatial_token_grid = max(1, int(spatial_token_grid))
        self.freq_mode = str(freq_mode)
        self.wavelet_level = max(1, int(wavelet_level))
        self.wavelet_type = str(wavelet_type)
        self.dropout = float(dropout)
        self.bce_output = bool(bce_output)
        self.use_grad_checkpoint = bool(use_grad_checkpoint)
        self._phase = 2

        self.frame_encoder = SingleStreamFrameEncoder(
            stream=single_stream,
            backbone=backbone,
            srm_filters=srm_filters,
            freq_mode=freq_mode,
            wavelet_level=self.wavelet_level,
            wavelet_type=wavelet_type,
            pretrained=pretrained,
        )
        feat_dim = self.frame_encoder.feat_dim

        if self.use_grad_checkpoint:
            self._enable_grad_checkpointing()

        self.temporal = TemporalTransformer(
            d_model=feat_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            max_frames=(self.n_frames * self.spatial_token_grid * self.spatial_token_grid) + 4,
        )
        self.temporal_diff = TemporalDifferenceModule(feat_dim)

        token_count = self.spatial_token_grid * self.spatial_token_grid
        self.spatial_token_embed = nn.Parameter(torch.zeros(1, 1, token_count, feat_dim))

        out_dim = 1 if self.bce_output else 2
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
        self._feat_dim = feat_dim

    def set_phase(self, phase: int) -> None:
        assert phase in (1, 2), "phase must be 1 or 2"
        self._phase = int(phase)
        if phase == 1:
            for p in self.temporal.parameters():
                p.requires_grad = False
            for p in self.temporal_diff.parameters():
                p.requires_grad = False
            for p in self.frame_encoder.parameters():
                p.requires_grad = True
            self.spatial_token_embed.requires_grad = True
            for p in self.classifier.parameters():
                p.requires_grad = True
            for p in self.frame_scorer.parameters():
                p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = True

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.frame_encoder.encoder.parameters():
            p.requires_grad = not freeze

    def _enable_grad_checkpointing(self) -> None:
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        def make_checkpointed(original_fn):
            def _ckpt_forward(*args, **kwargs):
                return ckpt_fn(original_fn, *args, use_reentrant=False, **kwargs)

            return _ckpt_forward

        encoder = self.frame_encoder.encoder
        if hasattr(encoder, "set_grad_checkpointing"):
            encoder.set_grad_checkpointing(True)
        else:
            for block in encoder._blocks:
                block.forward = make_checkpointed(block.forward)

    def _encode_frames_batched(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        feat_flat = self.frame_encoder.encode_frame(x_flat)
        return feat_flat.view(B, T, -1)

    def _encode_frame_tokens_batched(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        tok_flat = self.frame_encoder.encode_frame_tokens(
            x_flat,
            token_grid=self.spatial_token_grid,
        )
        K = tok_flat.shape[1]
        return tok_flat.view(B, T, K, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T != self.n_frames:
            raise ValueError(
                f"Expected {self.n_frames} frames per video, got {T}. "
                "Ensure VideoSequenceDataset n_frames matches model n_frames."
            )

        use_spatial_tokens = self.spatial_token_grid > 1
        if not self.use_grad_checkpoint:
            if use_spatial_tokens:
                features = self._encode_frame_tokens_batched(x)
            else:
                features = self._encode_frames_batched(x)
        else:
            frame_features = []
            for t in range(T):
                if use_spatial_tokens:
                    feat = self.frame_encoder.encode_frame_tokens(
                        x[:, t],
                        token_grid=self.spatial_token_grid,
                    )
                else:
                    feat = self.frame_encoder.encode_frame(x[:, t])
                frame_features.append(feat)
            features = torch.stack(frame_features, dim=1)

        if use_spatial_tokens:
            features = features + self.spatial_token_embed[:, :, : features.shape[2], :]

        if self._phase != 1:
            if use_spatial_tokens:
                assert features.ndim == 4, f"Expected [B,T,K,D], got {tuple(features.shape)}"
            else:
                assert features.ndim == 3, f"Expected [B,T,D], got {tuple(features.shape)}"
            features = self.temporal_diff(features)
            if features.shape[-1] != self._feat_dim:
                raise RuntimeError(
                    f"Temporal difference changed feature dim from {self._feat_dim} "
                    f"to {features.shape[-1]}"
                )

        if self._phase == 1:
            frame_repr = features.mean(dim=2) if use_spatial_tokens else features
            frame_w = torch.softmax(self.frame_scorer(frame_repr), dim=1)
            video_repr = (frame_repr * frame_w).sum(dim=1)
        else:
            if use_spatial_tokens:
                Bf, Tf, Kf, Df = features.shape
                video_repr = self.temporal(features.reshape(Bf, Tf * Kf, Df))
            else:
                video_repr = self.temporal(features)

        logits = self.classifier(video_repr)
        if self.bce_output:
            logits = logits.squeeze(-1)
        return logits

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
    ) -> None:
        ckpt: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "architecture": "temporal_single_stream",
            "label_convention": "real=0,fake=1",
            "score_target": "fake",
            "backbone": self.backbone_name,
            "single_stream": self.single_stream,
            "n_frames": self.n_frames,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "srm_filters": self.srm_filters,
            "spatial_token_grid": self.spatial_token_grid,
            "freq_mode": self.freq_mode,
            "wavelet_level": self.wavelet_level,
            "wavelet_type": self.wavelet_type,
            "dropout": self.dropout,
            "bce_output": self.bce_output,
            "out_dim": self.classifier[-1].out_features,
        }
        if epoch is not None:
            ckpt["epoch"] = epoch
        if optimizer_state is not None:
            ckpt["optimizer_state_dict"] = optimizer_state
        if scheduler_state is not None:
            ckpt["scheduler_state_dict"] = scheduler_state
        if metrics is not None:
            ckpt["metrics"] = metrics
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, device: str = "cpu") -> dict:
        ckpt = torch_load_checkpoint(path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)

        adapter_key = "frame_encoder._wavelet_ml_adapter.0.weight"
        if adapter_key in state:
            ckpt_channels = int(state[adapter_key].shape[1])
            if ckpt_channels != int(self.frame_encoder.wavelet_level):
                self.wavelet_level = ckpt_channels
                self.frame_encoder.wavelet_level = ckpt_channels
                self.frame_encoder._wavelet_ml_transform.level = ckpt_channels
                self.frame_encoder._wavelet_ml_adapter[0] = nn.Conv2d(
                    ckpt_channels, 3, kernel_size=1, bias=False
                )
                self.frame_encoder._wavelet_ml_adapter[0].to(next(self.parameters()).device)

        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            logger.warning("[load_checkpoint] %d missing keys (e.g. %s)", len(missing), missing[:3])
        if unexpected:
            logger.warning(
                "[load_checkpoint] %d unexpected keys (e.g. %s)",
                len(unexpected),
                unexpected[:3],
            )
        return ckpt if isinstance(ckpt, dict) else {}
