"""Multi-stream spatial encoder and feature fusion model."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from tri_stream_deepfake_detection.utils.checkpoint import torch_load_checkpoint
from tri_stream_deepfake_detection.models.wavelet_stream import MultiLevelWaveletTransform


logger = logging.getLogger(__name__)

VALID_STREAMS = ("rgb", "freq", "srm")


def normalize_active_streams(active_streams: Optional[List[str]] = None) -> Tuple[str, ...]:
    if active_streams is None:
        return VALID_STREAMS
    streams = tuple(str(s).strip().lower() for s in active_streams if str(s).strip())
    if not streams:
        raise ValueError("active_streams must contain at least one stream")
    invalid = [s for s in streams if s not in VALID_STREAMS]
    if invalid:
        raise ValueError(f"Invalid active streams: {invalid}; expected subset of {VALID_STREAMS}")
    if len(set(streams)) != len(streams):
        raise ValueError(f"Duplicate active streams are not allowed: {streams}")
    return streams

def _effnet_input_size(model_name: str) -> int:
    sizes = {"b0": 224, "b1": 240, "b2": 260, "b3": 300,
             "b4": 380, "b5": 456, "b6": 528, "b7": 600}
    name = model_name.lower()
    for k, v in sizes.items():
        if k in name:
            return v
    return 224


def _effnet_feat_dim(model_name: str) -> int:
    dims = {"b0": 1280, "b1": 1280, "b2": 1408, "b3": 1536,
            "b4": 1792, "b5": 2048, "b6": 2304, "b7": 2560}
    name = model_name.lower()
    for k, v in dims.items():
        if k in name:
            return v
    return 1280


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB to grayscale using ITU-R BT.601 weights."""
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


class SpatialAttentionGate(nn.Module):
    """CBAM-style spatial attention before global pooling."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * attn


class ChannelAttentionFusion(nn.Module):
    """Learn attention weights over stream features and return their weighted sum."""

    def __init__(
        self,
        feat_dim: int,
        num_streams: int = 3,
        reduction: int = 16,
        stream_dropout_p: float = 0.0,
        freq_logit_bias: float = 0.0,
        srm_logit_bias: float = 0.0,
        stream_names: Optional[List[str]] = None,
    ):
        super().__init__()
        hidden = max(feat_dim // reduction, 32)
        self.stream_names = normalize_active_streams(stream_names)
        if int(num_streams) != len(self.stream_names):
            raise ValueError(
                f"num_streams={num_streams} does not match stream_names={self.stream_names}"
            )
        self.num_streams = num_streams
        self.stream_dropout_p = float(stream_dropout_p)
        self.freq_logit_bias = nn.Parameter(
            torch.tensor(float(freq_logit_bias), dtype=torch.float32)
        )
        self.srm_logit_bias = nn.Parameter(
            torch.tensor(float(srm_logit_bias), dtype=torch.float32)
        )
        self.min_aux_scale = 0.3
        init_scale = torch.log(torch.expm1(torch.tensor(1.0 - self.min_aux_scale)))
        self.freq_scale_raw = nn.Parameter(init_scale.clone().view(1))
        self.srm_scale_raw = nn.Parameter(init_scale.clone().view(1))
        self.attn = nn.Sequential(
            nn.Linear(feat_dim * num_streams, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_streams),
        )
        self.stream_norms = nn.ModuleList(
            [nn.LayerNorm(feat_dim) for _ in range(num_streams)]
        )

    def forward(
        self,
        feats: List[torch.Tensor],
        return_weights: bool = False,
    ):
        if len(feats) != self.num_streams:
            raise ValueError(
                f"Expected {self.num_streams} feature streams {self.stream_names}, got {len(feats)}"
            )
        feats_norm = [
            self.stream_norms[i](feat) if i < len(self.stream_norms) else feat
            for i, feat in enumerate(feats)
        ]
        cat = torch.cat(feats_norm, dim=1)
        logits = self.attn(cat)
        bias_values = []
        for name in self.stream_names:
            if name == "freq":
                bias_values.append(self.freq_logit_bias.to(dtype=logits.dtype))
            elif name == "srm":
                bias_values.append(self.srm_logit_bias.to(dtype=logits.dtype))
            else:
                bias_values.append(logits.new_zeros(()))
        logits = logits + torch.stack(bias_values).view(1, -1)

        if self.training and self.stream_dropout_p > 0.0:
            B = logits.shape[0]
            keep_mask = torch.rand(
                B, self.num_streams, device=logits.device
            ) > self.stream_dropout_p
            no_keep = ~keep_mask.any(dim=1)
            if no_keep.any():
                fix_indices = torch.where(no_keep)[0]
                num_fix = fix_indices.shape[0]
                if num_fix:
                    rand_idx = torch.randint(0, self.num_streams, (num_fix,), device=logits.device)
                    keep_mask[fix_indices, rand_idx] = True
            logits = logits.masked_fill(~keep_mask, float("-inf"))

        w = F.softmax(logits, dim=1)

        feats_scaled = list(feats_norm)
        freq_scale = self.min_aux_scale + F.softplus(self.freq_scale_raw)
        srm_scale = self.min_aux_scale + F.softplus(self.srm_scale_raw)
        for i, name in enumerate(self.stream_names):
            if name == "freq":
                feats_scaled[i] = freq_scale * feats_scaled[i]
            elif name == "srm":
                feats_scaled[i] = srm_scale * feats_scaled[i]
        fused = sum(w[:, i : i + 1] * f for i, f in enumerate(feats_scaled))
        
        if return_weights:
            return fused, w
        return fused

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for old_name, new_name in (
            ("freq_scale", "freq_scale_raw"),
            ("srm_scale", "srm_scale_raw"),
        ):
            old_key = prefix + old_name
            new_key = prefix + new_name
            if old_key in state_dict and new_key not in state_dict:
                scale = state_dict.pop(old_key).detach().clone()
                shifted = torch.clamp(scale - self.min_aux_scale, min=1e-6)
                state_dict[new_key] = torch.log(torch.expm1(shifted))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class MultiStreamDeepFakeDetector(nn.Module):
    """Spatial RGB/frequency/SRM encoder with channel-attention fusion."""

    def __init__(
        self,
        rgb_model: str = "efficientnet-b4",
        freq_model: str = "efficientnet-b4",
        srm_model: str = "efficientnet-b4",
        srm_filters: int = 30,
        stream_dropout_p: float = 0.0,
        freq_logit_bias: float = 0.0,
        srm_logit_bias: float = 0.0,
        freq_mode: str = "wavelet_ml",
        wavelet_level: int = 1,
        wavelet_type: str = "db4",
        pretrained: bool = True,
        num_classes: int = 2,
        active_streams: Optional[List[str]] = None,
    ):
        super().__init__()
        self.active_streams = normalize_active_streams(active_streams)

        rgb_backbone = str(rgb_model)
        freq_backbone = str(freq_model) if str(freq_model).strip() else rgb_backbone
        srm_backbone = str(srm_model) if str(srm_model).strip() else rgb_backbone

        self._backbone_name = rgb_backbone
        self._rgb_backbone_name = rgb_backbone
        self._freq_backbone_name = freq_backbone
        self._srm_backbone_name = srm_backbone

        self._rgb_input_size = _effnet_input_size(rgb_backbone)
        self._freq_input_size = _effnet_input_size(freq_backbone)
        self._srm_input_size = _effnet_input_size(srm_backbone)
        self._srm_filters = int(srm_filters)

        self._rgb_feat_dim = _effnet_feat_dim(rgb_backbone)
        self._freq_feat_dim = _effnet_feat_dim(freq_backbone)
        self._srm_feat_dim = _effnet_feat_dim(srm_backbone)
        self._feat_dim = self._rgb_feat_dim
        mode = str(freq_mode).lower()
        if mode != "wavelet_ml":
            raise ValueError(f"freq_mode must be 'wavelet_ml' (got {freq_mode!r})")
        self._freq_mode = mode
        self._wavelet_level = max(1, int(wavelet_level))
        self._wavelet_type = str(wavelet_type).lower()
        if self._wavelet_type != "db4":
            raise ValueError(f"wavelet_type must be 'db4' (got {wavelet_type!r})")

        self.rgb_encoder = self._build_encoder(rgb_backbone, pretrained)
        self.freq_encoder = self._build_encoder(freq_backbone, pretrained)
        self.srm_encoder = self._build_encoder(srm_backbone, pretrained)

        self._proj_rgb = (
            nn.Identity()
            if self._rgb_feat_dim == self._feat_dim
            else nn.Linear(self._rgb_feat_dim, self._feat_dim)
        )
        self._proj_freq = (
            nn.Identity()
            if self._freq_feat_dim == self._feat_dim
            else nn.Linear(self._freq_feat_dim, self._feat_dim)
        )
        self._proj_srm = (
            nn.Identity()
            if self._srm_feat_dim == self._feat_dim
            else nn.Linear(self._srm_feat_dim, self._feat_dim)
        )

        self._srm_conv = nn.Conv2d(1, srm_filters, kernel_size=5, padding=2, bias=False)
        self._srm_to3 = nn.Conv2d(srm_filters, 3, kernel_size=1, bias=False)
        self._srm_bn = nn.GroupNorm(1, 3)
        self._init_srm_kernels()

        self._freq_adapter = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self._freq_adapter[-1].weight)
        self._freq_mix = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))

        self._wavelet_ml_transform = MultiLevelWaveletTransform(
            wavelet=self._wavelet_type,
            level=self._wavelet_level,
        )
        self._wavelet_ml_adapter = nn.Sequential(
            nn.Conv2d(self._wavelet_level, 3, kernel_size=1, bias=False),
            nn.GroupNorm(1, 3),
        )
        self._freq_layer_norm = nn.LayerNorm(self._feat_dim)

        self._spatial_attn = nn.ModuleDict({
            'rgb': SpatialAttentionGate(),
            'freq': SpatialAttentionGate(),
            'srm': SpatialAttentionGate(),
        })

        self.fusion = ChannelAttentionFusion(
            feat_dim=self._feat_dim,
            num_streams=len(self.active_streams),
            reduction=16,
            stream_dropout_p=stream_dropout_p,
            freq_logit_bias=freq_logit_bias,
            srm_logit_bias=srm_logit_bias,
            stream_names=list(self.active_streams),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self._feat_dim),
            nn.Dropout(0.5),
            nn.Linear(self._feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_classes),
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def _expand_fusion_weights(self, w: torch.Tensor) -> torch.Tensor:
        full = w.new_zeros(w.shape[0], len(VALID_STREAMS))
        for i, name in enumerate(self.active_streams):
            full[:, VALID_STREAMS.index(name)] = w[:, i]
        return full

    @staticmethod
    def _build_encoder(model_name: str, pretrained: bool) -> EfficientNet:
        if pretrained:
            return EfficientNet.from_pretrained(model_name)
        return EfficientNet.from_name(model_name)

    def _init_srm_kernels(self) -> None:
        """Initialise SRM conv with a diverse high-pass filter bank."""
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
        lap3[1:4, 1:4] = torch.tensor([
            [0, -1, 0], [-1, 4, -1], [0, -1, 0]
        ], dtype=torch.float32)
        filters.append(lap3)

        lap5 = torch.tensor([
            [0, 0, -1, 0, 0],
            [0, 0, -2, 0, 0],
            [-1, -2, 16, -2, -1],
            [0, 0, -2, 0, 0],
            [0, 0, -1, 0, 0],
        ], dtype=torch.float32)
        filters.append(lap5)

        sq3 = torch.zeros(5, 5)
        sq3[1:4, 1:4] = torch.tensor([
            [-1, 2, -1], [2, -4, 2], [-1, 2, -1]
        ], dtype=torch.float32)
        filters.append(sq3)

        diag = torch.zeros(5, 5)
        diag[0, 0] = 1; diag[1, 1] = -2; diag[2, 2] = 1
        filters.append(diag)

        base_count = len(filters)
        noise_gen = torch.Generator()
        noise_gen.manual_seed(42)
        while len(filters) < self._srm_filters:
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

    def _srm_highpass_weight(self) -> torch.Tensor:
        """Return SRM kernels constrained to high-pass, normalized form."""
        w = self._srm_conv.weight
        w = w - w.mean(dim=(-2, -1), keepdim=True)
        return w / (w.abs().sum(dim=(-2, -1), keepdim=True) + 1e-6)

    def _encode(
        self, encoder: EfficientNet, x: torch.Tensor,
        spatial_attn: Optional[SpatialAttentionGate] = None,
    ) -> torch.Tensor:
        """Forward through EfficientNet and return a pooled feature vector."""
        fmap = self._encode_feature_map(encoder, x, spatial_attn)
        return self._pool_feature_map(encoder, fmap)

    def _encode_feature_map(
        self,
        encoder: EfficientNet,
        x: torch.Tensor,
        spatial_attn: Optional[SpatialAttentionGate] = None,
    ) -> torch.Tensor:
        fmap = encoder.extract_features(x)
        if spatial_attn is not None:
            fmap = spatial_attn(fmap)
        return fmap

    @staticmethod
    def _pool_feature_map(encoder: EfficientNet, fmap: torch.Tensor) -> torch.Tensor:
        x = encoder._avg_pooling(fmap)
        x = x.flatten(start_dim=1)
        x = encoder._dropout(x)
        return x

    @staticmethod
    def _tokens_from_feature_map(
        encoder: EfficientNet,
        fmap: torch.Tensor,
        projector: nn.Module,
        token_grid: int,
    ) -> torch.Tensor:
        """Convert a feature map to local spatial tokens."""
        g = max(1, int(token_grid))
        tokens = F.adaptive_avg_pool2d(fmap, output_size=(g, g))
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = encoder._dropout(tokens)
        return projector(tokens)

    def _compute_freq_channels(self, x_rgb: torch.Tensor) -> torch.Tensor:
        return self._compute_freq_channels_wavelet_ml(x_rgb)

    def _compute_freq_channels_wavelet_ml(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """Compute multi-level wavelet frequency channels."""
        x01 = self._denorm(x_rgb)
        x01 = F.interpolate(
            x01,
            size=(self._freq_input_size, self._freq_input_size),
            mode="bilinear",
            align_corners=False,
        )

        wav = self._wavelet_ml_transform(x01)
        wav_3ch = self._wavelet_ml_adapter(wav)
        freq01 = torch.sigmoid(wav_3ch)

        adapt = torch.tanh(self._freq_adapter(freq01))
        mix = torch.clamp(self._freq_mix, 0.0, 0.35)
        freq01 = (freq01 + mix * adapt).clamp(0.0, 1.0)

        return self._renorm(freq01)

    def _compute_srm_channels(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """Apply learnable high-pass filters and compress residuals to 3 channels."""
        x01 = self._denorm(x_rgb)
        gray = _rgb_to_gray(x01)
        gray = F.interpolate(
            gray, size=(self._srm_input_size, self._srm_input_size),
            mode="bilinear", align_corners=False
        )
        noise = torch.abs(F.conv2d(gray, self._srm_highpass_weight(), padding=2))
        out3 = self._srm_bn(self._srm_to3(noise))
        out01 = torch.sigmoid(out3)
        return self._renorm(out01)

    def _encode_stream_vector(self, stream: str, x_rgb: torch.Tensor) -> torch.Tensor:
        if stream == "rgb":
            return self._proj_rgb(self._encode(self.rgb_encoder, x_rgb, self._spatial_attn["rgb"]))
        if stream == "freq":
            x_freq = self._compute_freq_channels(x_rgb)
            feat = self._proj_freq(self._encode(self.freq_encoder, x_freq, self._spatial_attn["freq"]))
            return self._freq_layer_norm(feat)
        if stream == "srm":
            x_srm = self._compute_srm_channels(x_rgb)
            return self._proj_srm(self._encode(self.srm_encoder, x_srm, self._spatial_attn["srm"]))
        raise ValueError(f"Unsupported stream: {stream}")

    def _encode_active_vectors(self, x_rgb: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        feats_by_stream = {
            stream: self._encode_stream_vector(stream, x_rgb)
            for stream in self.active_streams
        }
        return [feats_by_stream[s] for s in self.active_streams], feats_by_stream

    def _encode_stream_tokens(
        self,
        stream: str,
        x_rgb: torch.Tensor,
        token_grid: int,
    ) -> torch.Tensor:
        if stream == "rgb":
            fmap = self._encode_feature_map(
                self.rgb_encoder, x_rgb, self._spatial_attn["rgb"]
            )
            return self._tokens_from_feature_map(
                self.rgb_encoder, fmap, self._proj_rgb, token_grid
            )
        if stream == "freq":
            x_freq = self._compute_freq_channels(x_rgb)
            fmap = self._encode_feature_map(
                self.freq_encoder, x_freq, self._spatial_attn["freq"]
            )
            tokens = self._tokens_from_feature_map(
                self.freq_encoder, fmap, self._proj_freq, token_grid
            )
            return self._freq_layer_norm(tokens)
        if stream == "srm":
            x_srm = self._compute_srm_channels(x_rgb)
            fmap = self._encode_feature_map(
                self.srm_encoder, x_srm, self._spatial_attn["srm"]
            )
            return self._tokens_from_feature_map(
                self.srm_encoder, fmap, self._proj_srm, token_grid
            )
        raise ValueError(f"Unsupported stream: {stream}")

    def _encode_active_tokens(
        self,
        x_rgb: torch.Tensor,
        token_grid: int,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        tokens_by_stream = {
            stream: self._encode_stream_tokens(stream, x_rgb, token_grid)
            for stream in self.active_streams
        }
        return [tokens_by_stream[s] for s in self.active_streams], tokens_by_stream

    def encode_frame(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """Extract a fused feature vector for one frame."""
        if x_rgb.shape[-1] != self._rgb_input_size or x_rgb.shape[-2] != self._rgb_input_size:
            x_rgb = F.interpolate(
                x_rgb, size=(self._rgb_input_size, self._rgb_input_size),
                mode="bilinear", align_corners=False,
            )
        feats, _ = self._encode_active_vectors(x_rgb)
        return self.fusion(feats)

    def encode_frame_full(
        self, x_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused features, per-stream features, and fusion weights."""
        if x_rgb.shape[-1] != self._rgb_input_size or x_rgb.shape[-2] != self._rgb_input_size:
            x_rgb = F.interpolate(
                x_rgb, size=(self._rgb_input_size, self._rgb_input_size),
                mode="bilinear", align_corners=False,
            )
        feats, feats_by_stream = self._encode_active_vectors(x_rgb)
        fused, w = self.fusion(feats, return_weights=True)
        return (
            fused,
            feats_by_stream.get("rgb"),
            feats_by_stream.get("freq"),
            feats_by_stream.get("srm"),
            self._expand_fusion_weights(w),
        )

    def encode_frame_tokens(
        self,
        x_rgb: torch.Tensor,
        token_grid: int = 2,
    ) -> torch.Tensor:
        """Extract fused local spatial tokens for one frame."""
        fused_tokens, _, _, _, _ = self.encode_frame_tokens_full(
            x_rgb, token_grid=token_grid
        )
        return fused_tokens

    def encode_frame_tokens_full(
        self,
        x_rgb: torch.Tensor,
        token_grid: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused tokens, per-stream token means, and fusion weights."""
        if x_rgb.shape[-1] != self._rgb_input_size or x_rgb.shape[-2] != self._rgb_input_size:
            x_rgb = F.interpolate(
                x_rgb, size=(self._rgb_input_size, self._rgb_input_size),
                mode="bilinear", align_corners=False,
            )

        tokens, tokens_by_stream = self._encode_active_tokens(x_rgb, token_grid)
        B, K, D = tokens[0].shape
        flat_feats = [tok.reshape(B * K, D) for tok in tokens]
        fused_flat, w_flat = self.fusion(flat_feats, return_weights=True)
        fused_tokens = fused_flat.view(B, K, D)
        w = self._expand_fusion_weights(w_flat).view(B, K, len(VALID_STREAMS)).mean(dim=1)

        return (
            fused_tokens,
            tokens_by_stream.get("rgb").mean(dim=1) if "rgb" in tokens_by_stream else None,
            tokens_by_stream.get("freq").mean(dim=1) if "freq" in tokens_by_stream else None,
            tokens_by_stream.get("srm").mean(dim=1) if "srm" in tokens_by_stream else None,
            w,
        )

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        if x_rgb.shape[-1] != self._rgb_input_size or x_rgb.shape[-2] != self._rgb_input_size:
            x_rgb = F.interpolate(
                x_rgb, size=(self._rgb_input_size, self._rgb_input_size),
                mode="bilinear", align_corners=False,
            )

        feats, _ = self._encode_active_vectors(x_rgb)
        fused = self.fusion(feats)

        return self.classifier(fused)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: Optional[int] = None,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        ckpt: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "architecture": "multi_stream",
            "label_convention": "real=0,fake=1",
            "score_target": "fake",
            "backbone": self._backbone_name,
            "rgb_backbone": self._rgb_backbone_name,
            "freq_backbone": self._freq_backbone_name,
            "srm_backbone": self._srm_backbone_name,
            "active_streams": list(self.active_streams),
            "srm_filters": int(self._srm_filters),
            "stream_dropout_p": float(self.fusion.stream_dropout_p),
            "freq_logit_bias": float(self.fusion.freq_logit_bias.detach().cpu()),
            "srm_logit_bias": float(self.fusion.srm_logit_bias.detach().cpu()),
            "freq_scale": float(
                self.fusion.min_aux_scale
                + F.softplus(self.fusion.freq_scale_raw.detach()).cpu()
            ),
            "srm_scale": float(
                self.fusion.min_aux_scale
                + F.softplus(self.fusion.srm_scale_raw.detach()).cpu()
            ),
            "num_classes": self.classifier[-1].out_features,
            "freq_mode": str(self._freq_mode),
            "wavelet_level": int(self._wavelet_level),
            "wavelet_type": str(self._wavelet_type),
        }
        if epoch is not None:
            ckpt["epoch"] = epoch
        if optimizer_state is not None:
            ckpt["optimizer_state_dict"] = optimizer_state
        if scheduler_state is not None:
            ckpt["scheduler_state_dict"] = scheduler_state
        if metrics is not None:
            ckpt["metrics"] = metrics
        torch.save(ckpt, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> dict:
        ckpt = torch_load_checkpoint(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        adapter_key = "_wavelet_ml_adapter.0.weight"
        if adapter_key in state_dict:
            ckpt_channels = state_dict[adapter_key].shape[1]
            if ckpt_channels != self._wavelet_level:
                self._wavelet_level = ckpt_channels
                self._wavelet_ml_transform.level = ckpt_channels
                self._wavelet_ml_adapter[0] = nn.Conv2d(ckpt_channels, 3, kernel_size=1, bias=False)
                self._wavelet_ml_adapter[0].to(next(self.parameters()).device)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
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
        return ckpt if isinstance(ckpt, dict) else {}
