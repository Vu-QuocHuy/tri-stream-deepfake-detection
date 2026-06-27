"""Torch multi-level wavelet transform for the frequency stream."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLevelWaveletTransform(nn.Module):
    """Compute per-level DWT edge-energy channels."""

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 3,
        target_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.target_size = target_size

        name = str(wavelet).lower()
        if name != "db4":
            raise ValueError("Only 'db4' is supported for torch DWT")
        self._wavelet_name = name

        dec_lo, dec_hi = self._get_dec_filters()
        k = dec_lo.shape[0]
        ll = torch.ger(dec_lo, dec_lo).view(1, 1, k, k)
        lh = torch.ger(dec_hi, dec_lo).view(1, 1, k, k)
        hl = torch.ger(dec_lo, dec_hi).view(1, 1, k, k)
        hh = torch.ger(dec_hi, dec_hi).view(1, 1, k, k)
        kernels = torch.cat([ll, lh, hl, hh], dim=0)
        self.register_buffer("_dwt_kernels", kernels)
        self._kernel_size = k
        self._kernel_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _apply(self, fn):
        result = super()._apply(fn)
        self._kernel_cache.clear()
        return result

    def _kernels_for(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        cached = self._kernel_cache.get(key)
        if cached is None:
            cached = self._dwt_kernels.to(dtype=x.dtype, device=x.device)
            self._kernel_cache[key] = cached
        return cached

    def _extract_multilevel_energy(self, x: torch.Tensor) -> list[torch.Tensor]:
        energy_bands: list[torch.Tensor] = []
        cur = x
        pad = self._kernel_size // 2
        kernels = self._kernels_for(cur)

        for _ in range(self.level):
            cur_padded = F.pad(cur, (pad, pad, pad, pad), mode="replicate")
            bands = F.conv2d(cur_padded, kernels, stride=2)
            ll = bands[:, 0:1]
            lh = bands[:, 1:2]
            hl = bands[:, 2:3]
            hh = bands[:, 3:4]

            energy_sq = lh.float().square() + hl.float().square() + hh.float().square()
            energy = torch.sqrt(energy_sq + 1e-8)
            energy_bands.append(energy)
            cur = ll

        energy_bands.reverse()
        return energy_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.shape
        target_h = self.target_size[0] if self.target_size else H
        target_w = self.target_size[1] if self.target_size else W

        if C >= 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x[:, 0:1]

        energy_bands = self._extract_multilevel_energy(gray)

        resized: list[torch.Tensor] = []
        for band in energy_bands:
            t = F.interpolate(
                band,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            resized.append(t.squeeze(1))

        return torch.stack(resized, dim=1)

    def _get_dec_filters(self) -> tuple[torch.Tensor, torch.Tensor]:
        dec_lo = torch.tensor([
            -0.010597401784997278,
             0.032883011666982945,
             0.030841381835560764,
            -0.18703481171888114,
            -0.02798376941698385,
             0.6308807679298587,
             0.7148465705529154,
             0.23037781330885523,
        ], dtype=torch.float32)
        dec_hi = torch.tensor([
            -0.23037781330885523,
             0.7148465705529154,
            -0.6308807679298587,
            -0.02798376941698385,
             0.18703481171888114,
             0.030841381835560764,
            -0.032883011666982945,
            -0.010597401784997278,
        ], dtype=torch.float32)
        return dec_lo, dec_hi