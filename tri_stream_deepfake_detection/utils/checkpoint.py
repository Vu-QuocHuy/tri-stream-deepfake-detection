"""Checkpoint loading helpers."""

from __future__ import annotations

import inspect

import torch


def torch_load_checkpoint(path: str, map_location: str = "cpu"):
    """Load checkpoint using weights_only when the installed torch supports it."""
    if "weights_only" in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location)
