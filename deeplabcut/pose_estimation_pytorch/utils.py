#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.config.pose import DetectorConfig, PoseConfig


def fix_seeds(seed: int) -> None:
    """Fixes the random seed for python, numpy and pytorch.

    Args:
        seed: the seed to set
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class ResolvedDevices:
    """Devices resolved for a pose model and its (optional) object detector."""

    pose: str
    detector: str | None


def _model_supports_mps(model_config: PoseConfig | DetectorConfig) -> bool:
    """Whether ``device: auto`` may resolve to MPS for this model."""
    if isinstance(model_config, DetectorConfig):
        # Legacy policy: detectors never auto-resolve to MPS. Removed when the
        # detector MPS capability flip lands.
        return False
    return "resnet" in model_config.get("net_type", "")


def resolve_model_device(
    model_config: PoseConfig | DetectorConfig,
    override: str | None = None,
) -> str:
    """Resolves the device for a single model.

    An explicit device (from ``override`` or the config) is returned verbatim.
    ``auto`` resolves to the first available of CUDA, MPS (when the model
    supports it) and CPU.

    Args:
        model_config: The pose or detector configuration.
        override: If set, takes precedence over ``model_config.device``.

    Returns:
        the device on which the model should run
    """
    device = override if override is not None else model_config.device
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif _model_supports_mps(model_config) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def resolve_device(model_config: PoseConfig | DetectorConfig) -> str:
    """Determines which device should be used from the model config.

    Compatibility wrapper around :func:`resolve_model_device`.

    Args:
        model_config: The PyTorch pose or detector configuration.

    Returns:
        the device on which training should be run
    """
    return resolve_model_device(model_config)


def _legacy_detector_fallback(device: str) -> str:
    """Reproduces the historical detector MPS-to-CPU fallback.

    Transitional: call sites route through this instead of inline guards so the
    behavior change can land as a single, separable commit that removes it.
    """
    if device == "mps":
        return "cpu"
    return device


def resolve_pose_and_detector_devices(
    model_config: PoseConfig,
    *,
    device: str | None = None,
    detector_device: str | None = None,
) -> ResolvedDevices:
    """Resolves the devices for a pose model and its detector in one place.

    Precedence for the pose model: explicit ``device`` argument, then
    ``PoseConfig.device``. For the detector: explicit ``detector_device``
    argument, then the shared ``device`` argument, then the detector's own
    config. Under the legacy policy still in place, a detector whose device
    would resolve to MPS falls back to the CPU unless ``detector_device``
    requests MPS explicitly.

    Args:
        model_config: The pose configuration (with or without a detector).
        device: Optional device override applied to both models.
        detector_device: Optional detector-only override; wins over ``device``.

    Returns:
        the resolved (pose, detector) devices; ``detector`` is None when the
        config has no detector

    Raises:
        ValueError: if ``detector_device`` requests a device other than "cpu"
            for a configuration without a detector
    """
    detector_config = model_config.detector
    if detector_config is None:
        if detector_device is not None:
            if detector_device == "cpu":
                warnings.warn(
                    "detector_device='cpu' is a no-op for a configuration "
                    "without a detector.",
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "detector_device was given, but this model configuration "
                    "has no detector (bottom-up models run without one)."
                )
        return ResolvedDevices(pose=resolve_model_device(model_config, device), detector=None)

    pose = resolve_model_device(model_config, device)
    if detector_device is not None:
        detector = resolve_model_device(detector_config, detector_device)
    else:
        # Legacy policy: the detector follows the pose device with an
        # MPS-to-CPU fallback, matching the historical guards.
        detector = _legacy_detector_fallback(pose)
    return ResolvedDevices(pose=pose, detector=detector)
