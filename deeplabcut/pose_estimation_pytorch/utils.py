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

import functools
import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from packaging.version import InvalidVersion, Version

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


MIN_TORCH_FOR_DETECTOR_MPS = (2, 12)
"""Minimum torch version for which running detectors on Apple MPS is supported.

Older versions are known to hang or silently produce wrong results (see
DeepLabCut#3155 and DeepLabCut#2853); requests for detector MPS below this
floor raise instead of silently falling back.
"""

DETECTOR_MPS_VALIDATED_VARIANTS: frozenset[str] = frozenset({"ssdlite"})
"""Detector variants validated on Apple MPS (training + inference numerically
checked against CPU runs). ``device: auto`` only resolves to MPS for these;
explicit MPS requests for other variants run with a warning.

ssdlite: validated on Apple Silicon with torch 2.12.1 / torchvision 0.27.1 —
training losses within 1% of CPU per epoch, inference bounding boxes and
keypoints matching CPU within 1e-4 px (p95) over a 9000-frame real video, and
substantially faster than the CPU."""


@functools.lru_cache(maxsize=1)
def torch_meets_detector_mps_floor() -> bool:
    """Whether the installed torch version supports detectors on Apple MPS.

    Pre-releases and nightlies of the floor version do not qualify: validation
    was performed against released builds.
    """
    floor = ".".join(str(v) for v in MIN_TORCH_FOR_DETECTOR_MPS)
    try:
        return Version(torch.__version__) >= Version(floor)
    except InvalidVersion:
        return False


def is_mps_device(device: str | torch.device | None) -> bool:
    """Whether a device specification targets Apple MPS ("mps", "mps:0", ...)."""
    return device is not None and str(device).startswith("mps")


def detector_variant(config: DetectorConfig | dict) -> str | None:
    """Returns the canonical variant name of a detector config, if known."""
    model_cfg = config["model"] if isinstance(config, dict) else config.model
    variant = model_cfg.get("variant")
    if variant:
        return str(variant)
    if str(model_cfg.get("type", "")).lower() == "ssdlite":
        return "ssdlite"
    return None


def validate_detector_mps_request(variant: str | None, *, for_training: bool = False) -> None:
    """Checks an explicit request to run a detector on Apple MPS.

    Args:
        variant: The canonical detector variant name, if known.
        for_training: Whether the detector is about to be trained. Training
            unvalidated variants on MPS is refused outright: it has been
            observed to hang the GPU badly enough to trigger a system watchdog
            kernel panic (fasterrcnn_resnet50_fpn_v2 on Apple Silicon).

    Raises:
        RuntimeError: if the installed torch version is below the validated
            floor, where detectors are known to hang or corrupt results — or
            if an unvalidated variant would be trained on MPS.
    """
    if not torch_meets_detector_mps_floor():
        floor = ".".join(str(v) for v in MIN_TORCH_FOR_DETECTOR_MPS)
        raise RuntimeError(
            f"Running detectors on MPS requires torch >= {floor} (found "
            f"{torch.__version__}); older versions are known to hang or "
            'produce corrupted predictions. Use detector_device="cpu" (or '
            "detector.device: cpu in the model configuration) to keep the "
            "detector on the CPU."
        )
    if variant not in DETECTOR_MPS_VALIDATED_VARIANTS:
        if for_training:
            raise RuntimeError(
                f"Refusing to train detector variant {variant!r} on MPS: "
                "training unvalidated detectors on MPS can hang the GPU badly "
                "enough to trigger a system watchdog reboot (observed with "
                'fasterrcnn_resnet50_fpn_v2). Use detector_device="cpu" to '
                "train the detector on the CPU."
            )
        warnings.warn(
            f"Detector variant {variant!r} has not been validated on MPS. "
            "Compare predictions against a CPU run before trusting them.",
            stacklevel=3,
        )


def detector_auto_device(variant: str | None) -> str:
    """Auto device policy for a detector known only by its variant name."""
    if torch.cuda.is_available():
        return "cuda"
    if (
        torch_meets_detector_mps_floor()
        and variant in DETECTOR_MPS_VALIDATED_VARIANTS
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _model_supports_mps(model_config: PoseConfig | DetectorConfig) -> bool:
    """Whether ``device: auto`` may resolve to MPS for this model."""
    if isinstance(model_config, DetectorConfig):
        return torch_meets_detector_mps_floor() and detector_variant(model_config) in DETECTOR_MPS_VALIDATED_VARIANTS
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
    config; a detector left on ``auto`` additionally inherits an explicit
    CPU/CUDA device pinned at the top level of the configuration (an MPS pose
    device is never inherited — the detector then follows its own auto
    policy). ``auto`` resolves per model; for detectors it only selects MPS on
    validated torch versions and detector variants.

    This function only resolves devices; MPS validation (raising below the
    validated torch floor, warning or refusing unvalidated variants) happens
    where a detector runner or trainer is actually built, so resolving a
    configuration whose detector is never instantiated has no side effects.

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
                    "detector_device='cpu' is a no-op for a configuration without a detector.",
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "detector_device was given, but this model configuration "
                    "has no detector (bottom-up models run without one)."
                )
        return ResolvedDevices(pose=resolve_model_device(model_config, device), detector=None)

    pose = resolve_model_device(model_config, device)
    override = detector_device if detector_device is not None else device
    if override is not None:
        detector = resolve_model_device(detector_config, override)
    elif detector_config.device == "auto" and model_config.device != "auto" and not is_mps_device(model_config.device):
        # An explicit CPU/CUDA device pinned at the top level of the config
        # applies to the detector too (matching pre-3.1 behavior); an MPS pose
        # device is not inherited, since the detector may not support it.
        detector = str(model_config.device)
    else:
        detector = resolve_model_device(detector_config)
    return ResolvedDevices(pose=pose, detector=detector)
