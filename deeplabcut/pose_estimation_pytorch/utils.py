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


MIN_TORCH_FOR_DETECTOR_MPS = (2, 12)
"""Minimum torch version for running object detectors on Apple MPS.

Older versions are known to hang or return corrupted predictions (see
DeepLabCut#3155 and DeepLabCut#2853).
"""

DETECTOR_MPS_VALIDATED_VARIANTS: frozenset[str] = frozenset({"ssdlite"})
"""Detector variants validated for training (and inference) on Apple MPS.

``device: auto`` only resolves to MPS for these variants when a detector is
trained, and explicit MPS requests for any other variant fall back to the CPU
(see :func:`resolve_detector_device`). Inference uses the wider
:data:`DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS`.

ssdlite: validated on Apple Silicon (M5) with torch 2.12.1 / torchvision 0.27.1
and torch 2.13.0 / torchvision 0.28.0. Training losses were within 1.3% of the
CPU run per epoch; inference bounding boxes and keypoints matched the CPU run
within 1e-4 px (p95) on a 300-frame clip, and a 9000-frame video ran to
completion on MPS with no NaN or all-zero outputs. Training was ~7-12x and
end-to-end inference ~2-3x faster than on the CPU.

Faster R-CNN variants are deliberately NOT listed: training
fasterrcnn_resnet50_fpn on MPS hung the GPU on the first iteration, and
fasterrcnn_resnet50_fpn_v2 hung it hard enough to trigger a macOS watchdog
kernel panic. The root cause is the MPS
backward kernel of torchvision's ``roi_align``, which over-accumulates the
input gradient in proportion to the number of RoIs (the forward pass is
unaffected, which is why inference looks fine). It is fixed by
pytorch/vision#9510 (merged 2026-07-15), released in torchvision 0.29.0 (which
requires torch 2.14). Re-validate these variants on MPS against that release
before adding them here.
"""

DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS: frozenset[str] = frozenset(
    {"ssdlite", "fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2"}
)
"""Detector variants validated for inference on Apple MPS.

A superset of :data:`DETECTOR_MPS_VALIDATED_VARIANTS`: inference only runs the
forward pass, which the ``roi_align`` backward bug does not affect, so the
Faster R-CNN variants whose predictions matched the CPU can run on MPS for
inference while their training stays on the CPU.

fasterrcnn_resnet50_fpn and fasterrcnn_resnet50_fpn_v2: inference with
snapshots trained on real data matched the CPU run on Apple Silicon (M5) with
torch 2.12.1 / torchvision 0.27.1 and torch 2.13.0 / torchvision 0.28.0 (600/600
detections matched over a 300-frame clip, bounding boxes within 1e-4 px (p95),
scores within 1e-6), 2-5x faster than on the CPU.
fasterrcnn_mobilenet_v3_large_fpn is not listed: the only comparison so far
produced no detections on either device.
"""


def torch_meets_detector_mps_floor() -> bool:
    """Whether the installed torch version supports running detectors on Apple MPS.

    Only released builds qualify: pre-releases and nightlies (e.g. ``2.13.0a0+git1234``)
    of the floor version were not validated. Unparseable version strings are treated
    as unsupported.

    Returns:
        True if ``torch.__version__`` is a release >= ``MIN_TORCH_FOR_DETECTOR_MPS``.
    """
    floor = Version(".".join(str(v) for v in MIN_TORCH_FOR_DETECTOR_MPS))
    try:
        installed = Version(torch.__version__)
    except InvalidVersion:
        return False
    return not installed.is_prerelease and installed >= floor


def is_mps_device(device: str | torch.device | None) -> bool:
    """Whether a device specification targets Apple MPS (``"mps"``, ``"mps:0"``, ...).

    Args:
        device: The device specification to check.

    Returns:
        True if the device targets MPS.
    """
    return device is not None and str(device).startswith("mps")


def detector_variant(config: DetectorConfig | dict) -> str | None:
    """Returns the canonical variant name of a detector configuration, if known.

    Args:
        config: The detector configuration (``DetectorConfig`` or its dict form).

    Returns:
        ``model.variant`` when set, ``"ssdlite"`` for the SSDLite detector type,
        and None when the variant cannot be determined.
    """
    model = config["model"] if isinstance(config, dict) else config.model
    variant = model.get("variant")
    if variant:
        return str(variant)
    if str(model.get("type", "")).lower() == "ssdlite":
        return "ssdlite"
    return None


def detector_mps_supported(variant: str | None, *, for_training: bool = True) -> bool:
    """Whether a detector variant may run on Apple MPS on this machine.

    All of the following must hold: MPS is built into torch and available, the
    installed torch is a release >= ``MIN_TORCH_FOR_DETECTOR_MPS``, and the
    variant is in ``DETECTOR_MPS_VALIDATED_VARIANTS`` (training) or
    ``DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS`` (inference). Unknown variants
    (None) are treated as unvalidated.

    Args:
        variant: The canonical detector variant name, if known.
        for_training: Whether the detector is about to be trained; the training
            registry is stricter than the inference one.

    Returns:
        True if the detector can run on MPS.
    """
    registry = DETECTOR_MPS_VALIDATED_VARIANTS if for_training else DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS
    return (
        torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
        and torch_meets_detector_mps_floor()
        and variant in registry
    )


def resolve_detector_device(
    device: str | torch.device | None,
    variant: str | None,
    *,
    for_training: bool,
) -> str | torch.device | None:
    """Applies the MPS hardware gate to the device a detector is about to run on.

    Non-MPS devices are returned unchanged. An MPS device is returned unchanged when
    :func:`detector_mps_supported` holds for the variant; otherwise a warning naming
    the reason is emitted and the detector falls back to ``"cpu"``.

    Args:
        device: The requested detector device.
        variant: The canonical detector variant name, if known.
        for_training: Whether the detector is about to be trained. Training
            unvalidated variants on MPS has been observed to hang the GPU badly
            enough to trigger a system watchdog reboot, which the warning states.

    Returns:
        The device the detector should run on.
    """
    if device is None or not is_mps_device(device):
        return device
    if detector_mps_supported(variant, for_training=for_training):
        return device

    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        reason = (
            f"Detector device {str(device)!r} was requested, but MPS is not available on this machine "
            f"(torch.backends.mps.is_built()={torch.backends.mps.is_built()}, "
            f"is_available()={torch.backends.mps.is_available()})."
        )
    elif not torch_meets_detector_mps_floor():
        floor = ".".join(str(v) for v in MIN_TORCH_FOR_DETECTOR_MPS)
        reason = (
            f"Running detectors on MPS requires torch >= {floor} (found {torch.__version__}); "
            "older versions are known to hang or produce corrupted predictions."
        )
    else:
        task = "training" if for_training else "inference"
        reason = f"Detector variant {variant!r} has not been validated on MPS for {task}."
        if for_training:
            reason += (
                " Training unvalidated detectors on MPS has been observed to hang the GPU badly "
                "enough to trigger a system watchdog reboot (fasterrcnn_resnet50_fpn_v2 on Apple Silicon). "
                "This is a torchvision roi_align backward bug on MPS (pytorch/vision#9510), fixed in "
                "torchvision 0.29.0; training on MPS has not been validated against that release yet."
            )

    warnings.warn(
        f"{reason} The detector runs on the CPU instead. To silence this warning, run the detector "
        'on the CPU explicitly: pass device="cpu", or set detector.device: cpu in the model '
        "configuration (honoured by train_network, analyze_videos and evaluate_network when no "
        "device argument is given).",
        UserWarning,
        stacklevel=2,
    )
    return "cpu"


def resolve_device(model_config: PoseConfig | DetectorConfig, *, for_training: bool = True) -> str:
    """Determines which device should be used from the model config.

    When the device is set to 'auto':
        If an Nvidia GPU is available, selects the device as cuda:0.
        Selects 'mps' if available (on macOS) and the model supports it: resnet pose
        backbones, and detector variants validated on MPS on torch >= 2.12.
        Otherwise, returns 'cpu'.
    Otherwise, simply returns the selected device

    Args:
        model_config (PoseConfig | dict | str | Path): The PyTorch pose configuration.
        for_training: For detector configurations, whether the detector is about to be
            trained (the training registry of MPS-validated variants is stricter than
            the inference one). Ignored for pose configurations.

    Returns:
        the device on which training should be run
    """
    device = model_config.device

    if isinstance(model_config, DetectorConfig):
        supports_mps = detector_mps_supported(detector_variant(model_config), for_training=for_training)
    else:
        supports_mps = "resnet" in model_config.get("net_type", "")

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif supports_mps and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
