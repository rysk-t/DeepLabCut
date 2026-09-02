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
"""Tests for device resolution and the detector MPS hardware gate in pose_estimation_pytorch.utils"""

from __future__ import annotations

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.utils as dlc_utils
from deeplabcut.pose_estimation_pytorch.config.model import DetectorModelConfig
from deeplabcut.pose_estimation_pytorch.config.pose import DetectorConfig
from deeplabcut.pose_estimation_pytorch.utils import (
    detector_mps_supported,
    detector_variant,
    is_mps_device,
    resolve_detector_device,
    resolve_device,
    torch_meets_detector_mps_floor,
)


class FakePoseConfig:
    """Duck-typed pose config: .device, .net_type and .get()"""

    def __init__(self, device: str = "auto", net_type: str = "resnet_50"):
        self.device = device
        self.net_type = net_type

    def get(self, key, default=None):
        return getattr(self, key, default)


def make_detector(device: str = "auto", detector_type: str = "SSDLite", variant: str | None = None) -> DetectorConfig:
    return DetectorConfig(model=DetectorModelConfig(type=detector_type, variant=variant), device=device)


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def with_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def with_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


@pytest.fixture
def no_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture
def mps_floor_met(monkeypatch):
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: True)


@pytest.fixture
def mps_floor_not_met(monkeypatch):
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: False)


DEFAULT_TRAINING_REGISTRY = frozenset(dlc_utils.DETECTOR_MPS_VALIDATED_VARIANTS)
DEFAULT_INFERENCE_REGISTRY = frozenset(dlc_utils.DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS)


@pytest.fixture
def ssdlite_validated(monkeypatch):
    """ssdlite is the only validated variant, for training and inference alike."""
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", frozenset({"ssdlite"}))
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS", frozenset({"ssdlite"}))


@pytest.fixture
def no_validated_variants(monkeypatch):
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", frozenset())
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS", frozenset())


@pytest.fixture
def default_registries(monkeypatch):
    """The registries as shipped (other fixtures narrow them for determinism)."""
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", DEFAULT_TRAINING_REGISTRY)
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS", DEFAULT_INFERENCE_REGISTRY)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps", "mps:0"])
def test_explicit_device_returned_verbatim(device, no_cuda, with_mps, mps_floor_met, no_validated_variants):
    assert resolve_device(FakePoseConfig(device=device)) == device
    assert resolve_device(make_detector(device=device)) == device


def test_auto_pose_prefers_cuda(with_cuda, with_mps):
    assert resolve_device(FakePoseConfig()) == "cuda"


def test_auto_pose_resnet_uses_mps(no_cuda, with_mps):
    assert resolve_device(FakePoseConfig(net_type="resnet_50")) == "mps"
    assert resolve_device(FakePoseConfig(net_type="top_down_resnet_50")) == "mps"


def test_auto_pose_hrnet_stays_cpu(no_cuda, with_mps):
    assert resolve_device(FakePoseConfig(net_type="hrnet_w32")) == "cpu"


def test_auto_pose_without_accelerator(no_cuda, no_mps):
    assert resolve_device(FakePoseConfig()) == "cpu"


def test_auto_detector_validated_variant_uses_mps(no_cuda, with_mps, mps_floor_met, ssdlite_validated):
    assert resolve_device(make_detector()) == "mps"


def test_auto_detector_prefers_cuda(with_cuda, with_mps, mps_floor_met, ssdlite_validated):
    assert resolve_device(make_detector()) == "cuda"


def test_auto_detector_unvalidated_variant_stays_cpu(no_cuda, with_mps, mps_floor_met, no_validated_variants):
    assert resolve_device(make_detector()) == "cpu"


def test_auto_detector_fasterrcnn_stays_cpu(no_cuda, with_mps, mps_floor_met, ssdlite_validated):
    frcnn = make_detector(detector_type="FasterRCNN", variant="fasterrcnn_resnet50_fpn_v2")
    assert resolve_device(frcnn) == "cpu"


def test_auto_detector_below_floor_stays_cpu(no_cuda, with_mps, mps_floor_not_met, ssdlite_validated):
    assert resolve_device(make_detector()) == "cpu"


def test_auto_detector_without_mps_stays_cpu(no_cuda, no_mps, mps_floor_met, ssdlite_validated):
    assert resolve_device(make_detector()) == "cpu"


def test_auto_detector_mps_available_but_not_built_stays_cpu(monkeypatch, no_cuda, mps_floor_met, ssdlite_validated):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device(make_detector()) == "cpu"


def test_ssdlite_is_validated_by_default():
    assert "ssdlite" in dlc_utils.DETECTOR_MPS_VALIDATED_VARIANTS


# ---------------------------------------------------------------------------
# detector_variant
# ---------------------------------------------------------------------------


def test_detector_variant_from_explicit_variant():
    frcnn = make_detector(detector_type="FasterRCNN", variant="fasterrcnn_resnet50_fpn_v2")
    assert detector_variant(frcnn) == "fasterrcnn_resnet50_fpn_v2"


def test_detector_variant_from_ssdlite_type():
    assert detector_variant(make_detector(detector_type="SSDLite")) == "ssdlite"
    assert detector_variant(make_detector(detector_type="ssdlite")) == "ssdlite"


def test_detector_variant_unknown_without_variant():
    assert detector_variant(make_detector(detector_type="FasterRCNN")) is None


def test_detector_variant_from_dict():
    assert detector_variant({"model": {"type": "FasterRCNN", "variant": "fasterrcnn_mobilenet_v3_large_fpn"}}) == (
        "fasterrcnn_mobilenet_v3_large_fpn"
    )
    assert detector_variant({"model": {"type": "SSDLite"}}) == "ssdlite"
    assert detector_variant({"model": {"type": "FasterRCNN"}}) is None


# ---------------------------------------------------------------------------
# is_mps_device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device, expected",
    [
        ("mps", True),
        ("mps:0", True),
        (torch.device("mps"), True),
        ("cpu", False),
        ("cuda:0", False),
        (torch.device("cpu"), False),
        (None, False),
    ],
)
def test_is_mps_device(device, expected):
    assert is_mps_device(device) is expected


# ---------------------------------------------------------------------------
# torch_meets_detector_mps_floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version, expected",
    [
        ("2.12.0", True),
        ("2.12.1", True),
        ("2.13.0", True),
        ("2.13.0+cu128", True),
        ("2.11.9", False),
        ("2.13.0a0+git1234", False),
        ("2.12.0rc1", False),
        ("not-a-version", False),
    ],
)
def test_torch_meets_detector_mps_floor(monkeypatch, version, expected):
    monkeypatch.setattr(torch, "__version__", version)
    assert torch_meets_detector_mps_floor() is expected


# ---------------------------------------------------------------------------
# detector_mps_supported
# ---------------------------------------------------------------------------


def test_detector_mps_supported_all_conditions(with_mps, mps_floor_met, ssdlite_validated):
    assert detector_mps_supported("ssdlite") is True


def test_detector_mps_supported_without_mps(no_mps, mps_floor_met, ssdlite_validated):
    assert detector_mps_supported("ssdlite") is False


def test_detector_mps_supported_available_but_not_built(monkeypatch, mps_floor_met, ssdlite_validated):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert detector_mps_supported("ssdlite") is False


def test_detector_mps_supported_built_but_not_available(monkeypatch, mps_floor_met, ssdlite_validated):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert detector_mps_supported("ssdlite") is False


def test_detector_mps_supported_below_floor(with_mps, mps_floor_not_met, ssdlite_validated):
    assert detector_mps_supported("ssdlite") is False


@pytest.mark.parametrize("variant", ["fasterrcnn_resnet50_fpn_v2", "fasterrcnn_mobilenet_v3_large_fpn", None])
def test_detector_mps_supported_unvalidated_variant(variant, with_mps, mps_floor_met, ssdlite_validated):
    assert detector_mps_supported(variant) is False


def test_detector_mps_supported_empty_validated_set(with_mps, mps_floor_met, no_validated_variants):
    assert detector_mps_supported("ssdlite") is False


# ---------------------------------------------------------------------------
# resolve_detector_device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda:1", None])
def test_resolve_detector_device_non_mps_passthrough(device, no_mps, mps_floor_not_met, no_validated_variants, recwarn):
    assert resolve_detector_device(device, None, for_training=True) == device
    assert resolve_detector_device(device, None, for_training=False) == device
    assert len(recwarn) == 0


@pytest.mark.parametrize("device", ["mps", "mps:0"])
def test_resolve_detector_device_validated_unchanged(device, with_mps, mps_floor_met, ssdlite_validated, recwarn):
    assert resolve_detector_device(device, "ssdlite", for_training=True) == device
    assert resolve_detector_device(device, "ssdlite", for_training=False) == device
    assert len(recwarn) == 0


def test_resolve_detector_device_torch_device_validated_unchanged(with_mps, mps_floor_met, ssdlite_validated, recwarn):
    device = torch.device("mps")
    assert resolve_detector_device(device, "ssdlite", for_training=False) is device
    assert len(recwarn) == 0


def test_resolve_detector_device_mps_unavailable_falls_back(no_mps, mps_floor_met, ssdlite_validated):
    with pytest.warns(UserWarning, match="not available") as record:
        assert resolve_detector_device("mps", "ssdlite", for_training=False) == "cpu"
    assert "detector.device: cpu" in str(record[0].message)


def test_resolve_detector_device_below_floor_falls_back(with_mps, mps_floor_not_met, ssdlite_validated):
    with pytest.warns(UserWarning, match="2.12") as record:
        assert resolve_detector_device("mps", "ssdlite", for_training=False) == "cpu"
    assert "detector.device: cpu" in str(record[0].message)


def test_resolve_detector_device_unvalidated_falls_back(with_mps, mps_floor_met, ssdlite_validated):
    with pytest.warns(UserWarning, match="fasterrcnn_resnet50_fpn_v2") as record:
        assert resolve_detector_device("mps", "fasterrcnn_resnet50_fpn_v2", for_training=False) == "cpu"
    message = str(record[0].message)
    assert "not been validated" in message
    assert "detector.device: cpu" in message


def test_resolve_detector_device_unknown_variant_falls_back(with_mps, mps_floor_met, ssdlite_validated):
    with pytest.warns(UserWarning, match="None"):
        assert resolve_detector_device("mps", None, for_training=False) == "cpu"


def test_resolve_detector_device_training_unvalidated_mentions_watchdog(with_mps, mps_floor_met, ssdlite_validated):
    with pytest.warns(UserWarning) as record:
        assert resolve_detector_device("mps", "fasterrcnn_resnet50_fpn_v2", for_training=True) == "cpu"
    message = str(record[0].message)
    assert "fasterrcnn_resnet50_fpn_v2" in message
    assert "watchdog" in message
    assert "reboot" in message


def test_resolve_detector_device_inference_unvalidated_no_watchdog(with_mps, mps_floor_met, ssdlite_validated):
    with pytest.warns(UserWarning) as record:
        assert resolve_detector_device("mps", "fasterrcnn_resnet50_fpn_v2", for_training=False) == "cpu"
    message = str(record[0].message)
    assert "watchdog" not in message
    assert "reboot" not in message


# ---------------------------------------------------------------------------
# training vs inference registries
# ---------------------------------------------------------------------------

FRCNN_V1 = "fasterrcnn_resnet50_fpn"
FRCNN_V2 = "fasterrcnn_resnet50_fpn_v2"
FRCNN_MOBILENET = "fasterrcnn_mobilenet_v3_large_fpn"


def test_default_registries():
    assert DEFAULT_TRAINING_REGISTRY == {"ssdlite"}
    assert DEFAULT_INFERENCE_REGISTRY == {"ssdlite", FRCNN_V1, FRCNN_V2}
    assert DEFAULT_TRAINING_REGISTRY <= DEFAULT_INFERENCE_REGISTRY
    assert FRCNN_MOBILENET not in DEFAULT_INFERENCE_REGISTRY


@pytest.mark.parametrize("variant", [FRCNN_V1, FRCNN_V2])
def test_detector_mps_supported_inference_only_variants(variant, with_mps, mps_floor_met, default_registries):
    assert detector_mps_supported(variant, for_training=False) is True
    assert detector_mps_supported(variant, for_training=True) is False
    assert detector_mps_supported(variant) is False  # training is the conservative default


def test_detector_mps_supported_mobilenet_neither(with_mps, mps_floor_met, default_registries):
    assert detector_mps_supported(FRCNN_MOBILENET, for_training=False) is False
    assert detector_mps_supported(FRCNN_MOBILENET, for_training=True) is False


def test_resolve_device_detector_auto_depends_on_task(no_cuda, with_mps, mps_floor_met, default_registries):
    config = DetectorConfig(model=DetectorModelConfig(type="FasterRCNN", variant=FRCNN_V2), device="auto")
    assert resolve_device(config) == "cpu"  # training
    assert resolve_device(config, for_training=False) == "mps"  # inference
    ssdlite = DetectorConfig(model=DetectorModelConfig(type="SSDLite"), device="auto")
    assert resolve_device(ssdlite) == "mps"
    assert resolve_device(ssdlite, for_training=False) == "mps"


def test_resolve_detector_device_inference_only_variant(with_mps, mps_floor_met, default_registries, recwarn):
    assert dlc_utils.resolve_detector_device("mps", FRCNN_V2, for_training=False) == "mps"
    assert dlc_utils.resolve_detector_device("mps:0", FRCNN_V1, for_training=False) == "mps:0"
    assert recwarn.list == []
    assert dlc_utils.resolve_detector_device("mps", FRCNN_V2, for_training=True) == "cpu"
    (warning,) = recwarn.list
    message = str(warning.message)
    assert FRCNN_V2 in message and "for training" in message and "watchdog" in message


def test_resolve_detector_device_inference_unvalidated_names_the_task(with_mps, mps_floor_met, default_registries):
    with pytest.warns(UserWarning, match="for inference"):
        assert dlc_utils.resolve_detector_device("mps", FRCNN_MOBILENET, for_training=False) == "cpu"
