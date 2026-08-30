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
"""Tests for centralized device resolution in pose_estimation_pytorch.utils"""

from __future__ import annotations

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.utils as dlc_utils
from deeplabcut.pose_estimation_pytorch.config.model import DetectorModelConfig
from deeplabcut.pose_estimation_pytorch.config.pose import DetectorConfig
from deeplabcut.pose_estimation_pytorch.utils import (
    resolve_device,
    resolve_model_device,
    resolve_pose_and_detector_devices,
)


class FakePoseConfig:
    """Duck-typed pose config: .device, .get('net_type'), .detector"""

    def __init__(self, device="auto", net_type="resnet_50", detector=None):
        self.device = device
        self.net_type = net_type
        self.detector = detector

    def get(self, key, default=None):
        return getattr(self, key, default)


def make_detector(device: str = "auto") -> DetectorConfig:
    return DetectorConfig(model=DetectorModelConfig(type="SSDLite"), device=device)


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def with_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


@pytest.fixture
def no_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture
def mps_floor_met(monkeypatch):
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: True)


@pytest.fixture
def mps_floor_not_met(monkeypatch):
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: False)


@pytest.fixture
def ssdlite_validated(monkeypatch):
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", frozenset({"ssdlite"}))


@pytest.fixture
def no_validated_variants(monkeypatch):
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", frozenset())


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_explicit_device_returned_verbatim(device, no_cuda, with_mps):
    assert resolve_model_device(FakePoseConfig(device=device)) == device
    assert resolve_model_device(make_detector(device=device)) == device
    assert resolve_model_device(FakePoseConfig(), override=device) == device


def test_auto_pose_prefers_cuda(monkeypatch, with_mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_model_device(FakePoseConfig()) == "cuda"


def test_auto_pose_resnet_uses_mps(no_cuda, with_mps):
    assert resolve_model_device(FakePoseConfig(net_type="resnet_50")) == "mps"
    assert resolve_model_device(FakePoseConfig(net_type="top_down_resnet_50")) == "mps"


def test_auto_pose_hrnet_stays_cpu(no_cuda, with_mps):
    assert resolve_model_device(FakePoseConfig(net_type="hrnet_w32")) == "cpu"


def test_auto_pose_without_accelerator(no_cuda, no_mps):
    assert resolve_model_device(FakePoseConfig()) == "cpu"


def test_auto_detector_requires_validated_variant(no_cuda, with_mps, mps_floor_met, no_validated_variants):
    # not in the validated set -> cpu
    assert resolve_model_device(make_detector()) == "cpu"


def test_ssdlite_is_validated_by_default():
    assert "ssdlite" in dlc_utils.DETECTOR_MPS_VALIDATED_VARIANTS


def test_auto_detector_validated_variant_uses_mps(no_cuda, with_mps, mps_floor_met, ssdlite_validated):
    assert resolve_model_device(make_detector()) == "mps"


def test_auto_detector_below_floor_stays_cpu(no_cuda, with_mps, mps_floor_not_met, ssdlite_validated):
    assert resolve_model_device(make_detector()) == "cpu"


def test_auto_detector_prefers_cuda(monkeypatch, with_mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_model_device(make_detector()) == "cuda"


def test_resolve_device_wrapper_matches(no_cuda, with_mps):
    for cfg in (FakePoseConfig(), make_detector(), FakePoseConfig(device="mps")):
        assert resolve_device(cfg) == resolve_model_device(cfg)


def test_detector_variant_names():
    assert dlc_utils.detector_variant(make_detector()) == "ssdlite"
    frcnn = DetectorConfig(model=DetectorModelConfig(type="FasterRCNN", variant="fasterrcnn_resnet50_fpn_v2"))
    assert dlc_utils.detector_variant(frcnn) == "fasterrcnn_resnet50_fpn_v2"


def test_validate_detector_mps_raises_below_floor(mps_floor_not_met):
    with pytest.raises(RuntimeError, match="detector_device"):
        dlc_utils.validate_detector_mps_request("ssdlite")


def test_validate_detector_mps_warns_for_unvalidated(mps_floor_met):
    with pytest.warns(UserWarning, match="not been validated"):
        dlc_utils.validate_detector_mps_request("fasterrcnn_resnet50_fpn_v2")


def test_validate_detector_mps_silent_for_validated(mps_floor_met, ssdlite_validated, recwarn):
    dlc_utils.validate_detector_mps_request("ssdlite")
    assert len(recwarn) == 0


def test_pair_without_detector(no_cuda, with_mps):
    resolved = resolve_pose_and_detector_devices(FakePoseConfig(device="mps"))
    assert resolved.pose == "mps"
    assert resolved.detector is None


def test_pair_without_detector_rejects_detector_device(no_cuda, with_mps):
    with pytest.raises(ValueError, match="no detector"):
        resolve_pose_and_detector_devices(FakePoseConfig(), detector_device="mps")


def test_pair_without_detector_warns_on_cpu_noop(no_cuda, with_mps):
    with pytest.warns(UserWarning, match="no-op"):
        resolved = resolve_pose_and_detector_devices(FakePoseConfig(), detector_device="cpu")
    assert resolved.detector is None


@pytest.mark.parametrize(
    "device, detector_device, cfg_device, expected_pose, expected_detector",
    [
        # shared explicit device applies to both models; explicit mps is honored
        ("mps", None, "auto", "mps", "mps"),
        ("cpu", None, "auto", "cpu", "cpu"),
        ("cuda:1", None, "auto", "cuda:1", "cuda:1"),
        # no overrides: each model resolves from its own config
        (None, None, "mps", "mps", "cpu"),  # detector config is auto -> cpu (unvalidated)
        (None, None, "auto", "mps", "cpu"),
        # detector-only override wins over the shared device
        ("mps", "cpu", "auto", "mps", "cpu"),
        ("cpu", "mps", "auto", "cpu", "mps"),
        ("mps", "auto", "auto", "mps", "cpu"),
    ],
)
def test_pair_resolution(
    device,
    detector_device,
    cfg_device,
    expected_pose,
    expected_detector,
    no_cuda,
    with_mps,
    mps_floor_met,
    no_validated_variants,
    recwarn,
):
    cfg = FakePoseConfig(device=cfg_device, detector=make_detector())
    resolved = resolve_pose_and_detector_devices(cfg, device=device, detector_device=detector_device)
    assert resolved.pose == expected_pose
    assert resolved.detector == expected_detector


def test_pair_validated_variant_auto_resolves_to_mps(no_cuda, with_mps, mps_floor_met, ssdlite_validated):
    cfg = FakePoseConfig(device="auto", detector=make_detector())
    resolved = resolve_pose_and_detector_devices(cfg)
    assert resolved.pose == "mps"
    assert resolved.detector == "mps"


def test_pair_explicit_mps_resolves_without_side_effects(no_cuda, with_mps, mps_floor_not_met, recwarn):
    # The resolver never validates: raising/warning happens where a detector
    # runner or trainer is actually built.
    cfg = FakePoseConfig(device="auto", detector=make_detector())
    resolved = resolve_pose_and_detector_devices(cfg, detector_device="mps")
    assert resolved.detector == "mps"
    assert len(recwarn) == 0


def test_pair_detector_auto_inherits_explicit_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cfg = FakePoseConfig(device="cuda:1", detector=make_detector())
    resolved = resolve_pose_and_detector_devices(cfg)
    assert resolved.pose == "cuda:1"
    assert resolved.detector == "cuda:1"


def test_pair_detector_auto_inherits_pinned_cpu(monkeypatch, with_mps, mps_floor_met, ssdlite_validated):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cfg = FakePoseConfig(device="cpu", detector=make_detector())
    resolved = resolve_pose_and_detector_devices(cfg)
    assert resolved.pose == "cpu"
    assert resolved.detector == "cpu"


def test_pair_detector_auto_does_not_inherit_mps(no_cuda, with_mps, mps_floor_met, no_validated_variants):
    # An MPS pose device is never inherited: the detector follows its own
    # auto policy (cpu here, since no variant is validated).
    cfg = FakePoseConfig(device="mps", detector=make_detector())
    resolved = resolve_pose_and_detector_devices(cfg)
    assert resolved.pose == "mps"
    assert resolved.detector == "cpu"


def test_pair_explicit_detector_config_wins_over_pin(no_cuda, no_mps):
    cfg = FakePoseConfig(device="cuda:1", detector=make_detector(device="cpu"))
    resolved = resolve_pose_and_detector_devices(cfg)
    assert resolved.detector == "cpu"


def test_validate_detector_mps_training_refuses_unvalidated(mps_floor_met, no_validated_variants):
    with pytest.raises(RuntimeError, match="Refusing to train"):
        dlc_utils.validate_detector_mps_request("fasterrcnn_resnet50_fpn_v2", for_training=True)


def test_validate_detector_mps_training_allows_validated(mps_floor_met, ssdlite_validated, recwarn):
    dlc_utils.validate_detector_mps_request("ssdlite", for_training=True)
    assert len(recwarn) == 0


def test_is_mps_device():
    assert dlc_utils.is_mps_device("mps")
    assert dlc_utils.is_mps_device("mps:0")
    assert dlc_utils.is_mps_device(torch.device("mps"))
    assert not dlc_utils.is_mps_device("cpu")
    assert not dlc_utils.is_mps_device("cuda:0")
    assert not dlc_utils.is_mps_device(None)


@pytest.mark.parametrize(
    "version, expected",
    [
        ("2.12.0", True),
        ("2.12.1", True),
        ("2.12.1+cu121", True),
        ("2.13.0", True),
        ("2.11.2", False),
        ("2.12.0a0", False),  # pre-releases of the floor do not qualify
        ("2.12.0.dev20260401+cpu", False),
        ("not-a-version", False),
    ],
)
def test_torch_floor_version_parsing(monkeypatch, version, expected):
    monkeypatch.setattr(torch, "__version__", version)
    dlc_utils.torch_meets_detector_mps_floor.cache_clear()
    try:
        assert dlc_utils.torch_meets_detector_mps_floor() is expected
    finally:
        dlc_utils.torch_meets_detector_mps_floor.cache_clear()
