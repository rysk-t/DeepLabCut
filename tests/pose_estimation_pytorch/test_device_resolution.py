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


def test_auto_detector_never_mps_under_legacy_policy(no_cuda, with_mps):
    assert resolve_model_device(make_detector()) == "cpu"


def test_auto_detector_prefers_cuda(monkeypatch, with_mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_model_device(make_detector()) == "cuda"


def test_resolve_device_wrapper_matches(no_cuda, with_mps):
    for cfg in (FakePoseConfig(), make_detector(), FakePoseConfig(device="mps")):
        assert resolve_device(cfg) == resolve_model_device(cfg)


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
    "device, cfg_device, expected_pose, expected_detector",
    [
        ("mps", "auto", "mps", "cpu"),  # legacy fallback
        ("cpu", "auto", "cpu", "cpu"),
        (None, "mps", "mps", "cpu"),  # config-driven mps also falls back
        (None, "auto", "mps", "cpu"),  # auto resolves pose to mps, detector cpu
        ("cuda:1", "auto", "cuda:1", "cuda:1"),
    ],
)
def test_pair_legacy_policy(device, cfg_device, expected_pose, expected_detector, no_cuda, with_mps):
    cfg = FakePoseConfig(device=cfg_device, detector=make_detector())
    resolved = resolve_pose_and_detector_devices(cfg, device=device)
    assert resolved.pose == expected_pose
    assert resolved.detector == expected_detector


def test_pair_explicit_detector_device_wins(no_cuda, with_mps):
    cfg = FakePoseConfig(device="mps", detector=make_detector())
    resolved = resolve_pose_and_detector_devices(cfg, device="mps", detector_device="mps")
    assert resolved.pose == "mps"
    assert resolved.detector == "mps"

    resolved = resolve_pose_and_detector_devices(cfg, device="cpu", detector_device="mps")
    assert resolved.pose == "cpu"
    assert resolved.detector == "mps"

    # detector_device="auto" resolves via the detector config (legacy: cpu)
    resolved = resolve_pose_and_detector_devices(cfg, device="mps", detector_device="auto")
    assert resolved.detector == "cpu"
