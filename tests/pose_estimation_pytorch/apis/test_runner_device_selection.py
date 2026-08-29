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
"""Pins the device each inference-runner builder hands to build_inference_runner.

These tests freeze the CURRENT behavior (including the historical asymmetry of
get_detector_inference_runner) so the detector-MPS capability change can land
as one reviewable commit that updates these expectations explicitly.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml

import deeplabcut.pose_estimation_pytorch.apis.utils as api_utils
from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
from deeplabcut.pose_estimation_pytorch.runners.inference import DetectorInferenceRunner

FIXTURE_PROJECT = Path(__file__).parent.parent / "config" / "fixtures" / "multianimal_project_v0.yaml"


@pytest.fixture
def pose_config(tmp_path) -> PoseConfig:
    with open(FIXTURE_PROJECT) as f:
        project = yaml.safe_load(f)
    project["uniquebodyparts"] = []  # unique bodyparts are unsupported for top-down
    return PoseConfig.build(
        project_config=project,
        pose_config_path=tmp_path / "pose_cfg.yaml",
        top_down=True,
        net_type="resnet_50",
        detector_type="ssdlite",
        save=False,
    )


@pytest.fixture
def captured(monkeypatch):
    """Stub out model construction and capture the device given to each runner."""
    devices: dict = {}

    def fake_build_inference_runner(*, task, device, **kwargs):
        devices[str(task)] = device
        runner = MagicMock(spec=DetectorInferenceRunner)
        runner.device = device
        return runner

    monkeypatch.setattr(api_utils, "build_inference_runner", fake_build_inference_runner)
    monkeypatch.setattr(api_utils.DETECTORS, "build", lambda cfg: MagicMock())
    monkeypatch.setattr(api_utils.PoseModel, "build", lambda cfg: MagicMock())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    return devices


@pytest.mark.parametrize(
    "device, expected_pose, expected_detector",
    [
        ("mps", "mps", "cpu"),  # legacy fallback keeps the detector on the CPU
        ("cpu", "cpu", "cpu"),
        (None, "mps", "cpu"),  # auto: resnet pose -> mps, detector falls back
    ],
)
def test_get_inference_runners_devices(pose_config, captured, device, expected_pose, expected_detector):
    api_utils.get_inference_runners(
        model_config=pose_config,
        snapshot_path="snapshot.pt",
        detector_path="snapshot-detector.pt",
        device=device,
    )
    assert captured["Task.TOP_DOWN"] == expected_pose
    assert captured["Task.DETECT"] == expected_detector


@pytest.mark.parametrize(
    "device, config_device, expected",
    [
        ("mps", "auto", "cpu"),  # explicit mps goes through the legacy fallback
        ("cpu", "auto", "cpu"),
        (None, "auto", "mps"),  # historical asymmetry: config-resolved mps passes
        (None, "mps", "mps"),
    ],
)
def test_get_detector_inference_runner_devices(pose_config, captured, device, config_device, expected):
    pose_config["device"] = config_device
    api_utils.get_detector_inference_runner(
        model_config=pose_config,
        snapshot_path="snapshot-detector.pt",
        device=device,
    )
    assert captured["Task.DETECT"] == expected


@pytest.mark.parametrize(
    "device, config_device, expected",
    [
        ("mps", None, "cpu"),  # unconditional legacy fallback
        ("cpu", None, "cpu"),
        (None, "mps", "cpu"),  # even config-resolved mps is forced to the cpu
    ],
)
def test_filtered_coco_detector_devices(pose_config, captured, monkeypatch, device, config_device, expected):
    detector = MagicMock()
    detector.eval.return_value = detector
    monkeypatch.setitem(
        api_utils.TORCHVISION_DETECTORS,
        "fasterrcnn_mobilenet_v3_large_fpn",
        {"weights": None, "fn": lambda weights, box_score_thresh: detector},
    )
    monkeypatch.setattr(api_utils, "FilteredDetector", lambda *a, **k: MagicMock())
    if config_device is not None:
        pose_config["device"] = config_device
    api_utils.get_filtered_coco_detector_inference_runner(
        model_name="fasterrcnn_mobilenet_v3_large_fpn",
        category_id=1,
        model_config=pose_config,
        device=device,
    )
    assert captured["Task.DETECT"] == expected
