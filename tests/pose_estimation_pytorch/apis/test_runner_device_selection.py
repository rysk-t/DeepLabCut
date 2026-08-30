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

The detector device contract: explicit detector_device > shared device >
DetectorConfig.device; ``auto`` only selects MPS for validated torch versions
and detector variants; explicit MPS is honored (raising below the torch floor,
warning for unvalidated variants) instead of silently falling back to the CPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml

import deeplabcut.pose_estimation_pytorch.apis.utils as api_utils
import deeplabcut.pose_estimation_pytorch.utils as dlc_utils
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
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: True)
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", frozenset())
    return devices


@pytest.mark.parametrize(
    "device, expected_pose, expected_detector",
    [
        ("mps", "mps", "mps"),  # explicit shared device applies to the detector too
        ("cpu", "cpu", "cpu"),
        (None, "mps", "cpu"),  # auto: resnet pose -> mps; unvalidated detector -> cpu
    ],
)
def test_get_inference_runners_devices(pose_config, captured, recwarn, device, expected_pose, expected_detector):
    api_utils.get_inference_runners(
        model_config=pose_config,
        snapshot_path="snapshot.pt",
        detector_path="snapshot-detector.pt",
        device=device,
    )
    assert captured["Task.TOP_DOWN"] == expected_pose
    assert captured["Task.DETECT"] == expected_detector


@pytest.mark.parametrize(
    "device, detector_device, expected_pose, expected_detector",
    [
        ("mps", "cpu", "mps", "cpu"),
        ("cpu", "cuda:1", "cpu", "cuda:1"),
        ("mps", "auto", "mps", "cpu"),  # auto resolves via the detector config
        ("cpu", "mps", "cpu", "mps"),  # explicit mps is honored (with a warning)
    ],
)
def test_get_inference_runners_detector_device(
    pose_config, captured, recwarn, device, detector_device, expected_pose, expected_detector
):
    api_utils.get_inference_runners(
        model_config=pose_config,
        snapshot_path="snapshot.pt",
        detector_path="snapshot-detector.pt",
        device=device,
        detector_device=detector_device,
    )
    assert captured["Task.TOP_DOWN"] == expected_pose
    assert captured["Task.DETECT"] == expected_detector


@pytest.mark.parametrize(
    "device, pose_cfg_device, detector_cfg_device, expected",
    [
        ("mps", "auto", "auto", "mps"),  # explicit mps honored (with a warning)
        ("cpu", "auto", "auto", "cpu"),
        (None, "mps", "auto", "cpu"),  # detector reads its own config, not the pose's
        (None, "auto", "mps", "mps"),  # explicit-by-config mps honored
    ],
)
def test_get_detector_inference_runner_devices(
    pose_config, captured, recwarn, device, pose_cfg_device, detector_cfg_device, expected
):
    pose_config["device"] = pose_cfg_device
    pose_config["detector"]["device"] = detector_cfg_device
    api_utils.get_detector_inference_runner(
        model_config=pose_config,
        snapshot_path="snapshot-detector.pt",
        device=device,
    )
    assert captured["Task.DETECT"] == expected


def test_get_detector_inference_runner_raises_below_floor(pose_config, captured, monkeypatch):
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: False)
    with pytest.raises(RuntimeError, match="torch >="):
        api_utils.get_detector_inference_runner(
            model_config=pose_config,
            snapshot_path="snapshot-detector.pt",
            device="mps",
        )


@pytest.mark.parametrize(
    "device, expected",
    [
        ("mps", "mps"),  # explicit mps honored (with a warning)
        ("cpu", "cpu"),
        (None, "cpu"),  # auto policy by torchvision model name; unvalidated -> cpu
    ],
)
def test_filtered_coco_detector_devices(pose_config, captured, recwarn, monkeypatch, device, expected):
    detector = MagicMock()
    detector.eval.return_value = detector
    monkeypatch.setitem(
        api_utils.TORCHVISION_DETECTORS,
        "fasterrcnn_mobilenet_v3_large_fpn",
        {"weights": None, "fn": lambda weights, box_score_thresh: detector},
    )
    monkeypatch.setattr(api_utils, "FilteredDetector", lambda *a, **k: MagicMock())
    api_utils.get_filtered_coco_detector_inference_runner(
        model_name="fasterrcnn_mobilenet_v3_large_fpn",
        category_id=1,
        model_config=pose_config,
        device=device,
    )
    assert captured["Task.DETECT"] == expected


def test_runner_devices_are_logged(pose_config, captured, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        api_utils.get_inference_runners(
            model_config=pose_config,
            snapshot_path="snapshot.pt",
            detector_path="snapshot-detector.pt",
            device="cpu",
        )
    assert "Pose inference runner device: cpu" in caplog.text
    assert "Detector inference runner device: cpu" in caplog.text


def test_get_inference_runners_auto_validated_variant(pose_config, captured, monkeypatch):
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", frozenset({"ssdlite"}))
    api_utils.get_inference_runners(
        model_config=pose_config,
        snapshot_path="snapshot.pt",
        detector_path="snapshot-detector.pt",
        device=None,
    )
    assert captured["Task.TOP_DOWN"] == "mps"
    assert captured["Task.DETECT"] == "mps"


def test_get_inference_runners_no_validation_without_detector_path(pose_config, captured, recwarn):
    # No detector runner is built, so no MPS validation fires for it.
    api_utils.get_inference_runners(
        model_config=pose_config,
        snapshot_path="snapshot.pt",
        detector_path=None,
        device="mps",
    )
    assert captured["Task.TOP_DOWN"] == "mps"
    assert len(recwarn) == 0


def test_detector_indexed_mps_raises_below_floor(pose_config, captured, monkeypatch):
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: False)
    with pytest.raises(RuntimeError, match="torch >="):
        api_utils.get_detector_inference_runner(
            model_config=pose_config,
            snapshot_path="snapshot-detector.pt",
            device="mps:0",  # indexed spelling must not bypass validation
        )


def test_filtered_coco_detector_honors_config_pin(pose_config, captured, monkeypatch, recwarn):
    detector = MagicMock()
    detector.eval.return_value = detector
    monkeypatch.setitem(
        api_utils.TORCHVISION_DETECTORS,
        "fasterrcnn_mobilenet_v3_large_fpn",
        {"weights": None, "fn": lambda weights, box_score_thresh: detector},
    )
    monkeypatch.setattr(api_utils, "FilteredDetector", lambda *a, **k: MagicMock())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pose_config["device"] = "cpu"
    api_utils.get_filtered_coco_detector_inference_runner(
        model_name="fasterrcnn_mobilenet_v3_large_fpn",
        category_id=1,
        model_config=pose_config,
        device=None,
    )
    assert captured["Task.DETECT"] == "cpu"


def test_torchvision_name_maps_to_validated_variant():
    assert api_utils._TORCHVISION_MODEL_VARIANTS["ssdlite320_mobilenet_v3_large"] == "ssdlite"


def test_detector_runner_resolves_explicit_auto(pose_config, captured, recwarn):
    api_utils.get_detector_inference_runner(
        model_config=pose_config,
        snapshot_path="snapshot-detector.pt",
        device="auto",  # must be resolved, never passed to torch verbatim
    )
    assert captured["Task.DETECT"] == "cpu"  # unvalidated variant -> cpu


def test_filtered_coco_detector_resolves_explicit_auto(pose_config, captured, monkeypatch, recwarn):
    detector = MagicMock()
    detector.eval.return_value = detector
    monkeypatch.setitem(
        api_utils.TORCHVISION_DETECTORS,
        "fasterrcnn_mobilenet_v3_large_fpn",
        {"weights": None, "fn": lambda weights, box_score_thresh: detector},
    )
    monkeypatch.setattr(api_utils, "FilteredDetector", lambda *a, **k: MagicMock())
    api_utils.get_filtered_coco_detector_inference_runner(
        model_name="fasterrcnn_mobilenet_v3_large_fpn",
        category_id=1,
        model_config=pose_config,
        device="auto",
    )
    assert captured["Task.DETECT"] == "cpu"


def test_detector_device_is_the_last_parameter():
    """detector_device was added to public signatures; keeping it last preserves
    positional-argument compatibility for pre-existing callers."""
    import importlib
    import inspect

    import deeplabcut.pose_estimation_pytorch.apis.evaluation as eval_mod
    import deeplabcut.pose_estimation_pytorch.apis.training as train_mod
    import deeplabcut.pose_estimation_pytorch.apis.videos as videos_mod
    from deeplabcut import compat

    # The apis package re-exports the analyze_images FUNCTION under the same
    # name as its module, so the module must be fetched via importlib.
    ai_mod = importlib.import_module("deeplabcut.pose_estimation_pytorch.apis.analyze_images")

    for fn in (
        compat.train_network,
        train_mod.train_network,
        videos_mod.analyze_videos,
        eval_mod.evaluate_network,
        eval_mod.evaluate_snapshot,
        ai_mod.analyze_images,
        ai_mod.analyze_image_folder,
        api_utils.get_inference_runners,
    ):
        assert list(inspect.signature(fn).parameters)[-1] == "detector_device", fn.__qualname__
