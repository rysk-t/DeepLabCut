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
"""Tests for the training API."""

import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.utils as dlc_utils
from deeplabcut.pose_estimation_pytorch.apis.training import train, train_network
from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.config.enums import DetectorType
from deeplabcut.pose_estimation_pytorch.config.pose import DetectorConfig, PoseConfig
from deeplabcut.pose_estimation_pytorch.task import Task


def _project_cfg(tmp_path: Path) -> dict:
    return {
        "multianimalproject": False,
        "project_path": str(tmp_path),
        "bodyparts": ["nose"],
        "uniquebodyparts": [],
        "individuals": ["mouse"],
    }


def _minimal_run_config(tmp_path: Path, *, resume_from: str | None = None) -> PoseConfig:
    cfg_path = tmp_path / "pytorch_config.yaml"
    pose_config = make_pytorch_pose_config(_project_cfg(tmp_path), str(cfg_path), net_type="resnet_50")
    if resume_from is not None:
        pose_config.resume_training_from = resume_from
    return pose_config


def _top_down_config(tmp_path: Path) -> PoseConfig:
    """A top-down ResNet pose config; its detector is an SSDLite (a variant validated on MPS)."""
    cfg_path = tmp_path / "pytorch_config.yaml"
    return make_pytorch_pose_config(_project_cfg(tmp_path), str(cfg_path), net_type="resnet_50", top_down=True)


def _make_loader(tmp_path: Path, run_config: PoseConfig | DetectorConfig) -> Mock:
    loader = Mock()
    loader.model_folder = tmp_path
    loader.model_cfg = run_config
    train_dataset = Mock(__len__=Mock(return_value=1))
    valid_dataset = Mock(__len__=Mock(return_value=1))
    loader.create_dataset = Mock(side_effect=[train_dataset, valid_dataset])
    return loader


@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_transforms", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.PoseModel.build", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_training_runner", return_value=Mock())
def test_train_uses_resume_training_from_config(
    mock_build_runner: Mock,
    mock_build_model: Mock,
    mock_build_transforms: Mock,
    tmp_path: Path,
) -> None:
    run_config = _minimal_run_config(tmp_path, resume_from="/train/snapshot-010.pt")
    loader = _make_loader(tmp_path, run_config)

    train(loader=loader, run_config=run_config, task=Task.BOTTOM_UP, device="cpu", snapshot_path=None)

    assert mock_build_runner.call_args.kwargs["snapshot_path"] == "/train/snapshot-010.pt"


# ---------------------------------------------------------------------------
# train(): detector MPS hardware gate
# ---------------------------------------------------------------------------

# Phrase every fallback warning of the gate ends with; used to tell them apart from
# unrelated warnings emitted while building the (mocked) training pipeline.
_GATE_WARNING_MARKER = "runs on the CPU instead"


def _set_mps_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mps: bool = True,
    floor: bool = True,
    validated: frozenset[str] = frozenset({"ssdlite"}),
) -> None:
    """Fakes the conditions of the detector MPS gate; no model is ever moved to a real device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: mps)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: floor)
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", validated)


def _run_config(kind: str, tmp_path: Path) -> PoseConfig | DetectorConfig:
    pose_config = _top_down_config(tmp_path)
    if kind == "pose":
        return pose_config
    if kind == "ssdlite":
        return pose_config["detector"]
    if kind == "fasterrcnn":
        # the default Faster R-CNN config: variant "fasterrcnn_resnet50_fpn_v2", not validated
        return DetectorConfig.build(1, DetectorType.FASTERRCNN_RESNET50_FPN_V2)
    if kind == "fasterrcnn-no-variant":
        detector_config = DetectorConfig.build(1, DetectorType.FASTERRCNN_RESNET50_FPN_V2)
        detector_config["model"]["variant"] = None  # unknown variant: treated as unvalidated
        return detector_config
    raise ValueError(f"Unknown run config kind: {kind}")


def _gate_warnings(recwarn: pytest.WarningsRecorder) -> list[warnings.WarningMessage]:
    return [w for w in recwarn if _GATE_WARNING_MARKER in str(w.message)]


_FRCNN_VARIANT = "fasterrcnn_resnet50_fpn_v2"
_FRCNN_TRAINING_WARNING = [_FRCNN_VARIANT, "not been validated", "watchdog", "reboot"]


@pytest.mark.parametrize(
    "kind, config_device, requested, environment, expected, expected_in_warning",
    [
        # explicit device argument (config device irrelevant)
        pytest.param("ssdlite", "auto", "mps", {}, "mps", None, id="validated-ssdlite-mps"),
        pytest.param(
            "ssdlite",
            "auto",
            "mps",
            {"validated": frozenset()},
            "cpu",
            ["ssdlite", "watchdog", "reboot"],
            id="unvalidated-mps",
        ),
        pytest.param(
            "ssdlite",
            "auto",
            "mps:0",
            {"validated": frozenset()},
            "cpu",
            ["ssdlite", "watchdog", "reboot"],
            id="unvalidated-mps0",
        ),
        pytest.param("ssdlite", "auto", "mps", {"floor": False}, "cpu", ["2.12"], id="torch-below-floor"),
        pytest.param("ssdlite", "auto", "mps", {"mps": False}, "cpu", ["not available"], id="mps-not-available"),
        pytest.param("fasterrcnn", "auto", "mps", {}, "cpu", _FRCNN_TRAINING_WARNING, id="fasterrcnn-mps"),
        pytest.param("fasterrcnn", "auto", "mps:0", {}, "cpu", _FRCNN_TRAINING_WARNING, id="fasterrcnn-mps0"),
        pytest.param(
            "fasterrcnn",
            "auto",
            torch.device("mps"),
            {},
            "cpu",
            _FRCNN_TRAINING_WARNING,
            id="fasterrcnn-torch-device-mps",
        ),
        pytest.param(
            "fasterrcnn-no-variant",
            "auto",
            "mps",
            {},
            "cpu",
            ["None", "not been validated", "watchdog"],
            id="fasterrcnn-without-variant",
        ),
        pytest.param("ssdlite", "auto", "cpu", {}, "cpu", None, id="explicit-cpu"),
        pytest.param("pose", "auto", "mps", {}, "mps", None, id="pose-model-untouched"),
        # device=None: the config device is used, which is how train_network() calls train()
        # once the detector has inherited the top-level device
        pytest.param("ssdlite", "mps", None, {}, "mps", None, id="config-mps-ssdlite-device-none"),
        pytest.param("ssdlite", "auto", None, {}, "mps", None, id="config-auto-ssdlite-device-none"),
        pytest.param(
            "fasterrcnn", "mps", None, {}, "cpu", _FRCNN_TRAINING_WARNING, id="config-mps-fasterrcnn-device-none"
        ),
        pytest.param(
            "fasterrcnn", "mps:0", None, {}, "cpu", _FRCNN_TRAINING_WARNING, id="config-mps0-fasterrcnn-device-none"
        ),
        # "auto" never resolves to MPS for an unvalidated variant, so there is nothing to warn about
        pytest.param("fasterrcnn", "auto", None, {}, "cpu", None, id="config-auto-fasterrcnn-device-none"),
        pytest.param("fasterrcnn-no-variant", "auto", None, {}, "cpu", None, id="config-auto-no-variant-device-none"),
        # device="auto" re-resolves from the config, overriding an explicit config device
        pytest.param("ssdlite", "cpu", "auto", {}, "mps", None, id="api-auto-ssdlite"),
        pytest.param("fasterrcnn", "mps", "auto", {}, "cpu", None, id="api-auto-fasterrcnn"),
    ],
)
@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_transforms", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.DETECTORS.build", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.PoseModel.build", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_training_runner", return_value=Mock())
def test_train_applies_detector_mps_gate(
    mock_build_runner: Mock,
    mock_build_pose_model: Mock,
    mock_build_detector: Mock,
    mock_build_transforms: Mock,
    kind: str,
    config_device: str,
    requested: str | torch.device | None,
    environment: dict,
    expected: str,
    expected_in_warning: list[str] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Detectors only train on MPS when the hardware gate holds; pose models are untouched.

    Covers the explicit device argument as well as the device=None / device="auto" paths,
    which resolve the device from the run configuration.
    """
    _set_mps_environment(monkeypatch, **environment)
    run_config = _run_config(kind, tmp_path)
    run_config["device"] = config_device
    task = Task.TOP_DOWN if kind == "pose" else Task.DETECT

    train(loader=_make_loader(tmp_path, run_config), run_config=run_config, task=task, device=requested)

    model = mock_build_pose_model.return_value if kind == "pose" else mock_build_detector.return_value
    assert model.to.call_args.args == (expected,)
    assert mock_build_runner.call_args.kwargs["device"] == expected

    gate_warnings = _gate_warnings(recwarn)
    if expected_in_warning is None:
        assert gate_warnings == []
    else:
        assert len(gate_warnings) == 1
        assert gate_warnings[0].category is UserWarning
        message = str(gate_warnings[0].message)
        for fragment in expected_in_warning:
            assert fragment in message, (fragment, message)


# ---------------------------------------------------------------------------
# train_network(): detector device inheritance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "detector_device, top_level_device, explicit_device, expected_detector_device, expected_detector_arg",
    [
        pytest.param("cpu", "auto", None, "cpu", None, id="explicit-detector-cpu-kept"),
        pytest.param("auto", "cuda:1", None, "cuda:1", None, id="detector-auto-inherits-top-level"),
        pytest.param("auto", "auto", None, "auto", None, id="both-auto"),
        pytest.param("cpu", "auto", "mps", "cpu", "mps", id="explicit-argument-forwarded"),
        # device="auto" means the auto policy for the pose model, but the detector keeps an
        # explicit detector.device and otherwise resolves its own auto policy (not forwarded)
        pytest.param("cpu", "auto", "auto", "cpu", None, id="auto-argument-keeps-detector-cpu"),
        pytest.param("cuda:1", "auto", "auto", "cuda:1", None, id="auto-argument-keeps-detector-cuda1"),
        pytest.param("auto", "cpu", "auto", "auto", None, id="auto-argument-detector-auto-policy"),
    ],
)
@patch("deeplabcut.pose_estimation_pytorch.apis.training.destroy_file_logging")
@patch("deeplabcut.pose_estimation_pytorch.apis.training.setup_file_logging")
@patch("deeplabcut.pose_estimation_pytorch.apis.training.train")
@patch("deeplabcut.pose_estimation_pytorch.apis.training.DLCLoader")
def test_train_network_detector_device_inheritance(
    mock_loader_cls: Mock,
    mock_train: Mock,
    mock_setup_file_logging: Mock,
    mock_destroy_file_logging: Mock,
    detector_device: str,
    top_level_device: str,
    explicit_device: str | None,
    expected_detector_device: str,
    expected_detector_arg: str | None,
    tmp_path: Path,
) -> None:
    """A detector left on auto inherits the top-level device; an explicit detector.device is kept."""
    pose_config = _top_down_config(tmp_path)
    pose_config["device"] = top_level_device
    pose_config["detector"]["device"] = detector_device
    loader = Mock()
    loader.model_cfg = pose_config
    loader.model_folder = tmp_path
    loader.model_config_path = tmp_path / "pytorch_config.yaml"
    mock_loader_cls.return_value = loader

    train_network(config=tmp_path / "config.yaml", device=explicit_device)

    assert mock_train.call_count == 2
    detector_call, pose_call = mock_train.call_args_list
    assert detector_call.kwargs["task"] == Task.DETECT
    assert detector_call.kwargs["run_config"]["device"] == expected_detector_device
    assert detector_call.kwargs["device"] == expected_detector_arg
    assert pose_call.kwargs["task"] == Task.TOP_DOWN
    assert pose_call.kwargs["run_config"]["device"] == top_level_device
    assert pose_call.kwargs["device"] == explicit_device


@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_transforms", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.DETECTORS.build", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_training_runner", return_value=Mock())
def test_train_uses_the_training_registry_not_the_inference_one(
    mock_build_runner: Mock, mock_build_detector: Mock, mock_build_transforms: Mock, tmp_path: Path, monkeypatch
) -> None:
    """Faster R-CNN v2 is validated for inference on MPS but must still train on the CPU."""
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: True)
    assert _FRCNN_VARIANT in dlc_utils.DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS
    assert _FRCNN_VARIANT not in dlc_utils.DETECTOR_MPS_VALIDATED_VARIANTS
    detector_config = DetectorConfig.build(1, DetectorType.FASTERRCNN_RESNET50_FPN_V2)
    with pytest.warns(UserWarning, match="watchdog"):
        train(
            loader=_make_loader(tmp_path, detector_config), run_config=detector_config, task=Task.DETECT, device="mps"
        )
    assert mock_build_runner.call_args.kwargs["device"] == "cpu"
