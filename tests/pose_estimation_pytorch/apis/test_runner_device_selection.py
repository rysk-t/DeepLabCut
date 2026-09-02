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
"""Tests for the device the inference-runner builders hand to ``build_inference_runner``.

The detector device follows one precedence rule: an explicit ``device`` argument wins,
then an explicit ``detector.device`` in the model configuration, then (when the top-level
device is ``auto`` too) the detector's own auto policy, otherwise the pose device.
A ``device="auto"`` argument applies the auto policy to both models.
The result then goes through ``resolve_detector_device``: MPS only survives for
validated variants, otherwise the detector runs on the CPU with a warning. The pose
device is never touched. Every test fakes an Apple Silicon machine, so nothing here
depends on the hardware running the tests.

``analyze_videos``, ``evaluate_network`` and ``train_network`` forward their ``device``
argument to the builders, so the rule above holds at those entry points. The tests named
``*_top_level_device_overwritten_*`` pin the builder-level rule for a caller that has
already written a device into the top-level configuration and passes no ``device``: an
explicit ``detector.device`` then still wins over the inherited top-level device.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml

import deeplabcut.pose_estimation_pytorch.apis.utils as api_utils
import deeplabcut.pose_estimation_pytorch.utils as dlc_utils
from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.config.model import DetectorModelConfig
from deeplabcut.pose_estimation_pytorch.config.pose import DetectorConfig
from deeplabcut.pose_estimation_pytorch.runners.inference import DetectorInferenceRunner
from deeplabcut.pose_estimation_pytorch.task import Task

FIXTURE_PROJECT = Path(__file__).parent.parent / "config" / "fixtures" / "multianimal_project_v0.yaml"

SSDLITE = "ssdlite"  # validated on MPS
FASTERRCNN = "fasterrcnn_resnet50_fpn_v2"  # treated as unvalidated by the apple_silicon fixture (see below)
MOBILENET = "fasterrcnn_mobilenet_v3_large_fpn"  # validated for nothing
DEFAULT_TRAINING_REGISTRY = frozenset(dlc_utils.DETECTOR_MPS_VALIDATED_VARIANTS)
DEFAULT_INFERENCE_REGISTRY = frozenset(dlc_utils.DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS)
# the torchvision COCO detectors; all Faster R-CNN, so none is validated under the apple_silicon
# fixture (the shipped inference registry validates v1/v2, see TestShippedRegistries)
TORCHVISION_NAMES = tuple(api_utils.TORCHVISION_DETECTORS)

FALLBACK_MARKER = "The detector runs on the CPU instead"


def fallback_warnings(recwarn) -> list[str]:
    """Returns the messages of the recorded detector CPU-fallback warnings."""
    return [str(w.message) for w in recwarn if FALLBACK_MARKER in str(w.message)]


@pytest.fixture
def make_pose_config(tmp_path):
    """Factory for a real top-down PoseConfig (resnet_50, device auto) with the given detector."""
    with open(FIXTURE_PROJECT) as f:
        project = yaml.safe_load(f)
    project["uniquebodyparts"] = []  # unique bodyparts are unsupported for top-down

    def _make(
        detector_type: str = SSDLITE, detector_device: str = "auto", device: str = "auto", net_type: str = "resnet_50"
    ):
        config = make_pytorch_pose_config(
            project,
            tmp_path / "pose_cfg.yaml",
            net_type=net_type,
            top_down=True,
            detector_type=detector_type,
            save=False,
        )
        config["device"] = device
        config["detector"]["device"] = detector_device
        return config

    return _make


@pytest.fixture
def apple_silicon(monkeypatch):
    """MPS built and available, no CUDA, torch above the floor and ssdlite the only validated variant."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: True)
    # both registries narrowed to ssdlite: these tests pin the gate mechanics, not the shipped lists
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", frozenset({SSDLITE}))
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS", frozenset({SSDLITE}))


@pytest.fixture
def captured(monkeypatch, apple_silicon) -> dict[Task, str]:
    """Stubs model construction and records the device given to build_inference_runner per task.

    Snapshot paths are never opened: they are only forwarded to the stubbed builder.
    """
    devices: dict[Task, str] = {}

    def fake_build_inference_runner(*, task, device, **kwargs):
        devices[task] = device
        runner = MagicMock(spec=DetectorInferenceRunner)
        runner.device = device
        return runner

    monkeypatch.setattr(api_utils, "build_inference_runner", fake_build_inference_runner)
    monkeypatch.setattr(api_utils.DETECTORS, "build", lambda cfg: MagicMock())
    monkeypatch.setattr(api_utils.PoseModel, "build", lambda cfg: MagicMock())
    monkeypatch.setattr(api_utils, "build_transforms", lambda cfg: MagicMock())
    return devices


@pytest.fixture
def torchvision_stub(monkeypatch) -> MagicMock:
    """Registers stub torchvision constructors so that no weights are downloaded."""
    detector = MagicMock()
    detector.eval.return_value = detector
    for name in TORCHVISION_NAMES:
        monkeypatch.setitem(
            api_utils.TORCHVISION_DETECTORS,
            name,
            {"weights": None, "fn": lambda weights, box_score_thresh: detector},
        )
    monkeypatch.setattr(api_utils, "FilteredDetector", lambda *args, **kwargs: MagicMock())
    return detector


# ---------------------------------------------------------------------------
# get_inference_runners
# ---------------------------------------------------------------------------


class TestGetInferenceRunners:
    @staticmethod
    def run(config, **kwargs):
        return api_utils.get_inference_runners(
            model_config=config,
            snapshot_path="snapshot.pt",
            detector_path="snapshot-detector.pt",
            **kwargs,
        )

    def test_auto_validated_detector_inherits_mps(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(SSDLITE))
        assert captured[Task.TOP_DOWN] == "mps"
        assert captured[Task.DETECT] == "mps"
        assert fallback_warnings(recwarn) == []

    def test_auto_unvalidated_detector_stays_on_cpu(self, make_pose_config, captured, recwarn):
        # both devices on auto: the detector follows its own auto policy, so nobody asked
        # for MPS and no warning is emitted
        self.run(make_pose_config(FASTERRCNN))
        assert captured[Task.TOP_DOWN] == "mps"
        assert captured[Task.DETECT] == "cpu"
        assert fallback_warnings(recwarn) == []

    def test_inherited_explicit_mps_unvalidated_detector_falls_back_to_cpu(self, make_pose_config, captured, recwarn):
        # an explicit top-level device: mps is inherited by a detector on auto, then gated
        self.run(make_pose_config(FASTERRCNN, device="mps"))
        assert captured[Task.TOP_DOWN] == "mps"
        assert captured[Task.DETECT] == "cpu"
        (message,) = fallback_warnings(recwarn)
        assert FASTERRCNN in message

    def test_explicit_mps_unvalidated_detector_falls_back_to_cpu(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(FASTERRCNN), device="mps")
        assert captured[Task.TOP_DOWN] == "mps"
        assert captured[Task.DETECT] == "cpu"
        (message,) = fallback_warnings(recwarn)
        assert FASTERRCNN in message

    @pytest.mark.parametrize("detector_type", [SSDLITE, FASTERRCNN])
    @pytest.mark.parametrize("device", ["cuda:1", "cpu"])
    def test_explicit_non_mps_device_passes_through(self, make_pose_config, captured, recwarn, device, detector_type):
        self.run(make_pose_config(detector_type), device=device)
        assert captured[Task.TOP_DOWN] == device
        assert captured[Task.DETECT] == device
        assert fallback_warnings(recwarn) == []

    def test_config_detector_device_honoured_without_device_argument(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(FASTERRCNN, detector_device="cpu"))
        assert captured[Task.TOP_DOWN] == "mps"
        assert captured[Task.DETECT] == "cpu"
        assert fallback_warnings(recwarn) == []

    def test_device_argument_beats_config_detector_device(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(SSDLITE, detector_device="cpu"), device="mps")
        assert captured[Task.TOP_DOWN] == "mps"
        assert captured[Task.DETECT] == "mps"
        assert fallback_warnings(recwarn) == []

    @pytest.mark.parametrize(
        "detector_device, expected_detector",
        [
            ("auto", "cuda"),  # the detector inherits the pose device, as before
            ("cpu", "cpu"),  # an explicit config pin is honoured on CUDA machines too
        ],
    )
    def test_cuda_machine_detector_device(
        self, make_pose_config, captured, recwarn, monkeypatch, detector_device, expected_detector
    ):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        self.run(make_pose_config(FASTERRCNN, detector_device=detector_device))
        assert captured[Task.TOP_DOWN] == "cuda"
        assert captured[Task.DETECT] == expected_detector
        assert fallback_warnings(recwarn) == []

    @pytest.mark.parametrize(
        "top_level, detector_type, detector_device, cuda, expected_detector",
        [
            ("mps", SSDLITE, "cpu", False, "cpu"),  # detector pinned to the CPU next to an MPS pose model
            ("cpu", SSDLITE, "mps", False, "mps"),  # validated variant pinned to MPS
            ("cpu", FASTERRCNN, "cuda:1", True, "cuda:1"),  # config pin honoured on a CUDA box
        ],
    )
    def test_top_level_device_overwritten_config_detector_device_wins(
        self,
        make_pose_config,
        captured,
        recwarn,
        monkeypatch,
        top_level,
        detector_type,
        detector_device,
        cuda,
        expected_detector,
    ):
        # A caller that has already written a device into the top-level config and passes
        # no device: an explicit detector.device still wins over the inherited device.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        self.run(make_pose_config(detector_type, detector_device=detector_device, device=top_level))
        assert captured[Task.TOP_DOWN] == top_level
        assert captured[Task.DETECT] == expected_detector
        assert fallback_warnings(recwarn) == []

    @pytest.mark.parametrize("top_level", ["auto", "cpu"])
    @pytest.mark.parametrize("detector_device, expected", [("auto", "mps"), ("cpu", "cpu")])
    def test_auto_argument_applies_the_auto_policy(
        self, make_pose_config, captured, recwarn, top_level, detector_device, expected
    ):
        # analyze_videos()/evaluate_network() may forward device="auto": it must never reach
        # the runners unresolved, it overrides an explicit top-level device with the auto
        # policy (like train()), and an explicit detector.device is still consulted.
        api_utils.get_inference_runners(
            model_config=make_pose_config(SSDLITE, detector_device=detector_device, device=top_level),
            snapshot_path="snapshot.pt",
            detector_path="snapshot-detector.pt",
            device="auto",
        )
        assert captured[Task.TOP_DOWN] == "mps"
        assert captured[Task.DETECT] == expected
        assert fallback_warnings(recwarn) == []

    def test_without_detector_path_no_detector_runner_and_no_warning(self, make_pose_config, captured, recwarn):
        pose_runner, detector_runner = api_utils.get_inference_runners(
            model_config=make_pose_config(FASTERRCNN),
            snapshot_path="snapshot.pt",
            detector_path=None,
        )
        assert detector_runner is None
        assert pose_runner.device == "mps"
        assert captured[Task.TOP_DOWN] == "mps"
        assert Task.DETECT not in captured
        assert fallback_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# get_detector_inference_runner
# ---------------------------------------------------------------------------


class TestGetDetectorInferenceRunner:
    @staticmethod
    def run(config, **kwargs):
        return api_utils.get_detector_inference_runner(
            model_config=config,
            snapshot_path="snapshot-detector.pt",
            **kwargs,
        )

    @pytest.mark.parametrize("top_level", ["auto", "cpu"])
    @pytest.mark.parametrize(
        "detector_type, detector_device, expected",
        [(SSDLITE, "auto", "mps"), (SSDLITE, "cpu", "cpu"), (FASTERRCNN, "auto", "cpu")],
    )
    def test_auto_argument_applies_the_auto_policy(
        self, make_pose_config, captured, recwarn, top_level, detector_type, detector_device, expected
    ):
        config = make_pose_config(detector_type, detector_device=detector_device, device=top_level)
        runner = self.run(config, device="auto")
        assert runner.device == expected
        assert captured[Task.DETECT] == expected
        assert "auto" not in captured.values()
        assert config["device"] == top_level  # the configuration is not mutated

    def test_device_none_unvalidated_detector_stays_on_cpu(self, make_pose_config, captured, recwarn):
        # Without a device argument the resnet pose config resolves to MPS, which used to
        # reach an unvalidated detector ungated (the analyze_videos() path before this fix).
        # Both devices on auto: the detector resolves its own policy, no warning.
        runner = self.run(make_pose_config(FASTERRCNN))
        assert runner.device == "cpu"
        assert captured[Task.DETECT] == "cpu"
        assert fallback_warnings(recwarn) == []

    def test_device_none_inherited_explicit_mps_falls_back_to_cpu(self, make_pose_config, captured, recwarn):
        # a top-level device: mps written by analyze_videos(device="mps") or set in the
        # configuration is inherited by a detector on auto, then gated with a warning
        runner = self.run(make_pose_config(FASTERRCNN, device="mps"))
        assert runner.device == "cpu"
        assert captured[Task.DETECT] == "cpu"
        (message,) = fallback_warnings(recwarn)
        assert FASTERRCNN in message

    def test_device_none_validated_detector_inherits_mps(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(SSDLITE))
        assert captured[Task.DETECT] == "mps"
        assert fallback_warnings(recwarn) == []

    def test_explicit_mps_validated_detector(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(SSDLITE), device="mps")
        assert captured[Task.DETECT] == "mps"
        assert fallback_warnings(recwarn) == []

    def test_explicit_indexed_mps_unvalidated_detector_falls_back_to_cpu(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(FASTERRCNN), device="mps:0")
        assert captured[Task.DETECT] == "cpu"
        (message,) = fallback_warnings(recwarn)
        assert FASTERRCNN in message

    @pytest.mark.parametrize("detector_type", [SSDLITE, FASTERRCNN])
    @pytest.mark.parametrize("device", ["cuda:1", "cpu"])
    def test_explicit_non_mps_device_passes_through(self, make_pose_config, captured, recwarn, device, detector_type):
        self.run(make_pose_config(detector_type), device=device)
        assert captured[Task.DETECT] == device
        assert fallback_warnings(recwarn) == []

    def test_config_detector_device_honoured_without_device_argument(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(FASTERRCNN, detector_device="cpu"))
        assert captured[Task.DETECT] == "cpu"
        assert fallback_warnings(recwarn) == []

    def test_device_argument_beats_config_detector_device(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(SSDLITE, detector_device="cpu"), device="mps")
        assert captured[Task.DETECT] == "mps"
        assert fallback_warnings(recwarn) == []

    @pytest.mark.parametrize(
        "top_level, detector_type, detector_device, cuda, expected_detector",
        [
            ("mps", SSDLITE, "cpu", False, "cpu"),  # detector pinned to the CPU next to an MPS pose model
            ("cpu", SSDLITE, "mps", False, "mps"),  # validated variant pinned to MPS
            ("cpu", FASTERRCNN, "cuda:1", True, "cuda:1"),  # config pin honoured on a CUDA box
        ],
    )
    def test_top_level_device_overwritten_config_detector_device_wins(
        self,
        make_pose_config,
        captured,
        recwarn,
        monkeypatch,
        top_level,
        detector_type,
        detector_device,
        cuda,
        expected_detector,
    ):
        # A caller that has already written a device into the top-level config and passes
        # no device: an explicit detector.device still wins over the inherited device.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        self.run(make_pose_config(detector_type, detector_device=detector_device, device=top_level))
        assert captured[Task.DETECT] == expected_detector
        assert fallback_warnings(recwarn) == []

    def test_top_level_device_overwritten_auto_detector_inherits_it(self, make_pose_config, captured, recwarn):
        self.run(make_pose_config(FASTERRCNN, device="cpu"))
        assert captured[Task.DETECT] == "cpu"
        assert fallback_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# get_filtered_coco_detector_inference_runner
# ---------------------------------------------------------------------------


class TestGetFilteredCocoDetectorInferenceRunner:
    @staticmethod
    def run(model_name, **kwargs):
        return api_utils.get_filtered_coco_detector_inference_runner(model_name=model_name, category_id=1, **kwargs)

    @pytest.mark.parametrize("model_name", TORCHVISION_NAMES)
    def test_explicit_mps_falls_back_to_cpu(self, captured, torchvision_stub, recwarn, model_name):
        self.run(model_name, device="mps", max_individuals=3, color_mode="RGB")
        assert captured[Task.DETECT] == "cpu"
        torchvision_stub.to.assert_called_once_with("cpu")
        (message,) = fallback_warnings(recwarn)
        assert model_name in message

    @pytest.mark.parametrize("model_name", TORCHVISION_NAMES)
    def test_auto_argument_with_config_applies_the_auto_policy(
        self, make_pose_config, captured, torchvision_stub, recwarn, monkeypatch, model_name
    ):
        # with a config on auto the result is the same as without a config: the detector's
        # own policy, keyed on the model name, whatever the pose backbone resolves to
        self.run(model_name, device="auto", model_config=make_pose_config(FASTERRCNN, device="cpu"))
        assert captured[Task.DETECT] == "cpu"
        assert "auto" not in captured.values()
        assert fallback_warnings(recwarn) == []
        monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS", frozenset({model_name}))
        self.run(model_name, device="auto", model_config=make_pose_config(FASTERRCNN, net_type="hrnet_w18"))
        assert captured[Task.DETECT] == "mps"  # even though the HRNet pose model stays on the CPU
        assert fallback_warnings(recwarn) == []

    @pytest.mark.parametrize("model_name", TORCHVISION_NAMES)
    def test_auto_argument_without_config_applies_the_auto_policy(
        self, captured, torchvision_stub, recwarn, monkeypatch, model_name
    ):
        # no model_config to resolve from: "auto" must still become a real device
        self.run(model_name, device="auto", max_individuals=3, color_mode="RGB")
        assert captured[Task.DETECT] == "cpu"  # not validated under the apple_silicon fixture
        assert fallback_warnings(recwarn) == []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        self.run(model_name, device="auto", max_individuals=3, color_mode="RGB")
        assert captured[Task.DETECT] == "cuda"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS", frozenset({model_name}))
        self.run(model_name, device="auto", max_individuals=3, color_mode="RGB")
        assert captured[Task.DETECT] == "mps"
        assert "auto" not in captured.values()

    @pytest.mark.parametrize("model_name", TORCHVISION_NAMES)
    def test_device_none_with_auto_config_stays_on_cpu(
        self, make_pose_config, captured, torchvision_stub, recwarn, model_name
    ):
        # config on auto: the (unvalidated) torchvision detector picks the CPU by its own
        # policy, no matter that the resnet pose model resolves to MPS; nobody asked for
        # MPS, so there is no warning
        self.run(model_name, model_config=make_pose_config(SSDLITE))
        assert captured[Task.DETECT] == "cpu"
        assert fallback_warnings(recwarn) == []

    @pytest.mark.parametrize("model_name", TORCHVISION_NAMES)
    @pytest.mark.parametrize(
        "top_level, expected, warns", [("cpu", "cpu", False), ("cuda:1", "cuda:1", False), ("mps", "cpu", True)]
    )
    def test_device_none_inherits_explicit_top_level_device(
        self, make_pose_config, captured, torchvision_stub, recwarn, model_name, top_level, expected, warns
    ):
        self.run(model_name, model_config=make_pose_config(SSDLITE, device=top_level))
        assert captured[Task.DETECT] == expected
        assert (len(fallback_warnings(recwarn)) == 1) is warns

    def test_validated_name_would_keep_mps(self, captured, torchvision_stub, recwarn, monkeypatch):
        # The gate is keyed on the torchvision model name; a validated name keeps MPS.
        monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS", frozenset({TORCHVISION_NAMES[0]}))
        self.run(TORCHVISION_NAMES[0], device="mps", max_individuals=3, color_mode="RGB")
        assert captured[Task.DETECT] == "mps"
        torchvision_stub.to.assert_called_once_with("mps")
        assert fallback_warnings(recwarn) == []

    @pytest.mark.parametrize("model_name", TORCHVISION_NAMES)
    def test_cpu_passes_through(self, captured, torchvision_stub, recwarn, model_name):
        self.run(model_name, device="cpu", max_individuals=3, color_mode="RGB")
        assert captured[Task.DETECT] == "cpu"
        assert fallback_warnings(recwarn) == []

    def test_unsupported_name_raises_before_the_gate(self, captured, torchvision_stub, recwarn):
        with pytest.raises(ValueError, match="Unsupported model"):
            self.run("ssdlite320_mobilenet_v3_large", device="mps", max_individuals=3, color_mode="RGB")
        assert fallback_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# the detector's auto policy is the same for training and inference
# ---------------------------------------------------------------------------


class TestDetectorAutoPolicyAcrossBackbones:
    """With both devices on auto the detector picks its own device, whatever the pose model resolved to.

    train_network() resolves the detector configuration on its own; the inference builders
    must agree, so a top-down HRNet model (CPU on Apple Silicon) still gets its ssdlite
    detector on MPS, and a Faster R-CNN detector stays on the CPU next to a ResNet on MPS.
    """

    @pytest.mark.parametrize("net_type, expected_pose", [("resnet_50", "mps"), ("hrnet_w18", "cpu")])
    @pytest.mark.parametrize("detector_type, expected_detector", [(SSDLITE, "mps"), (FASTERRCNN, "cpu")])
    def test_get_inference_runners(
        self, make_pose_config, captured, net_type, expected_pose, detector_type, expected_detector
    ):
        api_utils.get_inference_runners(
            model_config=make_pose_config(detector_type, net_type=net_type),
            snapshot_path="snapshot.pt",
            detector_path="snapshot-detector.pt",
        )
        assert captured[Task.TOP_DOWN] == expected_pose
        assert captured[Task.DETECT] == expected_detector

    @pytest.mark.parametrize("net_type", ["resnet_50", "hrnet_w18"])
    @pytest.mark.parametrize("detector_type, expected_detector", [(SSDLITE, "mps"), (FASTERRCNN, "cpu")])
    def test_get_detector_inference_runner(
        self, make_pose_config, captured, net_type, detector_type, expected_detector
    ):
        config = make_pose_config(detector_type, net_type=net_type)
        runner = api_utils.get_detector_inference_runner(model_config=config, snapshot_path="snapshot-detector.pt")
        assert runner.device == expected_detector
        # what train_network() resolves for the same configuration (detector config on auto)
        assert dlc_utils.resolve_device(config["detector"]) == expected_detector

    def test_explicit_top_level_device_is_inherited(self, make_pose_config, captured, recwarn):
        api_utils.get_inference_runners(
            model_config=make_pose_config(SSDLITE, device="cpu"),
            snapshot_path="snapshot.pt",
            detector_path="snapshot-detector.pt",
        )
        assert captured[Task.TOP_DOWN] == "cpu"
        assert captured[Task.DETECT] == "cpu"
        assert fallback_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# the shipped registries: Faster R-CNN v1/v2 run inference on MPS, never training
# ---------------------------------------------------------------------------


@pytest.fixture
def shipped_registries(monkeypatch, captured):
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", DEFAULT_TRAINING_REGISTRY)
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_INFERENCE_VALIDATED_VARIANTS", DEFAULT_INFERENCE_REGISTRY)


class TestShippedRegistries:
    @pytest.mark.parametrize("net_type", ["resnet_50", "hrnet_w18"])
    def test_fasterrcnn_v2_inference_uses_mps_under_auto(
        self, make_pose_config, captured, shipped_registries, recwarn, net_type
    ):
        api_utils.get_inference_runners(
            model_config=make_pose_config(FASTERRCNN, net_type=net_type),
            snapshot_path="snapshot.pt",
            detector_path="snapshot-detector.pt",
        )
        assert captured[Task.DETECT] == "mps"
        assert fallback_warnings(recwarn) == []

    def test_fasterrcnn_v2_explicit_mps_kept_for_inference(
        self, make_pose_config, captured, shipped_registries, recwarn
    ):
        runner = api_utils.get_detector_inference_runner(
            model_config=make_pose_config(FASTERRCNN), snapshot_path="snapshot-detector.pt", device="mps"
        )
        assert runner.device == "mps"
        assert fallback_warnings(recwarn) == []

    def test_mobilenet_stays_on_cpu(self, make_pose_config, captured, shipped_registries, recwarn):
        runner = api_utils.get_detector_inference_runner(
            model_config=make_pose_config(MOBILENET), snapshot_path="snapshot-detector.pt"
        )
        assert runner.device == "cpu"
        assert fallback_warnings(recwarn) == []
        api_utils.get_detector_inference_runner(
            model_config=make_pose_config(MOBILENET), snapshot_path="snapshot-detector.pt", device="mps"
        )
        (message,) = fallback_warnings(recwarn)
        assert MOBILENET in message and "for inference" in message

    @pytest.mark.parametrize(
        "model_name, expected",
        [("fasterrcnn_resnet50_fpn", "mps"), ("fasterrcnn_resnet50_fpn_v2", "mps"), (MOBILENET, "cpu")],
    )
    def test_filtered_coco_detectors(
        self, make_pose_config, captured, torchvision_stub, shipped_registries, recwarn, model_name, expected
    ):
        api_utils.get_filtered_coco_detector_inference_runner(
            model_name=model_name, category_id=1, model_config=make_pose_config(SSDLITE)
        )
        assert captured[Task.DETECT] == expected
        api_utils.get_filtered_coco_detector_inference_runner(
            model_name=model_name, category_id=1, device="mps", max_individuals=3, color_mode="RGB"
        )
        assert captured[Task.DETECT] == expected
        assert (len(fallback_warnings(recwarn)) == 1) is (expected == "cpu")

    def test_training_registry_is_not_widened(self, shipped_registries):
        # the inference list must never leak into training: Faster R-CNN training on MPS can panic the machine
        assert dlc_utils.resolve_detector_device("mps", FASTERRCNN, for_training=True) == "cpu"
        assert (
            dlc_utils.resolve_device(
                DetectorConfig(model=DetectorModelConfig(type="FasterRCNN", variant=FASTERRCNN), device="auto")
            )
            == "cpu"
        )
