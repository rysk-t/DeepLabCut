---
deeplabcut:
  last_content_updated: '2026-02-10'
  last_metadata_updated: '2026-03-06'
  ignore: false
  visibility: online
  status: outdated
  recommendation: update
  notes: Useful but needs to be updated and clarified.
---

(file:hardware-requirements)=

# Technical & hardware considerations

## Quick summary

On our {ref}`install page <sec:hardware-considerations-during-install>`
we highlight that for GPU computing through standard installation you need a NVIDIA GPU, with at least 8 GB of memory. If you have an Intel or AMD GPU, and are on windows, there is an alternative method of installation available which is shown on the [installation tips page](installation-tips) under "How to install Deeplabcut for Intel and AMD GPUs".
Note, some info is repeated here, and will be updated as systems and hardware changes.

### Computer

For reference, we use e.g. Dell workstations (79xx series) with **Ubuntu 16.04 LTS, 18.04 LTS, or 20.04 LTS** and run a Docker container that has TensorFlow, etc. installed (https://github.com/DeepLabCut/Docker4DeepLabCut2.0).

### Computer hardware

Ideally, you will use a strong GPU with *at least* 8GB memory such as the [NVIDIA GeForce 1080 Ti, 2080 Ti, or 3090](https://marketplace.nvidia.com/en-us/consumer/graphics-cards/). A GPU is not strictly necessary, but on a CPU the (training and evaluation) code is considerably slower (10x) for ResNets, but MobileNets and EfficientNets are slightly faster. Still, a GPU will give you a massive speed boost. You might also consider using cloud computing services like [Google cloud/amazon web services](https://github.com/DeepLabCut/DeepLabCut/issues/47) or Google Colaboratory.

```{note}
If you encounter errors during inference related to
`torch.inference_mode` and DirectML, set the environment variable
`DLC_DIRECTML_NO_GRAD=true` before starting Python. This switches the inference
context to `torch.no_grad`, which is compatible with the DirectML execution path.
```

#### Apple Silicon (MPS)

On Apple Silicon Macs, the PyTorch engine can use the GPU through Metal (`mps`).
Which models actually run there depends on the model type:

- **Pose models**: with `device: auto`, ResNet backbones run on MPS. Other
  backbones (e.g. HRNets) stay on the CPU unless `device` is set explicitly in
  `pytorch_config.yaml`.
- **Object detectors** (top-down / multi-animal models): a detector runs on MPS
  only when MPS is available, the installed torch is a release `>= 2.12`, and
  the detector variant has been validated on MPS for what it is about to do.
  Currently validated for training and inference: `ssdlite` (the default
  detector), on Apple Silicon with torch 2.12.1 and 2.13.0. Validated for
  inference only: `fasterrcnn_resnet50_fpn` and `fasterrcnn_resnet50_fpn_v2`
  (their predictions matched the CPU; their training on MPS is broken, see
  below). Anything else runs on the CPU, and so does the training of every
  Faster R-CNN variant; when MPS was requested explicitly (through the `device`
  argument, `detector.device`, or a top-level `device: mps`), a warning says
  why the detector was moved. On older torch versions, detectors
  hang or return corrupted detections on MPS (see
  [#3155](https://github.com/DeepLabCut/DeepLabCut/issues/3155) and
  [#2853](https://github.com/DeepLabCut/DeepLabCut/issues/2853)), and training
  the Faster R-CNN variants on MPS hung the GPU (`fasterrcnn_resnet50_fpn` on
  the first iteration; `fasterrcnn_resnet50_fpn_v2` hard enough to trigger a
  macOS watchdog kernel panic and reboot). The cause is a bug in the MPS backward
  kernel of torchvision's `roi_align`, which over-accumulates gradients in
  proportion to the number of RoIs (inference is unaffected); it is fixed in
  [pytorch/vision#9510](https://github.com/pytorch/vision/pull/9510), released
  in torchvision 0.29.0 (which requires torch 2.14). Faster R-CNN training on
  MPS has not been validated against that release yet, so it stays on the CPU
  on Apple Silicon. Faster R-CNN *inference* keeps running on MPS for the two
  validated variants (`analyze_videos` used to do this without any check of
  the torch version; the check is now applied there as well).
- **Which device the detector uses**: the `device` argument of the API
  (`analyze_videos`, `evaluate_network`, `train_network`) takes precedence, then
  `detector.device` in `pytorch_config.yaml` (when it is not `auto`), then the
  top-level `device`: an explicit top-level device is inherited when
  `detector.device` is `auto`, and when both are `auto` the detector picks its
  own device by the rule above (so a top-down HRNet model that stays on the CPU
  still trains and runs its `ssdlite` detector on MPS). The same rule applies
  to training and inference. `analyze_images`, `extract_maps` and
  `create_tracking_dataset` currently resolve one device for the pose model and
  the detector and do not read `detector.device`.
- To keep the detector on the CPU explicitly, pass `device="cpu"`, or set
  `detector.device: cpu` in `pytorch_config.yaml` (honoured by `train_network`,
  `analyze_videos` and `evaluate_network` when no `device` argument is given).
  Either silences the warning; a `device="mps"` argument overrides the
  configuration pin, so drop the argument or pass `device="cpu"` instead.

### Camera Hardware

The software is very robust to track data from any camera (cell phone cameras, grayscale, color; captured under infrared light, different manufacturers, etc.). See demos on our [website](https://www.mousemotorlab.org/deeplabcut/).

### Software

**Operating System:** Linux (Ubuntu), MacOS\* (Mojave), or Windows 10. However, the authors strongly recommend Ubuntu! \*MacOS does not support NVIDIA GPUs. On Apple Silicon the PyTorch engine can use the GPU through MPS for ResNet pose models, the `ssdlite` detector (training and inference) and the two validated Faster R-CNN detectors (inference only), see *Apple Silicon (MPS)* above; other models run on the CPU, so for those we suggest labeling and refining data locally and pushing the project to a cloud resource for GPU computing steps.

**Anaconda/Python3:** Anaconda: a free and open source distribution of the Python programming language (download from https://www.anaconda.com/). DeepLabCut is written in Python 3 (https://www.python.org/) and not compatible with Python 2.

**For the TensorFlow Engine:** You will need [TensorFlow](https://www.tensorflow.org/).
We used version 1.0 in the paper, later versions also work with the provided code (we
tested **TensorFlow versions 1.0 to 1.15, and 2.0 to 2.18**); we
recommend TF2.12 for Python 3.10 with GPU support. Note that native GPU support for Windows was dropped after TF version 2.10. We recommend Windows users to install [the Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) if they want to keep GPU support with TensorFlow.

To note, is it possible to run DeepLabCut on your CPU, but it will be VERY slow (see:
[Mathis & Warren](https://www.biorxiv.org/content/early/2018/10/30/457242)). However, this is the preferred path if you want to test
DeepLabCut on your own computer/data before purchasing a GPU, with the added benefit of
a straightforward installation! Otherwise, use our COLAB notebooks for GPU access for
testing.

Docker: We highly recommend advanced users use the supplied [Docker container](docker-containers).

NOTE: [Currently GPU support in Docker Desktop is only available on Windows with the
WSL2 backend.](https://docs.docker.com/desktop/features/gpu/)
