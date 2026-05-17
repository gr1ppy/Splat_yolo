<p align="center">
  <h1 align="center">🔬 Gaussian Preprocessor</h1>
  <p align="center">
    <strong>A Differentiable 2D Gaussian Splatting Preprocessor for YOLO-World</strong>
  </p>
  <p align="center">
    <em>End-to-end learnable image reconstruction via sparse Gaussians that maximize object detection accuracy</em>
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-blue?style=for-the-badge" alt="Quick Start"></a>
    <a href="#-architecture"><img src="https://img.shields.io/badge/Architecture-teal?style=for-the-badge" alt="Architecture"></a>
    <a href="#-modules"><img src="https://img.shields.io/badge/Modules-purple?style=for-the-badge" alt="Modules"></a>
    <a href="#-configuration"><img src="https://img.shields.io/badge/Config-orange?style=for-the-badge" alt="Config"></a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/CUDA-Required-76b900?logo=nvidia&logoColor=white" alt="CUDA">
    <img src="https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white" alt="Python">
  </p>
</p>

---

## 📖 Overview

[Based on Image-GS Algorithm: (https://github.com/NYU-ICL/image-gs);(https://arxiv.org/abs/2407.01866)]**Gaussian Preprocessor** prepends a differentiable 2D Gaussian rendering pipeline to the [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) object detector. Instead of feeding raw pixels into the backbone, input images are encoded into a sparse set of *N* 2D Gaussians that cluster around semantically important regions. These Gaussians are rendered back into a 2D tensor and passed to YOLO-World. The entire pipeline is trained **end-to-end** — YOLO's detection loss backpropagates through the renderer, teaching the saliency network and parameter heads to "draw" images that maximize detection accuracy.

### Key Highlights

- 🔄 **Fully Differentiable** — Gradients flow from detection loss back through the CUDA renderer to all learnable modules.
- ⚡ **GPU-Accelerated Rendering** — Uses [gsplat](https://github.com/nerfstudio-project/gsplat) CUDA kernels with tile-based rasterization.
- 🧠 **Multiple Saliency Modes** — Learned CNN saliency or parameter-free Sobel edge detection.
- 🎛️ **Pluggable Parameter Heads** — Shared MLP, independent per-param networks, direct optimization, or deformable point-to-image cross-attention transformer.
- ✂️ **Adaptive Gaussian Pruning** — Removes degenerate Gaussians via deterministic quality checks and/or a learned opacity network with straight-through estimation.
- 🎨 **Flexible Output Blending** — Pure rendering, learnable alpha blending, additive composition, or obscure (occlusion) mode.

---

## 🏗️ Architecture

The pipeline executes as a single continuous `torch.nn.Module.forward()` pass:

```
Input Image [B, 3, H, W]
  │
  ▼
┌─────────────────────────────┐
│  Saliency CNN / Edge Det.   │  → features [B, C, H, W]
│  (importance map [B,1,H,W]) │    + probability map
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Differentiable Point       │  → Gaussian centers [B, N, 2]
│  Sampler (multinomial)      │    (normalized [0,1] coords)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Parameter Head             │  → scale [2], rotation [1],
│  (MLP / Transformer / etc.) │    color [3] per Gaussian
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Gaussian Pruner            │  → removes low-quality
│  (deterministic + learned)  │    Gaussians before render
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  gsplat CUDA Renderer       │  → rendered image [B, 3, H, W]
│  (tile-based rasterization) │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  YOLO-World Backbone        │  → detection predictions
│  + Detection Head           │
└─────────────────────────────┘

  ◄── Detection loss backpropagates through the entire pipeline ──►
```

---

## 📦 Modules

### Module 1 — `SaliencyCNN`

Lightweight CNN using depthwise-separable 1×1 convolutions (`DWConv`) that preserves full spatial resolution. Outputs a feature map and a sigmoid saliency (importance) map for Gaussian center sampling.

```python
features, saliency = SaliencyCNN(in_channels=3, feat_channels=128)(image)
# features: [B, 128, H, W]   saliency: [B, 1, H, W]
```

### Module 1b — `EdgeGradientSampler`

Non-learnable alternative: computes per-channel Sobel gradients, combines them into a gradient magnitude map, squares and normalizes it into a probability distribution. Concentrates Gaussians on edges and detail-rich regions without any trainable saliency parameters. Follows the `_compute_gmap()` method from [Image-GS](https://github.com/).

### Module 2 — `DifferentiablePointSampler`

Samples *N* Gaussian center positions from the saliency map using `torch.multinomial`. Converts flat indices to normalized `[0, 1]` coordinates compatible with the gsplat CUDA kernel.

```python
xy = DifferentiablePointSampler(num_gaussians=2000)(saliency)
# xy: [B, 2000, 2]
```

### Module 3 — `GaussianParameterHead` (MLP)

Bilinearly samples the feature map at each Gaussian center via `F.grid_sample`, then feeds through a 3-layer MLP to predict 6 parameters per Gaussian:

| Parameter | Channels | Activation | Constraint |
|-----------|----------|------------|------------|
| Scale | 2 | `softplus + min_offset` | Always positive |
| Rotation | 1 | None | Unconstrained (radians) |
| Color | 3 | `sigmoid` | `[0, 1]` RGB |

### Module 3b — `DirectParameterHead`

Image-GS style direct optimization: creates raw `nn.Parameter` tensors for scale, rotation, and color. Uses inverse-scale parameterization and a dedicated Adam optimizer with per-parameter-group learning rates.

### Module 3c — `IndependentParameterHead`

Three independent neural networks — one per parameter group (scale, rotation, color). Each has its own gradient pathway and representation capacity.

### Module 4 — `GaussianPruner`

Removes low-contribution Gaussians before rendering:
- **Color magnitude** — discard if RGB L2 norm ≈ 0
- **Scale health** — discard degenerate (near-zero or exploded) scales
- **In-bounds check** — discard if center drifted outside `[0, 1]`
- **Learned opacity** *(optional)* — small MLP predicts importance score; uses straight-through hard threshold for training

### Module 4.5 — `DeformablePointTransformerHead`

A deformable point-to-image cross-attention mechanism. Each Gaussian generates a query from 2D sine-cosine positional embeddings, predicts *K* spatial offsets, samples *K* features from the saliency map, and aggregates them with learned attention weights. Uses a Transformer FFN with LayerNorm residual connections.

### Module 5 — `GaussianPreprocessor` (Full Pipeline)

Composes all modules into a single `nn.Module` with configurable saliency mode, parameter head, pruning, and output blending:

```python
preprocessor = GaussianPreprocessor(
    img_h=640, img_w=640,
    num_gaussians=2000,
    saliency_mode="cnn",          # or "edge"
    param_mode="mlp",             # or "independent", "direct", "deformable_transformer"
    blend=False, add=False, pure=True,
    pruning=True,
    use_learned_opacity=True,
).cuda()

rendered = preprocessor(images)   # [B, 3, 640, 640]
```

### Renderer — `GaussianRenderer`

Differentiable batched 2D Gaussian splatting renderer wrapping the [gsplat](https://github.com/nerfstudio-project/gsplat) CUDA kernels. Accepts packed `[B, N, 8]` parameters and renders `[B, 3, H, W]` images. Supports tile-based (fast) and brute-force (no-tiles) rasterization paths.

---

## ⚙️ Configuration

### YOLO-World Integration

The preprocessor is registered as a standard Ultralytics module and can be inserted as the **first layer** of any YOLO config:

```yaml
# yolov8-worldv2-gs.yaml
backbone:
  # GaussianPreprocessor: [img_h, img_w, num_gaussians, feat_channels, ...]
  - [-1, 1, GaussianPreprocessor, [640, 640, 2000, 32, [16, 32], 192, True, 1.0, 0.5,
     'cnn', False, False, True, False, False, 0.5, 'deformable_transformer', False]]

  # Standard YOLO backbone follows (indices shifted by +1)
  - [-1, 1, Conv, [64, 3, 2]]    # P1/2
  - [-1, 1, Conv, [128, 3, 2]]   # P2/4
  # ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img_h` | `int` | `640` | Input / output image height |
| `img_w` | `int` | `640` | Input / output image width |
| `num_gaussians` | `int` | `2000` | Number of Gaussians per image |
| `feat_channels` | `int` | `128` | Feature map channels |
| `hidden_channels` | `tuple` | `(32, 64)` | CNN hidden stage channels |
| `hidden_dim` | `int` | `64` | MLP hidden dimension |
| `use_tiled` | `bool` | `True` | Tile-based rasterization (faster) |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `min_scale` | `float` | `0.5` | Minimum Gaussian scale |
| `saliency_mode` | `str` | `"cnn"` | `"cnn"` or `"edge"` |
| `blend` | `bool` | `False` | Learnable alpha blend with input |
| `add` | `bool` | `False` | Learnable additive composition |
| `pure` | `bool` | `False` | Raw additive (rendered + input) |
| `obscure_mode` | `bool` | `False` | Occlusion/obscuring mode |
| `param_mode` | `str` | `"mlp"` | `"mlp"`, `"direct"`, `"independent"`, `"deformable_transformer"` |
| `pruning` | `bool` | `True` | Enable Gaussian pruning |
| `use_learned_opacity` | `bool` | `True` | Learned opacity in pruner |

### Output Modes

| Mode | Formula | Learnable α |
|------|---------|-------------|
| Default | `rendered` | No |
| `blend=True` | `α · rendered + (1-α) · input` | Yes |
| `add=True` | `α · rendered + input` | Yes |
| `pure=True` | `rendered + input` | No |
| `obscure_mode=True` | Learned color subtracted from input | Yes (color) |

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA support
- NVIDIA GPU (required for gsplat CUDA kernels)

### Installation

```bash
# 1. Clone this repository
git clone https://github.com/<username>/gaussian-preprocessor.git
cd gaussian-preprocessor

# 2. Install gsplat CUDA kernels
pip install -e image-gs-main/gsplat --no-build-isolation

# 3. Install Ultralytics (modified fork with GaussianPreprocessor)
pip install -e ultralytics-main
```

### Usage — Standalone

```python
import torch
from ultralytics.nn.modules.gaussian_preprocessor import GaussianPreprocessor

# Create preprocessor
preprocessor = GaussianPreprocessor(
    img_h=640, img_w=640,
    num_gaussians=2000,
    saliency_mode="cnn",
    param_mode="deformable_transformer",
    pure=True,
    pruning=True,
).cuda()

# Forward pass
images = torch.rand(4, 3, 640, 640, device="cuda")
rendered = preprocessor(images)  # [4, 3, 640, 640]

# Fully differentiable — gradients flow back
loss = rendered.sum()
loss.backward()
```

### Usage — YOLO-World Integration

```python
from ultralytics import YOLO

# Load the Gaussian-augmented YOLO-World model
model = YOLO("ultralytics-main/ultralytics/cfg/models/v8/yolov8-worldv2-gs.yaml")

# Train end-to-end on your dataset
model.train(data="your_dataset.yaml", epochs=100, imgsz=640)

# The preprocessor learns to reconstruct images that maximize detection AP
```

### Running Tests

```bash
# Requires a CUDA GPU (run on Colab A100/L4)
python test_full_pipeline.py
```

Tests verify:
- ✅ Standalone forward pass (pure mode + blend mode)
- ✅ Gradient flow back to input
- ✅ YOLO-World model parsing and integration
- ✅ Performance benchmarking

---

## 📂 Project Structure

```
gaussian-preprocessor/
├── ultralytics-main/
│   └── ultralytics/
│       ├── cfg/models/v8/
│       │   └── yolov8-worldv2-gs.yaml    # YOLO config with preprocessor
│       └── nn/modules/
│           ├── gaussian_preprocessor.py   # Core pipeline (1075 lines)
│           ├── gaussian_renderer.py       # gsplat CUDA renderer wrapper
│           ├── __init__.py                # Module exports
│           └── ...
├── image-gs-main/
│   └── gsplat/                            # CUDA kernels (2D Gaussian splatting)
├── test_full_pipeline.py                  # Integration tests
├── test_gaussian_renderer.py              # Renderer unit tests
└── README.md
```

---

## 🔬 Technical Details

### Differentiability

The pipeline is end-to-end differentiable with one exception: `torch.multinomial` sampling in the Point Sampler is non-differentiable. However, the downstream `F.grid_sample` in the Parameter Head makes feature extraction at sampled positions differentiable, and the gsplat CUDA renderer provides analytic gradients for all Gaussian parameters (position, scale, rotation, color).

### Renderer

The renderer uses a **sum-based** rasterization model (no explicit opacity/alpha compositing). Every Gaussian contributes additively to the rendered image. Two paths are available:

| Path | Method | Best For |
|------|--------|----------|
| **Tiled** | Tile-based rasterization (16×16 blocks) | Large N (> 1000), production |
| **No-tiles** | Brute-force per-pixel evaluation | Small N, debugging |

### Memory & Performance

| Config | GPU | Batch | N | Resolution | Time |
|--------|-----|-------|---|------------|------|
| Tiled + MLP | A100 | 4 | 2000 | 640×640 | ~15 ms |
| Tiled + Transformer | A100 | 4 | 2000 | 640×640 | ~22 ms |

*Benchmarks from `test_full_pipeline.py`. Actual timings depend on GPU and driver version.*

---

## 📚 References

- **Image-GS**: 2D Gaussian Splatting for image reconstruction — basis for the renderer and edge saliency
- **YOLO-World**: Open-vocabulary object detection — the downstream detection model
- **gsplat**: High-performance CUDA kernels for Gaussian splatting ([nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat))
- **Deformable DETR**: Inspiration for the deformable point-to-image cross-attention head ([fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR))

---

## 📄 License

This project is licensed under the [AGPL-3.0 License](LICENSE), consistent with the Ultralytics codebase.

---

<p align="center">
  <sub>Built with ❤️ as a research project exploring task-oriented neural rendering for object detection.</sub>
</p>
