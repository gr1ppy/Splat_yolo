# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Differentiable Gaussian Preprocessor pipeline for YOLO-World.

This module implements the full pipeline from task_overview:
    Input Image [B, 3, H, W]
      → SaliencyCNN (features + importance map)
      → DifferentiablePointSampler (multinomial → [B, N, 2] normalized coords)
      → GaussianParameterHead (grid_sample features + MLP → scale/rot/color)
      → GaussianRenderer (render [B, 3, H, W])
      → YOLO-World backbone

The entire pipeline is end-to-end differentiable (except the multinomial
sampling step, which uses straight-through estimation). YOLO-World's detection
loss backpropagates through the renderer, teaching the CNN/heads to "draw"
images that maximize detection accuracy.

Two saliency modes are available for Gaussian center placement:
  - ``'cnn'`` (default): Learned SaliencyCNN — end-to-end differentiable.
  - ``'edge'``: Sobel edge-gradient map (from image-gs-main) — no learnable
    params, concentrates Gaussians on edges/detail regions.

Requires: gsplat CUDA package for the renderer.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv
from .gaussian_renderer import GaussianRenderer


# ---------------------------------------------------------------------------
# Module 1: SaliencyCNN — Feature Extraction + Importance Map
# ---------------------------------------------------------------------------

class SaliencyCNN(nn.Module):
    """Lightweight CNN that extracts features and a saliency (importance) map.

    Uses depthwise-separable 1×1 convolutions (DWConv) to keep the full
    spatial resolution of the input, giving the Gaussian point sampler
    pixel-level precision.  All layers use ultralytics Conv / DWConv
    wrappers so they benefit from Conv-BN fusing during export/inference.

    Args:
        in_channels (int): Input image channels (default 3 for RGB).
        feat_channels (int): Feature map channels at the final stage.
        hidden_channels (tuple[int, ...]): Channels for intermediate stages.

    Returns (forward):
        features: [B, feat_channels, H, W]  — full-resolution features.
        saliency: [B, 1, H, W]              — soft probability map for sampling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 128,
        hidden_channels: tuple[int, ...] = (32, 64),
    ):
        super().__init__()
        c1, c2 = hidden_channels

        # Encoder: 1×1 DWConv blocks — full resolution preserved
        self.enc1 = DWConv(in_channels, c1, 1, 1)     # [B, c1, H, W]
        self.enc2 = DWConv(c1, c2, 1, 1)               # [B, c2, H, W]
        self.enc3 = DWConv(c2, feat_channels, 1, 1)     # [B, feat_channels, H, W]

        # Saliency branch: 1×1 conv → sigmoid (act=False since we apply sigmoid)
        self.sal_conv = Conv(feat_channels, 1, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features and saliency map at full resolution.

        Args:
            x: Input image [B, 3, H, W].

        Returns:
            features: [B, feat_channels, H, W]
            saliency: [B, 1, H, W]
        """
        # Encode — spatial dims unchanged
        f = self.enc1(x)    # [B, c1, H, W]
        f = self.enc2(f)    # [B, c2, H, W]
        f = self.enc3(f)    # [B, feat_channels, H, W]

        # Saliency: project to 1 channel, sigmoid
        sal = torch.sigmoid(self.sal_conv(f))  # [B, 1, H, W]

        return f, sal


# ---------------------------------------------------------------------------
# Module 1b: EdgeGradientSampler — Sobel Edge Detection (image-gs-main style)
# ---------------------------------------------------------------------------

class EdgeGradientSampler(nn.Module):
    """Non-learnable saliency via Sobel edge detection, matching image-gs-main.

    Computes per-channel Sobel gradients (X and Y), combines them into a
    gradient magnitude map, squares it, and normalises to a probability
    distribution.  This concentrates Gaussian centres on edges and detail-
    rich regions without any learnable parameters.

    For feature extraction, a lightweight 1\u00d71 conv encoder is still needed
    (used by the parameter head).  Set ``return_features=True`` to also
    produce a feature map alongside the saliency, enabling a drop-in
    replacement for ``SaliencyCNN``.

    This follows ``GaussianSplatting2D._compute_gmap()`` from image-gs-main
    which uses ``scipy.ndimage.sobel`` on each channel, then computes
    ``np.hypot(gy, gx)`` and squares it for the sampling distribution.

    Args:
        in_channels (int): Input image channels (default 3).
        feat_channels (int): Feature channels to produce (for param heads).
        hidden_channels (tuple[int, ...]): Hidden channels for the feature
            encoder (same as SaliencyCNN, for API compatibility).
        power (float): Exponent applied to gradient magnitude before
            normalisation.  Default 2.0 matches image-gs-main.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 128,
        hidden_channels: tuple[int, ...] = (32, 64),
        power: float = 2.0,
    ):
        super().__init__()
        self.power = power

        # Sobel kernels (3\u00d73) — registered as buffers (not learnable)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        # Repeat for each input channel (depthwise convolution)
        self.register_buffer(
            "sobel_x", sobel_x.repeat(in_channels, 1, 1, 1)
        )  # [C, 1, 3, 3]
        self.register_buffer(
            "sobel_y", sobel_y.repeat(in_channels, 1, 1, 1)
        )  # [C, 1, 3, 3]
        self.in_channels = in_channels

        # Lightweight feature encoder — same architecture as SaliencyCNN's
        # encoder but the saliency branch is replaced by edge detection
        c1, c2 = hidden_channels
        self.enc1 = DWConv(in_channels, c1, 1, 1)
        self.enc2 = DWConv(c1, c2, 1, 1)
        self.enc3 = DWConv(c2, feat_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Sobel edge saliency and extract features.

        Args:
            x: Input image [B, 3, H, W] in [0, 1].

        Returns:
            features: [B, feat_channels, H, W]
            saliency: [B, 1, H, W] edge-based probability map.
        """
        # ---- Features (learnable, for parameter heads) ----
        f = self.enc1(x)
        f = self.enc2(f)
        f = self.enc3(f)

        # ---- Sobel edge saliency (non-learnable) ----
        # Depthwise Sobel convolution per channel
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.in_channels)  # [B, C, H, W]
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.in_channels)  # [B, C, H, W]

        # Gradient magnitude per channel, then L2 norm across channels
        g_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)  # [B, C, H, W]
        g_norm = g_mag.norm(dim=1, keepdim=True)        # [B, 1, H, W]

        # Normalise to [0, 1] per image
        g_max = g_norm.flatten(2).max(dim=-1).values[:, :, None, None]  # [B, 1, 1, 1]
        g_norm = g_norm / (g_max + 1e-8)

        # Raise to power (sharpen) and add small epsilon to avoid all-zero maps
        sal = g_norm.pow(self.power) + 1e-8  # [B, 1, H, W]

        # Normalise to valid probability distribution
        sal = sal / sal.flatten(2).sum(dim=-1, keepdim=True).unsqueeze(-1)

        return f, sal


# ---------------------------------------------------------------------------
# Module 2: DifferentiablePointSampler — Sample N Centers
# ---------------------------------------------------------------------------

class DifferentiablePointSampler(nn.Module):
    """Sample N Gaussian center positions from a saliency map.

    Uses ``torch.multinomial`` to draw indices from the flattened probability
    map, then converts pixel indices to normalized [0, 1] coordinates that
    match the gsplat CUDA kernel convention (kernel multiplies by img dims).

    The sampling step itself is non-differentiable, but the downstream
    ``grid_sample`` in GaussianParameterHead makes the feature extraction
    at those positions differentiable, and the renderer provides gradients
    for all Gaussian parameters.

    Args:
        num_gaussians (int): Number of Gaussians to sample per image.
        temperature (float): Temperature for sharpening/softening the
            probability distribution before sampling.
    """

    def __init__(self, num_gaussians: int = 2000, temperature: float = 1.0):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.temperature = temperature

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """Sample Gaussian centers from saliency map.

        Args:
            saliency_map: [B, 1, H, W] probability map in (0, 1).

        Returns:
            xy: [B, N, 2] normalized coordinates in [0, 1].
        """
        B, _, H, W = saliency_map.shape

        # Flatten spatial dims → [B, H*W]
        probs = saliency_map.view(B, H * W)

        # Apply temperature scaling
        if self.temperature != 1.0:
            probs = probs.pow(1.0 / self.temperature)

        # Ensure valid distribution (no zeros, no NaN)
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample N indices via multinomial
        indices = torch.multinomial(probs, self.num_gaussians, replacement=True)  # [B, N]

        # Convert flat indices → (x, y) normalized coordinates
        y_idx = indices // W        # row index
        x_idx = indices % W         # col index

        # Center of pixel, normalized to [0, 1]
        x_norm = (x_idx.float() + 0.5) / W
        y_norm = (y_idx.float() + 0.5) / H

        xy = torch.stack([x_norm, y_norm], dim=-1)  # [B, N, 2]
        return xy


# ---------------------------------------------------------------------------
# Module 3: GaussianParameterHead — Predict Per-Gaussian Attributes (MLP)
# ---------------------------------------------------------------------------

class GaussianParameterHead(nn.Module):
    """Predict Gaussian parameters (scale, rotation, color) at sampled centers.

    Uses ``F.grid_sample`` to bilinearly sample the feature map at each
    Gaussian center position, then feeds through an MLP to predict the
    remaining 6 parameters:
        - scale [2]: controlled by softplus + min offset (always positive)
        - rotation [1]: unconstrained (radians)
        - color [3]: sigmoid → [0, 1] RGB

    The grid_sample operation is fully differentiable, so gradients flow
    from the renderer back through the feature extractor.

    Args:
        feat_channels (int): Number of channels in the input feature map.
        hidden_dim (int): Hidden dimension of the prediction MLP.
        output_dim (int): Output params per Gaussian (default 6: 2+1+3).
        min_scale (float): Minimum scale value to prevent degenerate Gaussians.
    """

    def __init__(
        self,
        feat_channels: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 6,
        min_scale: float = 0.5,
    ):
        super().__init__()
        self.min_scale = min_scale
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(feat_channels, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Gaussian parameters at sampled positions.

        Args:
            features: Feature map [B, C, H', W'].
            xy: Normalized Gaussian center positions [B, N, 2] in [0, 1].

        Returns:
            params: [B, N, 6] — [scale(2), rotation(1), color(3)].
        """
        # grid_sample expects grid in [-1, 1] with (x, y) ordering
        grid = xy * 2.0 - 1.0                     # [B, N, 2] in [-1, 1]
        grid = grid.unsqueeze(1)                   # [B, 1, N, 2]

        # Bilinear sample features at Gaussian centers
        sampled = F.grid_sample(
            features, grid, mode="bilinear", padding_mode="border", align_corners=False
        )  # [B, C, 1, N]
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # [B, N, C]

        # MLP → raw parameters
        raw = self.mlp(sampled)  # [B, N, 6]

        # Apply activation functions
        scale = F.softplus(raw[:, :, 0:2]) + self.min_scale  # positive, min 0.5
        rot = raw[:, :, 2:3]                                  # unconstrained (radians)
        color = torch.sigmoid(raw[:, :, 3:6])                # [0, 1] RGB

        return torch.cat([scale, rot, color], dim=-1)         # [B, N, 6]


# ---------------------------------------------------------------------------
# Module 3b: DirectParameterHead — Image-GS Style Direct Optimization
# ---------------------------------------------------------------------------

class DirectParameterHead(nn.Module):
    """Directly optimizable Gaussian parameters, following image-gs-main.

    Instead of predicting parameters via an MLP, this module creates raw
    ``nn.Parameter`` tensors for scale, rotation, and color — exactly as
    image-gs-main does.  Parameters are initialized from the input image:

        - **scale** → constant ``init_scale`` (in pixels), uses inverse-scale
          parameterisation (stored as ``1/scale``, rendered as ``1/param``).
        - **rotation** → zero (axis-aligned).
        - **color** → bilinearly sampled from the input image at the sampled
          Gaussian center positions (``grid_sample`` on the raw image).

    A dedicated Adam optimizer with per-parameter-group learning rates
    updates these parameters each forward pass during training, mirroring
    the image-gs-main optimisation loop.

    Args:
        num_gaussians (int): Number of Gaussians per image.
        init_scale (float): Initial Gaussian scale in pixels.
        pos_lr (float): Adam LR for xy positions.
        scale_lr (float): Adam LR for scales.
        rot_lr (float): Adam LR for rotation.
        feat_lr (float): Adam LR for colour features.
    """

    def __init__(
        self,
        num_gaussians: int = 2000,
        init_scale: float = 5.0,
        pos_lr: float = 5e-4,
        scale_lr: float = 2e-3,
        rot_lr: float = 2e-3,
        feat_lr: float = 5e-3,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.init_scale = init_scale

        # Raw parameters — nn.Parameters so they live on the correct device
        # and are saved/loaded with the model checkpoint.
        # Scale uses inverse parameterisation: stored as 1/scale
        self.raw_scale = nn.Parameter(
            torch.full((num_gaussians, 2), 1.0 / init_scale), requires_grad=True
        )
        # Rotation: initialised to 0 (axis-aligned)
        self.raw_rot = nn.Parameter(
            torch.zeros(num_gaussians, 1), requires_grad=True
        )
        # Color: will be re-initialised from each input image in forward()
        self.raw_feat = nn.Parameter(
            torch.rand(num_gaussians, 3), requires_grad=True
        )

        # Store LRs for the dedicated Adam optimiser (created lazily)
        self._lr = dict(pos_lr=pos_lr, scale_lr=scale_lr, rot_lr=rot_lr, feat_lr=feat_lr)
        self._optimizer = None  # built on first forward

    def _build_optimizer(self, xy: torch.Tensor):
        """Create the dedicated Adam optimiser for direct params + xy."""
        self._optimizer = torch.optim.Adam([
            {"params": [xy],            "lr": self._lr["pos_lr"]},
            {"params": [self.raw_scale], "lr": self._lr["scale_lr"]},
            {"params": [self.raw_rot],   "lr": self._lr["rot_lr"]},
            {"params": [self.raw_feat],  "lr": self._lr["feat_lr"]},
        ])

    def _init_color_from_image(self, image: torch.Tensor, xy: torch.Tensor):
        """Sample pixel colours from the input image at Gaussian centres.

        Follows image-gs-main's ``_get_target_features``.

        Args:
            image: Input image [B, 3, H, W] in [0, 1].
            xy: Normalised positions [B, N, 2] in [0, 1].
        """
        with torch.no_grad():
            grid = (xy * 2.0 - 1.0).unsqueeze(1)  # [B, 1, N, 2]
            # Sample from first image in batch (all share the same params)
            sampled = F.grid_sample(
                image[:1], grid[:1], mode="bilinear",
                padding_mode="border", align_corners=False,
            )  # [1, 3, 1, N]
            colors = sampled[0, :, 0, :].permute(1, 0)  # [N, 3]
            self.raw_feat.data.copy_(colors)

    def forward(
        self,
        features: torch.Tensor,  # unused but kept for API compat
        xy: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the 6 direct parameters, optionally initialising color.

        Args:
            features: Ignored (API compatibility with MLP head).
            xy: Normalised Gaussian centers [B, N, 2] in [0, 1].
            image: Original input image [B, 3, H, W] for colour init.

        Returns:
            params: [B, N, 6] — [scale(2), rotation(1), color(3)].
        """
        B = xy.shape[0]

        # Initialise colour from the input image on first real call
        if image is not None and self.training:
            self._init_color_from_image(image, xy)

        # Scale: inverse parameterisation → actual scale = 1 / raw_scale
        scale = 1.0 / self.raw_scale.clamp(min=1e-4)       # [N, 2]
        rot = self.raw_rot                                   # [N, 1]
        color = torch.sigmoid(self.raw_feat)                 # [N, 3]

        # Concatenate and expand to batch
        params = torch.cat([scale, rot, color], dim=-1)      # [N, 6]
        return params.unsqueeze(0).expand(B, -1, -1)         # [B, N, 6]

    def step(self, xy: torch.Tensor):
        """Run one Adam update step on the direct parameters.

        Should be called after ``loss.backward()`` in the training loop.

        Args:
            xy: The position tensor (must be the same object passed to
                forward so its ``.grad`` is populated).
        """
        if self._optimizer is None:
            self._build_optimizer(xy)
        self._optimizer.step()
        self._optimizer.zero_grad()



# ---------------------------------------------------------------------------
# Module 3c: IndependentParameterHead — Per-Parameter Neural Networks
# ---------------------------------------------------------------------------

class IndependentParameterHead(nn.Module):
    """Three independent neural networks, one per parameter group.

    Each parameter group (scale, rotation, color) has its own small network
    that takes bilinearly sampled features at the Gaussian centres and
    independently predicts its outputs.  This gives each parameter type its
    own learned representation capacity and gradient pathway.

    Architecture per network:
        features [B, N, C]  →  Linear(C, hidden_dim)  →  SiLU
                            →  Linear(hidden_dim, out_dim)  →  activation

    Activations:
        - **scale**: softplus + min_scale  (always positive)
        - **rotation**: none  (unconstrained radians)
        - **color**: sigmoid  → [0, 1] RGB

    Args:
        feat_channels (int): Number of channels in the input feature map.
        hidden_dim (int): Hidden dimension *per* sub-network.
        min_scale (float): Minimum scale value to prevent degenerate Gaussians.
    """

    def __init__(
        self,
        feat_channels: int = 128,
        hidden_dim: int = 64,
        min_scale: float = 0.5,
    ):
        super().__init__()
        self.min_scale = min_scale

        # Independent sub-networks
        self.scale_net = nn.Sequential(
            nn.Linear(feat_channels, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2),            # scale_x, scale_y
        )
        self.rot_net = nn.Sequential(
            nn.Linear(feat_channels, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),            # rotation angle
        )
        self.color_net = nn.Sequential(
            nn.Linear(feat_channels, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),            # R, G, B
        )

    def forward(
        self,
        features: torch.Tensor,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Gaussian parameters via independent sub-networks.

        Args:
            features: Feature map [B, C, H', W'].
            xy: Normalized Gaussian center positions [B, N, 2] in [0, 1].

        Returns:
            params: [B, N, 6] — [scale(2), rotation(1), color(3)].
        """
        # Bilinear sample features at Gaussian centres
        grid = (xy * 2.0 - 1.0).unsqueeze(1)      # [B, 1, N, 2]
        sampled = F.grid_sample(
            features, grid, mode="bilinear", padding_mode="border", align_corners=False
        )  # [B, C, 1, N]
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # [B, N, C]

        # Independent predictions
        scale = F.softplus(self.scale_net(sampled)) + self.min_scale  # [B, N, 2]
        rot = self.rot_net(sampled)                                    # [B, N, 1]
        color = torch.sigmoid(self.color_net(sampled))                # [B, N, 3]

        return torch.cat([scale, rot, color], dim=-1)                 # [B, N, 6]


# ---------------------------------------------------------------------------
# Module 4: GaussianPruner — Remove Low-Contribution Gaussians
# ---------------------------------------------------------------------------

class GaussianPruner(nn.Module):
    """Prune (discard) low-contribution Gaussians before rendering.

    Image-gs-main uses sum-based rasterization with NO explicit opacity
    channel.  Instead, every Gaussian always contributes additively.  To
    prune "bad" Gaussians we must define proxy quality criteria:

    1. **Color magnitude**: Gaussians whose RGB contribution is near-zero
       add nothing to the rendered image — discard them.
    2. **Scale health**: Gaussians with degenerate (near-zero or exploded)
       scales produce numerical issues — discard them.
    3. **In-bounds check**: Gaussians whose centres have drifted outside
       [0, 1] normalised image coordinates — discard them.
    4. **Learned opacity** (optional): A small network predicts a per-
       Gaussian importance score in [0, 1].  During training, a straight-
       through hard threshold zeros out unimportant Gaussians while still
       back-propagating gradients through the sigmoid.  During inference
       the hard mask is applied directly.

    The pruner operates on the concatenated ``[B, N, 8]`` parameter tensor
    (xy + scale + rot + color) and returns a pruned/masked version.  The
    number of output Gaussians may be less than or equal to N.

    Args:
        feat_channels (int): CNN feature channels (for learned opacity head).
        color_threshold (float): Min L2 norm of RGB to keep a Gaussian.
        scale_min (float): Min scale value (below → pruned).
        scale_max (float): Max scale value (above → pruned).
        opacity_threshold (float): Hard threshold for learned opacity.
        use_learned_opacity (bool): If True, add a small network that
            predicts per-Gaussian importance ("opacity").
        hidden_dim (int): Hidden dimension for the opacity MLP.
    """

    def __init__(
        self,
        feat_channels: int = 128,
        color_threshold: float = 0.01,
        scale_min: float = 0.1,
        scale_max: float = 200.0,
        opacity_threshold: float = 0.1,
        use_learned_opacity: bool = True,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.color_threshold = color_threshold
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.opacity_threshold = opacity_threshold
        self.use_learned_opacity = use_learned_opacity

        if use_learned_opacity:
            # Small MLP: sampled features → scalar importance score
            self.opacity_head = nn.Sequential(
                nn.Linear(feat_channels, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

    def _hard_prune_mask(self, params: torch.Tensor) -> torch.Tensor:
        """Compute a boolean keep-mask from deterministic quality criteria.

        Args:
            params: [B, N, 8] — [xy(2), scale(2), rot(1), color(3)].

        Returns:
            mask: [B, N] boolean tensor, True = keep.
        """
        xy = params[:, :, 0:2]       # [B, N, 2]
        scale = params[:, :, 2:4]    # [B, N, 2]
        color = params[:, :, 5:8]    # [B, N, 3]

        # 1. Color contribution: keep if L2 norm of RGB > threshold
        color_norm = color.norm(dim=-1)  # [B, N]
        color_ok = color_norm > self.color_threshold

        # 2. Scale health: keep if both scale dims are in [min, max]
        scale_ok = (
            (scale[:, :, 0] > self.scale_min) &
            (scale[:, :, 1] > self.scale_min) &
            (scale[:, :, 0] < self.scale_max) &
            (scale[:, :, 1] < self.scale_max)
        )  # [B, N]

        # 3. In-bounds: centres must be in [0, 1]
        bounds_ok = (
            (xy[:, :, 0] >= 0) & (xy[:, :, 0] <= 1) &
            (xy[:, :, 1] >= 0) & (xy[:, :, 1] <= 1)
        )  # [B, N]

        return color_ok & scale_ok & bounds_ok  # [B, N]

    def _learned_opacity_mask(
        self, features: torch.Tensor, xy: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-Gaussian importance and apply hard threshold.

        Uses a straight-through estimator so gradients flow through the
        hard threshold during training.  At eval time the hard mask is
        applied directly.

        Args:
            features: Feature map [B, C, H, W].
            xy: Normalised positions [B, N, 2].

        Returns:
            opacity_weight: [B, N, 1] soft weight (0 or 1 in forward,
                gradient-bearing in backward).
        """
        # Bilinear sample features at Gaussian centres
        grid = (xy * 2.0 - 1.0).unsqueeze(1)  # [B, 1, N, 2]
        sampled = F.grid_sample(
            features, grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        ).squeeze(2).permute(0, 2, 1)  # [B, N, C]

        # Predict importance logit → sigmoid → [0, 1]
        opacity = torch.sigmoid(self.opacity_head(sampled))  # [B, N, 1]

        # Straight-through hard threshold
        hard = (opacity > self.opacity_threshold).float()
        # STE: forward uses hard mask, backward uses sigmoid gradient
        opacity_weight = hard - opacity.detach() + opacity  # [B, N, 1]

        return opacity_weight

    def forward(
        self,
        params: torch.Tensor,
        features: torch.Tensor | None = None,
        xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply pruning to Gaussian parameters.

        Args:
            params: [B, N, 8] packed Gaussian parameters.
            features: [B, C, H, W] CNN features (needed for learned opacity).
            xy: [B, N, 2] normalised positions (needed for learned opacity).

        Returns:
            Pruned/masked params: [B, N, 8] with pruned Gaussians zeroed out
            (color set to 0 so they contribute nothing in sum-based rendering).
        """
        # Deterministic quality mask
        keep_mask = self._hard_prune_mask(params)  # [B, N]

        # Zero out pruned Gaussians (set color channels to 0)
        # This is equivalent to removing them from sum-based rendering
        mask_3d = keep_mask.unsqueeze(-1).float()  # [B, N, 1]
        params = params * mask_3d  # zeros out ALL channels for pruned Gaussians

        # Apply learned opacity weighting
        if self.use_learned_opacity and features is not None and xy is not None:
            opacity_weight = self._learned_opacity_mask(features, xy)  # [B, N, 1]
            # Multiply color channels by opacity weight
            color = params[:, :, 5:8] * opacity_weight
            params = torch.cat([params[:, :, :5], color], dim=-1)

        return params


# ---------------------------------------------------------------------------
# Module 4.5: Deformable Point Transformer Head
# ---------------------------------------------------------------------------

class DeformablePointTransformerHead(nn.Module):
    """Predicts Gaussian parameters using a Point-to-Image Cross-Attention mechanism.
    
    Instead of relying on local features at the precise (x,y) location, each
    Gaussian generates a query that predicts K offsets. It samples
    K features from the Saliency map via grid_sample, aggregates them with 
    learned attention weights, and uses a Transformer FFN to predict parameters.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128, K: int = 4):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        
        self.value_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Predict K spatial offsets per point (K * 2)
        self.offset_net = nn.Linear(hidden_dim, K * 2)
        nn.init.constant_(self.offset_net.weight, 0)
        nn.init.constant_(self.offset_net.bias, 0)
        
        # Predict K attention weights
        self.weight_net = nn.Linear(hidden_dim, K)
        nn.init.constant_(self.weight_net.weight, 0)
        nn.init.constant_(self.weight_net.bias, 0)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Final projection to 6 parameters (scale:2, rot:1, color:3)
        self.final_proj = nn.Linear(hidden_dim, 6)
        
    def get_2d_sincos_pos_embed(self, points: torch.Tensor) -> torch.Tensor:
        """Generate 2D sine-cosine positional embedding for (x, y) coords.
        points shape: [B, N, 2] in range [0, 1].
        """
        half_dim = self.hidden_dim // 4
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=points.device) * -emb)
        
        emb_x = points[:, :, 0:1] * emb.view(1, 1, -1)
        emb_y = points[:, :, 1:2] * emb.view(1, 1, -1)
        
        return torch.cat([torch.sin(emb_x), torch.cos(emb_x), torch.sin(emb_y), torch.cos(emb_y)], dim=-1)

    def forward(self, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        B, N, _ = points.shape
        
        # [B, C, H, W] -> [B, D, H, W]
        v_feat = self.value_proj(features)
        
        # Compute Queries
        pos_emb = self.get_2d_sincos_pos_embed(points) # [B, N, D]
        query = self.query_proj(pos_emb) # [B, N, D]
        
        # Compute Offsets and Attention Maps
        offsets = self.offset_net(query).view(B, N, self.K, 2)
        attn_weights = F.softmax(self.weight_net(query), dim=-1) # [B, N, K]
        
        # Sample Features
        sampling_locations = points.unsqueeze(2) + offsets # [B, N, K, 2]
        grid_coords = sampling_locations * 2.0 - 1.0 # map [0,1] to [-1,1]
        
        # sampled_v: [B, D, N, K]
        sampled_v = F.grid_sample(v_feat, grid_coords, align_corners=False)
        
        # Apply Cross-Attention
        attn_weights = attn_weights.unsqueeze(1) # [B, 1, N, K]
        attended_features = (sampled_v * attn_weights).sum(dim=-1) # [B, D, N]
        attended_features = attended_features.transpose(1, 2) # [B, N, D]
        
        # ResNet + FFN structure
        out = self.norm1(query + self.out_proj(attended_features))
        out = self.norm2(out + self.ffn(out))
        
        # Final Param Prediction map to [6] vector
        pred = self.final_proj(out)
        
        # Map activation functions for Gaussian domains
        scale = torch.exp(pred[..., 0:2])
        rot = pred[..., 2:3]                              # unconstrained angle
        color = torch.sigmoid(pred[..., 3:6])
        
        # Combine parameters 
        return torch.cat([scale, rot, color], dim=-1)


# ---------------------------------------------------------------------------
# Module 5: GaussianPreprocessor — Full Pipeline Wrapper
# ---------------------------------------------------------------------------

class GaussianPreprocessor(nn.Module):
    """Complete differentiable Gaussian Splatting preprocessor for YOLO-World.

    Composes the full pipeline:
        Input Image → SaliencyCNN / EdgeGradient → PointSampler
                    → ParameterHead → [Pruner] → Renderer

    **Saliency modes** (``saliency_mode``):
        - ``'cnn'`` (default): Learned SaliencyCNN produces features and
          a differentiable saliency map.  End-to-end trainable.
        - ``'edge'``: Sobel edge detection following image-gs-main's
          ``_compute_gmap()``.  No learnable saliency params — Gaussians
          concentrate on edges/detail.  Features are still extracted by a
          lightweight encoder for the parameter head.

    **Output modes** (mutually exclusive):
        - ``blend=True``: ``α * rendered + (1-α) * input`` with learnable α.
        - ``add=True``: ``α * rendered + input`` (additive with learnable α).
        - ``pure=True``: ``rendered + input`` (raw additive, no learnable α).
        - All False (default): Output is the rendered image only.

    **Parameter head modes** (``param_mode``):
        - ``'mlp'`` (default): Shared MLP predicts all 6 params.
        - ``'direct'``: Image-GS style raw nn.Parameters + Adam optimizer.
        - ``'independent'``: Three separate networks per param group.
        - ``'deformable_transformer'``: A custom Point-to-Image Cross Attention
          ViT mechanism extracting smart features globally.

    Args:
        img_h (int): Input/output image height.
        img_w (int): Input/output image width.
        num_gaussians (int): Number of Gaussians per image.
        feat_channels (int): Feature map channels.
        hidden_channels (tuple[int, ...]): Hidden channels for CNN stages.
        hidden_dim (int): MLP hidden dimension for parameter prediction.
        use_tiled (bool): Use tiled rasterization (faster for large N).
        temperature (float): Sampling temperature for the point sampler.
        min_scale (float): Minimum Gaussian scale.
        saliency_mode (str): ``'cnn'`` or ``'edge'``.
        blend (bool): If True, blend rendered output with original input.
        add (bool): If True, additive output with learnable alpha.
        pure (bool): If True, raw additive output (rendered + input).
        blend_init (float): Initial blend/add factor.
        param_mode (str): Parameter head mode: ``'mlp'``, ``'direct'``,
            or ``'independent'``.
        direct_params (bool): Legacy flag — if True, overrides param_mode
            to ``'direct'``.
        init_scale (float): Initial Gaussian scale in pixels (direct mode).
        pos_lr (float): Adam LR for positions (direct mode).
        scale_lr (float): Adam LR for scales (direct mode).
        rot_lr (float): Adam LR for rotation (direct mode).
        feat_lr (float): Adam LR for color features (direct mode).

    Example:
        >>> preprocessor = GaussianPreprocessor(img_h=640, img_w=640).cuda()
        >>> images = torch.randn(4, 3, 640, 640, device='cuda')
        >>> rendered = preprocessor(images)  # [4, 3, 640, 640]
    """

    def __init__(
        self,
        img_h: int = 640,
        img_w: int = 640,
        num_gaussians: int = 2000,
        feat_channels: int = 128,
        hidden_channels: tuple[int, ...] = (32, 64),
        hidden_dim: int = 64,
        use_tiled: bool = True,
        temperature: float = 1.0,
        min_scale: float = 0.5,
        saliency_mode: str = "cnn",
        blend: bool = False,
        add: bool = False,
        pure: bool = False,
        obscure_mode: bool = False,
        sample_from_img: bool = False,
        blend_init: float = 0.5,
        param_mode: str = "mlp",
        pruning: bool = True,
        color_threshold: float = 0.01,
        opacity_threshold: float = 0.1,
        use_learned_opacity: bool = True,
        direct_params: bool = False,
        init_scale: float = 5.0,
        pos_lr: float = 5e-4,
        scale_lr: float = 2e-3,
        rot_lr: float = 2e-3,
        feat_lr: float = 5e-3,
    ):
        super().__init__()

        self.blend_mode = blend
        self.add_mode = add
        self.pure_mode = pure
        self.saliency_mode = saliency_mode
        self.obscure_mode = obscure_mode
        self.sample_from_img = sample_from_img
        
        head_in_channels = 3 if sample_from_img else feat_channels

        if obscure_mode:
            # Single learnable color vector shared by all Gaussians.
            # Initialized to -1.0 so that when added to the image (pure/add mode),
            # it subtracts (darkens/obscures).
            self.obscure_color = nn.Parameter(torch.full((3,), -1.0))

        # Resolve param_mode (legacy compat: direct_params=True → 'direct')
        if direct_params:
            param_mode = "direct"
        self.param_mode = param_mode

        # Saliency / feature extractor
        if saliency_mode == "edge":
            self.saliency_cnn = EdgeGradientSampler(
                in_channels=3,
                feat_channels=feat_channels,
                hidden_channels=hidden_channels,
            )
        else:  # 'cnn' (default)
            self.saliency_cnn = SaliencyCNN(
                in_channels=3,
                feat_channels=feat_channels,
                hidden_channels=hidden_channels,
            )

        self.point_sampler = DifferentiablePointSampler(
            num_gaussians=num_gaussians,
            temperature=temperature,
        )

        if param_mode == "direct":
            self.param_head = DirectParameterHead(
                num_gaussians=num_gaussians,
                init_scale=init_scale,
                pos_lr=pos_lr,
                scale_lr=scale_lr,
                rot_lr=rot_lr,
                feat_lr=feat_lr,
            )
        elif param_mode == "independent":
            self.param_head = IndependentParameterHead(
                feat_channels=head_in_channels,
                hidden_dim=hidden_dim,
                min_scale=min_scale,
            )
        elif param_mode == "deformable_transformer":
            self.param_head = DeformablePointTransformerHead(
                in_channels=head_in_channels,
                hidden_dim=hidden_dim,
                K=4
            )
        else:
            self.param_head = GaussianParameterHead(
                feat_channels=head_in_channels,
                hidden_dim=hidden_dim,
                output_dim=6,
                min_scale=min_scale,
            )

        self.renderer = GaussianRenderer(
            img_h=img_h,
            img_w=img_w,
            num_gaussians=num_gaussians,
            feat_dim=3,
            use_tiled=use_tiled,
            clamp_output=not obscure_mode,
        )

        self.pruning = pruning
        if pruning:
            self.pruner = GaussianPruner(
                feat_channels=head_in_channels,
                color_threshold=color_threshold,
                opacity_threshold=opacity_threshold,
                use_learned_opacity=use_learned_opacity,
                hidden_dim=hidden_dim,
            )

        if blend or add:
            alpha_logit = torch.log(torch.tensor(blend_init / (1.0 - blend_init)))
            self.alpha_logit = nn.Parameter(alpha_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full preprocessor forward pass.

        Args:
            x: Input images [B, 3, H, W].

        Returns:
            Rendered (or blended) images [B, 3, H, W].
        """
        B, C, H, W = x.shape

        if H < 64 or W < 64 or x.std() < 1e-6:
            return x

        input_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.float()

        try:
            with torch.cuda.amp.autocast(enabled=False):
                features, saliency = self.saliency_cnn(x.float())

                xy = self.point_sampler(saliency)

                sampling_input = x if self.sample_from_img else features

                # Predict Gaussian parameters (scale, rot, color)
                if self.param_mode == "direct":
                    params = self.param_head(sampling_input, xy, image=x)
                else:  # 'mlp', 'independent', or 'deformable_transformer' — same call signature
                    params = self.param_head(sampling_input, xy)

                if self.obscure_mode:
                    # Map the 3-channel predicted color to a 1-channel intensity
                    intensity = params[:, :, 3:6].mean(dim=-1, keepdim=True)  # [B, N, 1] (color is at 3:6 for 2D gaussians)
                    # Scale the single globally-learned obscure color by the intensity
                    color = intensity * self.obscure_color  # [B, N, 3]
                    params = torch.cat([params[:, :, :3], color], dim=-1)

                gaussian_params = torch.cat([xy, params], dim=-1)

                # Prune low-contribution Gaussians before rendering
                if self.pruning:
                    gaussian_params = self.pruner(
                        gaussian_params, features=sampling_input, xy=xy,
                    )

                rendered = self.renderer(gaussian_params, img_h=H, img_w=W)

            rendered = rendered.to(dtype=input_dtype).contiguous()
            x_orig = x.to(dtype=input_dtype)

            # Output mode: blend, add, pure, or raw rendered
            if self.blend_mode:
                alpha = torch.sigmoid(self.alpha_logit)
                out = alpha * rendered + (1.0 - alpha) * x_orig
                return out.clamp(0.0, 1.0)
            elif self.add_mode:
                alpha = torch.sigmoid(self.alpha_logit)
                out = alpha * rendered + x_orig
                return out.clamp(0.0, 1.0)
            elif self.pure_mode:
                out = rendered + x_orig
                return out.clamp(0.0, 1.0)
            else:
                return rendered.clamp(0.0, 1.0)
        except RuntimeError:
            return x.to(dtype=input_dtype)
