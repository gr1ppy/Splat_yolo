# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Differentiable 2D Gaussian Splatting renderer module.

This module wraps the Image-GS gsplat CUDA kernels to provide a batched,
differentiable 2D Gaussian renderer that can be used as a layer in the
YOLO pipeline. It accepts packed Gaussian parameters [B, N, 8] and
renders [B, 3, H, W] image tensors.

Two rendering paths are available:
  - **Tiled** (default): Uses tile-based rasterization for speed with large N.
  - **No-tiles**: Brute-force per-pixel evaluation of all Gaussians.

Both paths use the same projection step to convert (xy, scale, rot) into
conic matrices, then rasterize with weighted color accumulation.

NOTE: ``rasterize_gaussians_simple`` is NOT used because its CUDA forward
and backward kernels are commented-out stubs in the current gsplat build.

Requires: gsplat CUDA package (pip install -e image-gs-main/gsplat)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Lazy-loaded gsplat functions (CUDA package, not always available)
_gsplat_loaded = False
project_gaussians_2d_scale_rot = None
rasterize_gaussians_sum = None
rasterize_gaussians_no_tiles = None


def _ensure_gsplat():
    """Lazy-import gsplat CUDA functions on first use."""
    global _gsplat_loaded, project_gaussians_2d_scale_rot, rasterize_gaussians_sum, rasterize_gaussians_no_tiles
    if not _gsplat_loaded:
        from gsplat import (
            project_gaussians_2d_scale_rot as _proj,
            rasterize_gaussians_sum as _rast_sum,
            rasterize_gaussians_no_tiles as _rast_no_tiles,
        )
        project_gaussians_2d_scale_rot = _proj
        rasterize_gaussians_sum = _rast_sum
        rasterize_gaussians_no_tiles = _rast_no_tiles
        _gsplat_loaded = True


def render_gaussians_2d(
    xy: torch.Tensor,
    scale: torch.Tensor,
    rot: torch.Tensor,
    feat: torch.Tensor,
    img_h: int,
    img_w: int,
    use_tiled: bool = True,
    block_h: int = 16,
    block_w: int = 16,
    topk_norm: bool = False,
) -> torch.Tensor:
    """Render a single image from 2D Gaussians using gsplat CUDA kernels.

    Both paths share the same projection step that converts raw Gaussian
    parameters (position, scale, rotation) into conic matrices and radii.

    Args:
        xy: Gaussian center positions, shape [N, 2]. Normalized coordinates
            in [0, 1] range — the CUDA kernel multiplies by (W, H) internally.
        scale: Gaussian scales, shape [N, 2]. Controls ellipse semi-axes.
        rot: Gaussian rotation angles in radians, shape [N, 1].
        feat: Gaussian feature/color values, shape [N, C].
        img_h: Output image height in pixels.
        img_w: Output image width in pixels.
        use_tiled: If True, use tile-based rasterization (faster for large N).
            If False, use brute-force no-tiles rasterization.
        block_h: Tile height (only used when use_tiled=True).
        block_w: Tile width (only used when use_tiled=True).
        topk_norm: Top-K normalization (only used when use_tiled=True).

    Returns:
        Rendered image tensor of shape [C, H, W].
    """
    tile_bounds = (
        (img_w + block_w - 1) // block_w,
        (img_h + block_h - 1) // block_h,
        1,
    )

    # Lazy-load gsplat CUDA functions
    _ensure_gsplat()

    # Step 1: Project — compute conic matrices, radii, and tile hit counts
    xys_proj, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
        means2d=xy.contiguous(),
        scales2d=scale.contiguous(),
        rotation=rot.contiguous(),
        img_height=img_h,
        img_width=img_w,
        tile_bounds=tile_bounds,
    )

    feat_dim = feat.shape[-1]

    # Step 2: Rasterize
    if use_tiled:
        out_img = rasterize_gaussians_sum(
            xys=xys_proj,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=feat.contiguous(),
            img_height=img_h,
            img_width=img_w,
            BLOCK_H=block_h,
            BLOCK_W=block_w,
            topk_norm=topk_norm,
        )
    else:
        out_img = rasterize_gaussians_no_tiles(
            xys=xys_proj,
            conics=conics,
            colors=feat.contiguous(),
            img_height=img_h,
            img_width=img_w,
        )

    # Reshape: rasterizer outputs [H, W, C] → we need [C, H, W]
    out_img = out_img.view(-1, img_h, img_w, feat_dim)
    out_img = out_img.permute(0, 3, 1, 2).contiguous()
    return out_img.squeeze(dim=0)  # [C, H, W]


def batched_render_gaussians_2d(
    xy: torch.Tensor,
    scale: torch.Tensor,
    rot: torch.Tensor,
    feat: torch.Tensor,
    img_h: int,
    img_w: int,
    use_tiled: bool = True,
    block_h: int = 16,
    block_w: int = 16,
    topk_norm: bool = False,
) -> torch.Tensor:
    """Render a batch of images from batched 2D Gaussian parameters.

    Iterates over the batch dimension and calls the single-image renderer
    for each sample, then stacks the results. The gsplat CUDA kernels are
    inherently single-image, so batching is handled at the Python level.

    Args:
        xy: Batched Gaussian positions, shape [B, N, 2].
        scale: Batched Gaussian scales, shape [B, N, 2].
        rot: Batched Gaussian rotations, shape [B, N, 1].
        feat: Batched Gaussian features/colors, shape [B, N, C].
        img_h: Output image height.
        img_w: Output image width.
        use_tiled: If True, use tiled rendering path.
        block_h: Tile height for tiled rasterization.
        block_w: Tile width for tiled rasterization.
        topk_norm: Top-K normalization for tiled rasterization.

    Returns:
        Batched rendered images, shape [B, C, H, W].
    """
    B = xy.shape[0]
    rendered = []

    for b in range(B):
        img = render_gaussians_2d(
            xy=xy[b],
            scale=scale[b],
            rot=rot[b],
            feat=feat[b],
            img_h=img_h,
            img_w=img_w,
            use_tiled=use_tiled,
            block_h=block_h,
            block_w=block_w,
            topk_norm=topk_norm,
        )
        rendered.append(img)

    return torch.stack(rendered, dim=0)  # [B, C, H, W]


class GaussianRenderer(nn.Module):
    """Differentiable batched 2D Gaussian Splatting renderer.

    Takes a tensor of packed Gaussian parameters [B, N, 8] and renders
    [B, 3, H, W] image tensors using the gsplat CUDA kernels. Designed
    to be inserted into the YOLO pipeline as a differentiable preprocessor.

    The input parameter tensor has the following channel layout:
        - channels 0:2  → xy position (NORMALIZED [0, 1] coords; the CUDA
          kernel multiplies by image size internally)
        - channels 2:4  → scale (positive, controls Gaussian spread in pixels)
        - channels 4:5  → rotation angle (radians)
        - channels 5:8  → RGB color

    Two rendering modes:
        - ``use_tiled=True`` (default): Tile-based rasterization. Faster for
          large N because tiles skip non-overlapping Gaussians.
        - ``use_tiled=False``: No-tiles rasterization. Brute-force per-pixel
          evaluation; simpler but slower for large N.

    Attributes:
        img_h (int): Output image height.
        img_w (int): Output image width.
        num_gaussians (int): Expected number of Gaussians per image.
        feat_dim (int): Feature dimension (default 3 for RGB).
        use_tiled (bool): Whether to use tiled rasterization.
        block_h (int): Tile block height.
        block_w (int): Tile block width.
        topk_norm (bool): Whether to use top-K normalization.
        clamp_output (bool): Whether to clamp output to [0, 1].

    Example:
        >>> renderer = GaussianRenderer(img_h=640, img_w=640, num_gaussians=2000)
        >>> params = torch.randn(4, 2000, 8, device='cuda')  # needs valid ranges
        >>> images = renderer(params)  # [4, 3, 640, 640]
    """

    def __init__(
        self,
        img_h: int,
        img_w: int,
        num_gaussians: int,
        feat_dim: int = 3,
        use_tiled: bool = True,
        block_h: int = 16,
        block_w: int = 16,
        topk_norm: bool = False,
        clamp_output: bool = True,
    ):
        """Initialize the GaussianRenderer.

        Args:
            img_h: Output image height in pixels.
            img_w: Output image width in pixels.
            num_gaussians: Number of Gaussians per image (N).
            feat_dim: Feature/color channels (default 3 for RGB).
            use_tiled: Use tiled rasterization (faster for large N).
            block_h: Tile block height (must match CUDA kernel, default 16).
            block_w: Tile block width (must match CUDA kernel, default 16).
            topk_norm: Use top-K normalization in the rasterizer.
            clamp_output: Clamp rendered output to [0, 1] range.
        """
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.num_gaussians = num_gaussians
        self.feat_dim = feat_dim
        self.use_tiled = use_tiled
        self.block_h = block_h
        self.block_w = block_w
        self.topk_norm = topk_norm
        self.clamp_output = clamp_output

    def forward(self, gaussian_params: torch.Tensor, img_h: int = None, img_w: int = None) -> torch.Tensor:
        """Render images from packed Gaussian parameters.

        Args:
            gaussian_params: Packed Gaussian parameters tensor of shape
                [B, N, 8] where the 8 channels are...
            img_h: Optional target height. Defaults to self.img_h.
            img_w: Optional target width. Defaults to self.img_w.
        """
        img_h = img_h or self.img_h
        img_w = img_w or self.img_w
        B, N, P = gaussian_params.shape
        assert P == self.feat_dim + 5, (
            f"Expected {self.feat_dim + 5} parameters per Gaussian "
            f"(2 xy + 2 scale + 1 rot + {self.feat_dim} feat), got {P}"
        )

        # Unpack parameters
        xy = gaussian_params[:, :, 0:2]                        # [B, N, 2]
        scale = gaussian_params[:, :, 2:4]                     # [B, N, 2]
        rot = gaussian_params[:, :, 4:5]                       # [B, N, 1]
        feat = gaussian_params[:, :, 5 : 5 + self.feat_dim]   # [B, N, C]

        # Render batch
        images = batched_render_gaussians_2d(
            xy=xy,
            scale=scale,
            rot=rot,
            feat=feat,
            img_h=img_h,
            img_w=img_w,
            use_tiled=self.use_tiled,
            block_h=self.block_h,
            block_w=self.block_w,
            topk_norm=self.topk_norm,
        )

        if self.clamp_output:
            images = torch.clamp(images, 0.0, 1.0)

        return images
