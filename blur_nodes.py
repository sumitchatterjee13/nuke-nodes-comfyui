"""
Blur and filtering nodes that replicate Nuke's blur functionality
"""

import logging
import math

import torch
import torch.nn.functional as F

from .utils import (
    NukeNodeBase,
    apply_mask_mix,
    ensure_batch_dim,
    mask_to_bhw1,
    normalize_tensor,
)

logger = logging.getLogger(__name__)

# Gaussian truncation-radius multiplier per quality (gaussian kernels only)
QUALITY_MULTIPLIERS = {"low": 1.0, "medium": 2.0, "high": 3.0}
# Defocus "disk": number of stacked gaussian passes per quality
DISK_PASSES = {"low": 1, "medium": 2, "high": 3}
# Defocus "hexagon": coverage sub-samples per pixel axis per quality
HEXAGON_SUPERSAMPLES = {"low": 1, "medium": 2, "high": 4}


def _pad_axis(tensor, pad, axis, edge_mode):
    """Pad an NCHW tensor by ``pad`` pixels on both sides of one axis.

    ``axis`` is ``"x"`` (width) or ``"y"`` (height).

    ``edge_mode``:
      - ``"zero"``: constant zero padding (Nuke "crop" semantics - outside
        the format is black, so blurs fade to black at the edges).
      - ``"edge"``: reflect padding; when the pad is not strictly smaller
        than the image dimension on that axis (torch's reflect limit) the
        pass falls back to replicate padding instead of raising.
    """
    if pad <= 0:
        return tensor
    pads = (pad, pad, 0, 0) if axis == "x" else (0, 0, pad, pad)
    if edge_mode == "zero":
        return F.pad(tensor, pads, mode="constant", value=0.0)
    dim = tensor.shape[3] if axis == "x" else tensor.shape[2]
    mode = "reflect" if pad < dim else "replicate"
    return F.pad(tensor, pads, mode=mode)


def _depthwise_conv2d(tensor, kernel_2d, edge_mode="edge"):
    """Convolve every channel of an NCHW tensor with the same 2-D kernel
    ``[kh, kw]`` (odd sizes), padding each axis with ``_pad_axis`` so the
    spatial size is preserved."""
    kh, kw = kernel_2d.shape
    channels = tensor.shape[1]
    weight = kernel_2d.to(tensor.dtype).view(1, 1, kh, kw).repeat(channels, 1, 1, 1)
    padded = _pad_axis(tensor, kw // 2, "x", edge_mode)
    padded = _pad_axis(padded, kh // 2, "y", edge_mode)
    return F.conv2d(padded, weight, groups=channels)


class NukeBlur(NukeNodeBase):
    """
    Separable blur node with separate X/Y controls (similar to Nuke's Blur node)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size_x": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "size_y": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "filter": (
                    ["gaussian", "box", "triangle", "quadratic"],
                    {"default": "gaussian"},
                ),
                "quality": (["low", "medium", "high"], {"default": "medium"}),
                "crop": ("BOOLEAN", {"default": True}),
                "mix": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"
    CATEGORY = "Nuke/Filter"

    FILTERS = ("gaussian", "box", "triangle", "quadratic")

    def blur(self, image, size_x, size_y, filter, quality, crop, mix, mask=None):
        """
        Apply a separable blur with separate X/Y controls.

        ``crop`` follows Nuke: True treats everything outside the image as
        black (the blur fades to black at the format edges); False treats
        the outside as the edge colour (reflect padding, no fade).
        """
        img = ensure_batch_dim(image)

        # Separate RGB and alpha channels
        if img.shape[3] >= 4:
            rgb = img[:, :, :, :3]
            alpha = img[:, :, :, 3:]
        else:
            rgb = img
            alpha = None

        # Convert to tensor format for convolution (B, C, H, W)
        rgb_tensor = rgb.permute(0, 3, 1, 2)

        edge_mode = "zero" if crop else "edge"

        # Apply blur
        if (size_x > 0 or size_y > 0) and filter in self.FILTERS:
            blurred = self._separable_blur(
                rgb_tensor, size_x, size_y, filter, quality, edge_mode
            )
        else:
            blurred = rgb_tensor

        # Convert back to ComfyUI format
        blurred = blurred.permute(0, 2, 3, 1)

        # Blend by mix and the optional MASK ([H,W] or [B,H,W])
        blurred = apply_mask_mix(rgb, blurred, mask, mix)

        # Recombine with alpha
        if alpha is not None:
            result = torch.cat([blurred, alpha], dim=3)
        else:
            result = blurred

        return (normalize_tensor(result),)

    # ------------------------------------------------------------------ kernels

    def _kernel_1d(self, filter, size, quality, device):
        """Build the normalised 1-D kernel for one axis, or return None when
        the pass is a no-op (size <= 0, or a kernel of a single tap).

        Box / triangle / quadratic are exact finite-support shapes: every
        integer tap inside the support (radius ``int(size * 2)``) is used,
        so ``quality`` has nothing to add and is ignored for them. The
        gaussian has infinite support; ``quality`` sets how far the kernel
        is truncated (radius ``int(size * qmult * 2)``) while
        ``sigma = size / 3`` stays fixed.
        """
        if size <= 0:
            return None
        if filter == "gaussian":
            radius = int(size * QUALITY_MULTIPLIERS[quality] * 2)
        else:
            radius = int(size * 2)
        kernel_size = radius * 2 + 1
        if kernel_size <= 1:
            return None

        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        center = kernel_size // 2
        if filter == "gaussian":
            sigma = size / 3.0
            kernel = torch.exp(-((coords - center) ** 2) / (2 * sigma**2))
        elif filter == "box":
            kernel = torch.ones(kernel_size, dtype=torch.float32, device=device)
        elif filter == "triangle":
            kernel = 1 - torch.abs(coords - center) / (center + 1)
        elif filter == "quadratic":
            distances = torch.abs(coords - center) / (center + 1)
            kernel = torch.clamp(1 - distances**2, min=0)
        else:
            return None
        return kernel / kernel.sum()

    def _separable_blur(self, img_tensor, size_x, size_y, filter, quality, edge_mode):
        """Horizontal pass (width axis) then vertical pass (height axis)."""
        result = img_tensor
        channels = img_tensor.shape[1]

        kernel_x = self._kernel_1d(filter, size_x, quality, img_tensor.device)
        if kernel_x is not None:
            ks = kernel_x.numel()
            weight = kernel_x.to(img_tensor.dtype).view(1, 1, 1, ks).repeat(channels, 1, 1, 1)
            result = _pad_axis(result, ks // 2, "x", edge_mode)
            result = F.conv2d(result, weight, groups=channels)

        kernel_y = self._kernel_1d(filter, size_y, quality, img_tensor.device)
        if kernel_y is not None:
            ks = kernel_y.numel()
            weight = kernel_y.to(img_tensor.dtype).view(1, 1, ks, 1).repeat(channels, 1, 1, 1)
            result = _pad_axis(result, ks // 2, "y", edge_mode)
            result = F.conv2d(result, weight, groups=channels)

        return result


class NukeMotionBlur(NukeNodeBase):
    """
    Directional motion blur node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "distance": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "angle": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
                ),
                "samples": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "shutter": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "center_bias": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "mix": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "motion_blur"
    CATEGORY = "Nuke/Filter"

    def motion_blur(self, image, distance, angle, samples, shutter, center_bias, mix):
        """
        Apply directional motion blur: a weighted average of ``samples``
        copies of the image shifted along the motion vector.
        """
        img = ensure_batch_dim(image)

        if distance <= 0:
            return (img,)

        # Convert angle to radians
        angle_rad = math.radians(angle)

        # Calculate motion vector
        dx = math.cos(angle_rad) * distance
        dy = math.sin(angle_rad) * distance

        # Convert to tensor format
        img_tensor = img.permute(0, 3, 1, 2)
        height, width = img_tensor.shape[2], img_tensor.shape[3]

        positions = self._sample_positions(samples, shutter)
        weights = self._sample_weights(samples, center_bias)

        if all(t == 0 for t in positions):
            # samples == 1 or shutter == 0: every sample is the unshifted
            # image, so the result is the input itself (bit-exact).
            return (normalize_tensor(img),)

        # Accumulate samples
        accumulated = torch.zeros_like(img_tensor)
        total_weight = 0.0

        for t, weight in zip(positions, weights):
            if weight <= 0:
                continue

            offset_x = dx * t
            offset_y = dy * t

            if offset_x == 0 and offset_y == 0:
                # Zero shift is the image itself - keep it bit-exact
                sample = img_tensor
            else:
                grid = self._create_motion_grid(
                    offset_x, offset_y, height, width, img_tensor.device
                )
                # Grid is built for batch 1; expand to the batch
                grid = grid.to(img_tensor.dtype).expand(img_tensor.shape[0], -1, -1, -1)
                sample = F.grid_sample(
                    img_tensor,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )

            accumulated += sample * weight
            total_weight += weight

        # Weighted average of the samples
        blurred_tensor = accumulated / total_weight

        # Convert back to ComfyUI format
        blurred = blurred_tensor.permute(0, 2, 3, 1)

        # Mix with original
        result = img + (blurred - img) * mix

        return (normalize_tensor(result),)

    @staticmethod
    def _sample_positions(samples, shutter):
        """Shutter-relative positions t in [-shutter/2, +shutter/2], evenly spaced."""
        if samples <= 1:
            return [0.0]
        return [(i / (samples - 1) - 0.5) * shutter for i in range(samples)]

    @staticmethod
    def _sample_weights(samples, center_bias):
        """Per-sample weights: ``1 + center_bias * (1 - 2 * |u|)`` where u is
        the sample's normalised position in [-0.5, 0.5]. Positive bias weights
        the centre of the shutter more (sharper core, softer trails); negative
        bias weights the ends more (ghosted look). The end samples always
        keep weight 1, so the total weight is positive for samples >= 2. A
        single sample always has weight 1."""
        if samples <= 1:
            return [1.0]
        bias = max(-1.0, min(1.0, float(center_bias)))
        weights = []
        for i in range(samples):
            u = i / (samples - 1) - 0.5
            weights.append(max(0.0, 1.0 + bias * (1.0 - 2.0 * abs(u))))
        return weights

    def _create_motion_grid(self, offset_x, offset_y, height, width, device):
        """Create a pixel-centre-aligned sampling grid shifted by a pixel offset.

        With ``align_corners=False`` the normalised coordinate of pixel
        centre ``col`` is ``2 * (col + 0.5) / W - 1`` and a shift of
        ``offset_px`` pixels is ``2 * offset_px / W`` (same for rows / H).
        """
        x_coords = (2.0 * (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width) - 1.0
        y_coords = (2.0 * (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / height) - 1.0
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Convert pixel offset to normalised units
        x_grid_offset = x_grid + 2.0 * offset_x / width
        y_grid_offset = y_grid + 2.0 * offset_y / height

        # Stack coordinates (x first)
        grid = torch.stack([x_grid_offset, y_grid_offset], dim=2).unsqueeze(0)

        return grid


class NukeDefocus(NukeNodeBase):
    """
    Depth-of-field style defocus blur
    """

    METHODS = ("gaussian", "disk", "hexagon")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "defocus": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "aspect_ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01},
                ),
                "quality": (["low", "medium", "high"], {"default": "medium"}),
                "method": (["gaussian", "disk", "hexagon"], {"default": "disk"}),
                "mix": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "depth_map": ("MASK",),
                "focus_distance": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "defocus"
    CATEGORY = "Nuke/Filter"

    def defocus(
        self,
        image,
        defocus,
        aspect_ratio,
        quality,
        method,
        mix,
        depth_map=None,
        focus_distance=0.5,
    ):
        """
        Apply a defocus blur. The blur is uniform over the image; an optional
        depth map only scales its strength by the largest deviation from
        ``focus_distance`` (see the guide).
        """
        img = ensure_batch_dim(image)

        if defocus <= 0:
            return (img,)

        # Calculate blur amount based on depth map (MASK) if provided
        if depth_map is not None:
            depth_value = mask_to_bhw1(
                depth_map, img.shape[1], img.shape[2], img.device
            ).to(img.dtype)

            # Calculate blur amount based on distance from focus
            blur_amount = torch.abs(depth_value - focus_distance) * defocus
        else:
            # Uniform blur
            blur_amount = torch.full_like(img[:, :, :, :1], defocus)

        # Apply blur (uniform, strength = max of blur_amount)
        result = self._apply_variable_blur(
            img, blur_amount, aspect_ratio, quality, method
        )

        # Mix with original
        result = apply_mask_mix(img, result, None, mix)

        return (normalize_tensor(result),)

    def _apply_variable_blur(self, img, blur_amount, aspect_ratio, quality, method):
        """Apply a uniform blur whose strength is the maximum blur amount.

        ``aspect_ratio`` scales the HORIZONTAL extent of the bokeh relative
        to the vertical one (> 1 = wider than tall).
        """
        max_blur = torch.max(blur_amount).item()

        if max_blur <= 0:
            return img

        if method not in self.METHODS:
            logger.warning(
                "NukeDefocus: unknown method %r, falling back to 'disk'", method
            )
            method = "disk"

        # Convert to tensor format
        img_tensor = img.permute(0, 3, 1, 2)
        device = img_tensor.device

        # Horizontal / vertical bokeh sizes in pixels
        size_x = max_blur
        size_y = max_blur / aspect_ratio

        if method == "gaussian":
            quality_mult = QUALITY_MULTIPLIERS[quality]
            radius_x = max(1, int(size_x * quality_mult * 2))
            radius_y = max(1, int(size_y * quality_mult * 2))
            kernel = self._create_gaussian_kernel_2d(
                radius_x, radius_y, size_x / 3.0, size_y / 3.0, device
            )
            result = _depthwise_conv2d(img_tensor, kernel)

        elif method == "disk":
            # Soft-disk approximation: `passes` stacked gaussian passes whose
            # combined sigma is size / 2 on each axis at every quality
            # (per-pass sigma = total sigma / sqrt(passes)). Each pass
            # covers at least 3 sigma and never less than 1 pixel.
            passes = DISK_PASSES[quality]
            sigma_x = (size_x / 2.0) / math.sqrt(passes)
            sigma_y = (size_y / 2.0) / math.sqrt(passes)
            radius_x = max(1, int(math.ceil(3.0 * sigma_x)))
            radius_y = max(1, int(math.ceil(3.0 * sigma_y)))
            kernel = self._create_gaussian_kernel_2d(
                radius_x, radius_y, sigma_x, sigma_y, device
            )
            result = img_tensor
            for _ in range(passes):
                result = _depthwise_conv2d(result, kernel)

        else:  # hexagon
            kernel = self._create_hexagon_kernel(
                size_x, aspect_ratio, HEXAGON_SUPERSAMPLES[quality], device
            )
            result = _depthwise_conv2d(img_tensor, kernel)

        # Convert back to ComfyUI format
        return result.permute(0, 2, 3, 1)

    def _create_gaussian_kernel_2d(self, radius_x, radius_y, sigma_x, sigma_y, device):
        """Create a normalised 2-D Gaussian kernel of shape
        ``[2 * radius_y + 1, 2 * radius_x + 1]``; ``sigma_x`` governs the
        horizontal (column) spread and ``sigma_y`` the vertical (row) spread."""
        sigma_x = max(float(sigma_x), 1e-4)
        sigma_y = max(float(sigma_y), 1e-4)
        coords_x = torch.arange(-radius_x, radius_x + 1, dtype=torch.float32, device=device)
        coords_y = torch.arange(-radius_y, radius_y + 1, dtype=torch.float32, device=device)
        y_grid, x_grid = torch.meshgrid(coords_y, coords_x, indexing="ij")

        kernel = torch.exp(
            -(x_grid**2) / (2 * sigma_x**2) - (y_grid**2) / (2 * sigma_y**2)
        )
        kernel = kernel / kernel.sum()

        return kernel

    def _create_hexagon_kernel(self, radius, aspect_ratio, supersamples, device):
        """Rasterise a regular flat-top hexagon into a normalised 2-D kernel.

        The hexagon has circumradius ``max(radius, 1)`` pixels horizontally
        (vertex to vertex along the row axis) and a half-height of
        ``radius * sqrt(3) / 2 / aspect_ratio`` (flat edges top and bottom).
        Each pixel's weight is the fraction of ``supersamples x supersamples``
        sub-positions inside the hexagon.
        """
        r = max(float(radius), 1.0)
        half_height = r * math.sqrt(3.0) / 2.0 / aspect_ratio
        radius_x = int(math.ceil(r))
        radius_y = int(math.ceil(half_height))
        ss = max(1, int(supersamples))

        sub = (torch.arange(ss, dtype=torch.float32, device=device) + 0.5) / ss - 0.5
        xs = (torch.arange(-radius_x, radius_x + 1, dtype=torch.float32, device=device)[:, None] + sub[None, :]).reshape(-1)
        ys = (torch.arange(-radius_y, radius_y + 1, dtype=torch.float32, device=device)[:, None] + sub[None, :]).reshape(-1)
        # Undo the vertical aspect scaling so the test is against a regular hexagon
        ys_unit = ys * aspect_ratio
        ax = torch.abs(xs)[None, :]
        ay = torch.abs(ys_unit)[:, None]
        inside = (ay <= r * math.sqrt(3.0) / 2.0 + 1e-6) & (ax + ay / math.sqrt(3.0) <= r + 1e-6)
        coverage = inside.to(torch.float32).reshape(2 * radius_y + 1, ss, 2 * radius_x + 1, ss).mean(dim=(1, 3))

        total = coverage.sum()
        if total <= 0:
            coverage[radius_y, radius_x] = 1.0
            total = coverage.sum()
        return coverage / total


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeBlur": NukeBlur,
    "NukeMotionBlur": NukeMotionBlur,
    "NukeDefocus": NukeDefocus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeBlur": "Nuke Blur",
    "NukeMotionBlur": "Nuke Motion Blur",
    "NukeDefocus": "Nuke Defocus",
}
