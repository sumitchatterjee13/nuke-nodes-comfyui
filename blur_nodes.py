"""
Blur and filtering nodes that replicate Nuke's blur functionality
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor


class NukeBlur(NukeNodeBase):
    """
    Gaussian blur node with separate X/Y controls (similar to Nuke's Blur node)
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
                "mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"
    CATEGORY = "Nuke/Filter"

    def blur(self, image, size_x, size_y, filter, quality, crop, mix, mask=None):
        """
        Apply Gaussian blur with separate X/Y controls
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

        # Apply blur
        if size_x > 0 or size_y > 0:
            if filter == "gaussian":
                blurred = self._gaussian_blur(rgb_tensor, size_x, size_y, quality)
            elif filter == "box":
                blurred = self._box_blur(rgb_tensor, size_x, size_y)
            elif filter == "triangle":
                blurred = self._triangle_blur(rgb_tensor, size_x, size_y)
            elif filter == "quadratic":
                blurred = self._quadratic_blur(rgb_tensor, size_x, size_y)
            else:
                blurred = rgb_tensor
        else:
            blurred = rgb_tensor

        # Convert back to ComfyUI format
        blurred = blurred.permute(0, 2, 3, 1)

        # Apply mask if provided
        if mask is not None:
            mask = ensure_batch_dim(mask)
            if mask.shape[1:3] != blurred.shape[1:3]:
                mask = F.interpolate(
                    mask.permute(0, 3, 1, 2),
                    size=blurred.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

            mask_alpha = mask[:, :, :, :1]
            blurred = rgb + (blurred - rgb) * mask_alpha * mix
        else:
            blurred = rgb + (blurred - rgb) * mix

        # Recombine with alpha
        if alpha is not None:
            result = torch.cat([blurred, alpha], dim=3)
        else:
            result = blurred

        return (normalize_tensor(result),)

    def _gaussian_blur(self, img_tensor, size_x, size_y, quality):
        """Apply separable Gaussian blur"""
        # Quality settings
        quality_multipliers = {"low": 1.0, "medium": 2.0, "high": 3.0}
        quality_mult = quality_multipliers[quality]

        # Calculate kernel sizes
        kernel_size_x = int(size_x * quality_mult * 2) * 2 + 1  # Ensure odd
        kernel_size_y = int(size_y * quality_mult * 2) * 2 + 1

        result = img_tensor

        # Apply horizontal blur
        if size_x > 0 and kernel_size_x > 1:
            sigma_x = size_x / 3.0  # Standard deviation
            kernel_x = self._create_gaussian_kernel(
                kernel_size_x, sigma_x, img_tensor.device
            )
            kernel_x = kernel_x.view(1, 1, 1, -1).repeat(img_tensor.shape[1], 1, 1, 1)

            # Apply padding
            pad_x = kernel_size_x // 2
            result = F.pad(result, (pad_x, pad_x, 0, 0), mode="reflect")
            result = F.conv2d(result, kernel_x, groups=img_tensor.shape[1])

        # Apply vertical blur
        if size_y > 0 and kernel_size_y > 1:
            sigma_y = size_y / 3.0
            kernel_y = self._create_gaussian_kernel(
                kernel_size_y, sigma_y, img_tensor.device
            )
            kernel_y = kernel_y.view(1, 1, -1, 1).repeat(img_tensor.shape[1], 1, 1, 1)

            # Apply padding
            pad_y = kernel_size_y // 2
            result = F.pad(result, (0, 0, pad_y, pad_y), mode="reflect")
            result = F.conv2d(result, kernel_y, groups=img_tensor.shape[1])

        return result

    def _create_gaussian_kernel(self, kernel_size, sigma, device):
        """Create 1D Gaussian kernel"""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords -= kernel_size // 2

        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        return kernel

    def _box_blur(self, img_tensor, size_x, size_y):
        """Apply box blur (uniform kernel)"""
        result = img_tensor

        # Horizontal box blur
        if size_x > 0:
            kernel_size_x = int(size_x * 2) * 2 + 1
            kernel_x = (
                torch.ones(1, 1, 1, kernel_size_x, device=img_tensor.device)
                / kernel_size_x
            )
            kernel_x = kernel_x.repeat(img_tensor.shape[1], 1, 1, 1)

            pad_x = kernel_size_x // 2
            result = F.pad(result, (pad_x, pad_x, 0, 0), mode="reflect")
            result = F.conv2d(result, kernel_x, groups=img_tensor.shape[1])

        # Vertical box blur
        if size_y > 0:
            kernel_size_y = int(size_y * 2) * 2 + 1
            kernel_y = (
                torch.ones(1, 1, kernel_size_y, 1, device=img_tensor.device)
                / kernel_size_y
            )
            kernel_y = kernel_y.repeat(img_tensor.shape[1], 1, 1, 1)

            pad_y = kernel_size_y // 2
            result = F.pad(result, (0, 0, pad_y, pad_y), mode="reflect")
            result = F.conv2d(result, kernel_y, groups=img_tensor.shape[1])

        return result

    def _triangle_blur(self, img_tensor, size_x, size_y):
        """Apply triangle blur (triangular kernel)"""
        result = img_tensor

        # Horizontal triangle blur
        if size_x > 0:
            kernel_size_x = int(size_x * 2) * 2 + 1
            center = kernel_size_x // 2
            coords = torch.arange(
                kernel_size_x, dtype=torch.float32, device=img_tensor.device
            )
            kernel_x = 1 - torch.abs(coords - center) / (center + 1)
            kernel_x = kernel_x / kernel_x.sum()
            kernel_x = kernel_x.view(1, 1, 1, -1).repeat(img_tensor.shape[1], 1, 1, 1)

            pad_x = kernel_size_x // 2
            result = F.pad(result, (pad_x, pad_x, 0, 0), mode="reflect")
            result = F.conv2d(result, kernel_x, groups=img_tensor.shape[1])

        # Vertical triangle blur
        if size_y > 0:
            kernel_size_y = int(size_y * 2) * 2 + 1
            center = kernel_size_y // 2
            coords = torch.arange(
                kernel_size_y, dtype=torch.float32, device=img_tensor.device
            )
            kernel_y = 1 - torch.abs(coords - center) / (center + 1)
            kernel_y = kernel_y / kernel_y.sum()
            kernel_y = kernel_y.view(1, 1, -1, 1).repeat(img_tensor.shape[1], 1, 1, 1)

            pad_y = kernel_size_y // 2
            result = F.pad(result, (0, 0, pad_y, pad_y), mode="reflect")
            result = F.conv2d(result, kernel_y, groups=img_tensor.shape[1])

        return result

    def _quadratic_blur(self, img_tensor, size_x, size_y):
        """Apply quadratic blur (quadratic kernel)"""
        result = img_tensor

        # Horizontal quadratic blur
        if size_x > 0:
            kernel_size_x = int(size_x * 2) * 2 + 1
            center = kernel_size_x // 2
            coords = torch.arange(
                kernel_size_x, dtype=torch.float32, device=img_tensor.device
            )
            distances = torch.abs(coords - center) / (center + 1)
            kernel_x = torch.clamp(1 - distances**2, min=0)
            kernel_x = kernel_x / kernel_x.sum()
            kernel_x = kernel_x.view(1, 1, 1, -1).repeat(img_tensor.shape[1], 1, 1, 1)

            pad_x = kernel_size_x // 2
            result = F.pad(result, (pad_x, pad_x, 0, 0), mode="reflect")
            result = F.conv2d(result, kernel_x, groups=img_tensor.shape[1])

        # Vertical quadratic blur
        if size_y > 0:
            kernel_size_y = int(size_y * 2) * 2 + 1
            center = kernel_size_y // 2
            coords = torch.arange(
                kernel_size_y, dtype=torch.float32, device=img_tensor.device
            )
            distances = torch.abs(coords - center) / (center + 1)
            kernel_y = torch.clamp(1 - distances**2, min=0)
            kernel_y = kernel_y / kernel_y.sum()
            kernel_y = kernel_y.view(1, 1, -1, 1).repeat(img_tensor.shape[1], 1, 1, 1)

            pad_y = kernel_size_y // 2
            result = F.pad(result, (0, 0, pad_y, pad_y), mode="reflect")
            result = F.conv2d(result, kernel_y, groups=img_tensor.shape[1])

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
        Apply directional motion blur
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

        # Accumulate samples
        accumulated = torch.zeros_like(img_tensor)
        total_weight = 0

        for i in range(samples):
            # Calculate sample position
            if samples == 1:
                t = 0
            else:
                t = (i / (samples - 1) - 0.5) * shutter

            # Apply center bias
            if center_bias != 0:
                # Bias towards center
                t = t * (1 + center_bias * (1 - 2 * abs(t)))

            # Calculate offset
            offset_x = dx * t
            offset_y = dy * t

            # Create sampling grid
            grid = self._create_motion_grid(
                offset_x, offset_y, height, width, img_tensor.device
            )

            # Sample image
            sample = F.grid_sample(
                img_tensor, grid, mode="bilinear", align_corners=False
            )

            # Weight based on distance from center (optional)
            weight = 1.0
            accumulated += sample * weight
            total_weight += weight

        # Average samples
        blurred_tensor = accumulated / total_weight

        # Convert back to ComfyUI format
        blurred = blurred_tensor.permute(0, 2, 3, 1)

        # Mix with original
        result = img + (blurred - img) * mix

        return (normalize_tensor(result),)

    def _create_motion_grid(self, offset_x, offset_y, height, width, device):
        """Create sampling grid for motion blur"""
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Apply offset (convert pixel offset to normalized coordinates)
        x_offset_norm = offset_x * 2 / width
        y_offset_norm = offset_y * 2 / height

        x_grid_offset = x_grid + x_offset_norm
        y_grid_offset = y_grid + y_offset_norm

        # Stack coordinates
        grid = torch.stack([x_grid_offset, y_grid_offset], dim=2).unsqueeze(0)

        return grid


class NukeDefocus(NukeNodeBase):
    """
    Depth-of-field style defocus blur
    """

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
                "depth_map": ("IMAGE",),
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
        Apply depth-of-field defocus blur
        """
        img = ensure_batch_dim(image)

        if defocus <= 0:
            return (img,)

        # Calculate blur amount based on depth map if provided
        if depth_map is not None:
            depth = ensure_batch_dim(depth_map)
            if depth.shape[1:3] != img.shape[1:3]:
                depth = F.interpolate(
                    depth.permute(0, 3, 1, 2),
                    size=img.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

            # Calculate blur amount based on distance from focus
            depth_value = depth[:, :, :, :1]  # Use first channel
            blur_amount = torch.abs(depth_value - focus_distance) * defocus
        else:
            # Uniform blur
            blur_amount = torch.full_like(img[:, :, :, :1], defocus)

        # Apply variable blur
        result = self._apply_variable_blur(
            img, blur_amount, aspect_ratio, quality, method
        )

        # Mix with original
        result = img + (result - img) * mix

        return (normalize_tensor(result),)

    def _apply_variable_blur(self, img, blur_amount, aspect_ratio, quality, method):
        """Apply spatially varying blur"""
        # For simplicity, we'll apply uniform blur based on maximum blur amount
        # A full implementation would use more sophisticated techniques

        max_blur = torch.max(blur_amount).item()

        if max_blur <= 0:
            return img

        # Convert to tensor format
        img_tensor = img.permute(0, 3, 1, 2)

        if method == "gaussian":
            # Apply Gaussian blur
            size_x = max_blur
            size_y = max_blur / aspect_ratio

            # Create Gaussian kernel
            quality_mult = {"low": 1.0, "medium": 2.0, "high": 3.0}[quality]
            kernel_size = int(max_blur * quality_mult * 2) * 2 + 1

            if kernel_size > 1:
                sigma = max_blur / 3.0
                kernel = self._create_gaussian_kernel_2d(
                    kernel_size, sigma, sigma / aspect_ratio, img_tensor.device
                )
                kernel = (
                    kernel.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(img_tensor.shape[1], 1, 1, 1)
                )

                pad = kernel_size // 2
                img_padded = F.pad(img_tensor, (pad, pad, pad, pad), mode="reflect")
                result = F.conv2d(img_padded, kernel, groups=img_tensor.shape[1])
            else:
                result = img_tensor

        elif method == "disk":
            # Disk blur approximation using multiple Gaussian passes
            result = img_tensor
            passes = {"low": 1, "medium": 2, "high": 3}[quality]

            for _ in range(passes):
                size_x = max_blur / passes
                size_y = size_x / aspect_ratio

                kernel_size = int(size_x * 2) * 2 + 1
                if kernel_size > 1:
                    sigma_x = size_x / 2.0
                    sigma_y = size_y / 2.0
                    kernel = self._create_gaussian_kernel_2d(
                        kernel_size, sigma_x, sigma_y, img_tensor.device
                    )
                    kernel = (
                        kernel.unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(result.shape[1], 1, 1, 1)
                    )

                    pad = kernel_size // 2
                    result_padded = F.pad(result, (pad, pad, pad, pad), mode="reflect")
                    result = F.conv2d(result_padded, kernel, groups=result.shape[1])

        elif method == "hexagon":
            # Hexagonal blur approximation
            result = img_tensor
            # This would require a more complex implementation
            # For now, fall back to Gaussian
            size_x = max_blur
            size_y = max_blur / aspect_ratio

            kernel_size = int(max_blur * 2) * 2 + 1
            if kernel_size > 1:
                sigma = max_blur / 3.0
                kernel = self._create_gaussian_kernel_2d(
                    kernel_size, sigma, sigma / aspect_ratio, img_tensor.device
                )
                kernel = (
                    kernel.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(img_tensor.shape[1], 1, 1, 1)
                )

                pad = kernel_size // 2
                img_padded = F.pad(img_tensor, (pad, pad, pad, pad), mode="reflect")
                result = F.conv2d(img_padded, kernel, groups=img_tensor.shape[1])
            else:
                result = img_tensor

        # Convert back to ComfyUI format
        return result.permute(0, 2, 3, 1)

    def _create_gaussian_kernel_2d(self, size, sigma_x, sigma_y, device):
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size, dtype=torch.float32, device=device)
        coords -= size // 2

        x_grid, y_grid = torch.meshgrid(coords, coords, indexing="ij")

        kernel = torch.exp(
            -(x_grid**2) / (2 * sigma_x**2) - (y_grid**2) / (2 * sigma_y**2)
        )
        kernel = kernel / kernel.sum()

        return kernel


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
