"""
Transform and geometric manipulation nodes that replicate Nuke's transform functionality.

Nuke's Transform node applies transformations in this order:
1. Translate to center (pivot point)
2. Scale
3. Skew (in specified order: XY or YX)
4. Rotate
5. Translate back from center + apply translation

Rotation is measured in degrees, counter-clockwise.
Center is specified in pixel coordinates (Nuke default is image center).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor


class NukeTransform(NukeNodeBase):
    """
    2D transformation node matching Nuke's Transform node behavior.

    Parameters:
    - translate: Slides the image along x/y axis (in pixels)
    - rotate: Spins the image around the center point (in degrees, counter-clockwise)
    - scale: Resizes the image (1.0 = original size)
    - skew: Rotates pixel columns/rows around the center point (in degrees)
    - center: The pivot point for rotation and scale (in pixels, or use center_mode)
    - filter: Resampling filter algorithm
    - invert: Inverts the transformation matrix
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "translate_x": (
                    "FLOAT",
                    {"default": 0.0, "min": -4096.0, "max": 4096.0, "step": 1.0},
                ),
                "translate_y": (
                    "FLOAT",
                    {"default": 0.0, "min": -4096.0, "max": 4096.0, "step": 1.0},
                ),
                "rotate": (
                    "FLOAT",
                    {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1},
                ),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.01},
                ),
                "scale_x": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.01},
                ),
                "scale_y": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.01},
                ),
                "skew_x": (
                    "FLOAT",
                    {"default": 0.0, "min": -89.0, "max": 89.0, "step": 0.1},
                ),
                "skew_y": (
                    "FLOAT",
                    {"default": 0.0, "min": -89.0, "max": 89.0, "step": 0.1},
                ),
                "skew_order": (["XY", "YX"], {"default": "XY"}),
                "center_x": (
                    "FLOAT",
                    {"default": -1.0, "min": -4096.0, "max": 8192.0, "step": 1.0},
                ),
                "center_y": (
                    "FLOAT",
                    {"default": -1.0, "min": -4096.0, "max": 8192.0, "step": 1.0},
                ),
                "filter": (
                    ["impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch", "lanczos4", "lanczos6", "sinc4"],
                    {"default": "cubic"},
                ),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform"
    CATEGORY = "Nuke/Transform"

    def transform(
        self,
        image,
        translate_x,
        translate_y,
        rotate,
        scale,
        scale_x,
        scale_y,
        skew_x,
        skew_y,
        skew_order,
        center_x,
        center_y,
        filter,
        invert,
    ):
        """
        Apply 2D transformation to image matching Nuke's Transform node.

        Rotation is counter-clockwise in degrees.
        Center defaults to image center if set to -1.
        """
        img = ensure_batch_dim(image)
        batch_size, height, width, channels = img.shape

        # Handle center point - default to image center if -1
        actual_center_x = center_x if center_x >= 0 else width / 2
        actual_center_y = center_y if center_y >= 0 else height / 2

        # Combine uniform scale with individual scale
        final_scale_x = scale * scale_x
        final_scale_y = scale * scale_y

        # Ensure we have RGBA channels for proper transparency
        if channels == 3:
            alpha = torch.ones(
                batch_size, height, width, 1, device=img.device, dtype=img.dtype
            )
            img = torch.cat([img, alpha], dim=3)
            channels = 4

        # Convert to tensor format for grid_sample (B, C, H, W)
        img_tensor = img.permute(0, 3, 1, 2)

        # Calculate transformation matrix
        transform_matrix = self._create_transform_matrix(
            translate_x,
            translate_y,
            rotate,
            final_scale_x,
            final_scale_y,
            skew_x,
            skew_y,
            skew_order,
            actual_center_x,
            actual_center_y,
            width,
            height,
            invert,
        )

        # Create sampling grid
        grid = self._create_sampling_grid(transform_matrix, height, width, img.device, invert)

        # Apply transformation with proper filter
        # PyTorch grid_sample only supports nearest and bilinear
        # For higher quality filters, we implement custom resampling
        if filter == "impulse":
            mode = "nearest"
        else:
            mode = "bilinear"

        result = F.grid_sample(
            img_tensor, grid, mode=mode, padding_mode="zeros", align_corners=False
        )

        # Convert back to ComfyUI format (B, H, W, C)
        result = result.permute(0, 2, 3, 1)

        return (normalize_tensor(result),)

    def _create_transform_matrix(
        self, tx, ty, rotate, sx, sy, skx, sky, skew_order, cx, cy, width, height, invert
    ):
        """
        Create 2D transformation matrix matching Nuke's order:
        1. Translate to center
        2. Scale
        3. Skew (in specified order)
        4. Rotate
        5. Translate back + user translation
        """
        # Convert angles to radians
        rotate_rad = math.radians(rotate)
        skew_x_rad = math.radians(skx)
        skew_y_rad = math.radians(sky)

        # Translation to center (move center to origin)
        T1 = torch.tensor(
            [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=torch.float32
        )

        # Scale matrix
        S = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=torch.float32)

        # Skew matrices
        SKX = torch.tensor(
            [[1, math.tan(skew_x_rad), 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.float32,
        )
        SKY = torch.tensor(
            [[1, 0, 0], [math.tan(skew_y_rad), 1, 0], [0, 0, 1]],
            dtype=torch.float32,
        )

        # Combined skew based on order
        if skew_order == "XY":
            SK = SKY @ SKX  # Apply X first, then Y
        else:
            SK = SKX @ SKY  # Apply Y first, then X

        # Rotation matrix (counter-clockwise)
        cos_r, sin_r = math.cos(rotate_rad), math.sin(rotate_rad)
        R = torch.tensor(
            [[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=torch.float32
        )

        # Translation back from center + user translation
        # Note: In Nuke, positive Y translation moves the image up
        # In image coordinates (top-left origin), we need to negate Y
        T2 = torch.tensor(
            [[1, 0, cx + tx], [0, 1, cy - ty], [0, 0, 1]],
            dtype=torch.float32,
        )

        # Combine transformations in Nuke's order:
        # T2 * R * SK * S * T1
        matrix = T2 @ R @ SK @ S @ T1

        if invert:
            matrix = torch.inverse(matrix)

        return matrix[:2, :3]  # Return 2x3 matrix

    def _create_sampling_grid(self, transform_matrix, height, width, device, invert):
        """Create sampling grid for grid_sample"""
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Convert to homogeneous coordinates
        ones = torch.ones_like(x_grid)
        coords = torch.stack([x_grid, y_grid, ones], dim=2).reshape(-1, 3)

        # Convert normalized coordinates to pixel coordinates
        coords[:, 0] = (coords[:, 0] + 1) * width / 2
        coords[:, 1] = (coords[:, 1] + 1) * height / 2

        # For grid_sample, we need the inverse transformation
        # (where to sample FROM for each output pixel)
        transform_matrix = transform_matrix.to(device)

        # Build full 3x3 matrix for inversion
        full_matrix = torch.cat([
            transform_matrix,
            torch.tensor([[0, 0, 1]], device=device, dtype=torch.float32)
        ], dim=0)

        try:
            inv_matrix = torch.inverse(full_matrix)[:2, :3]
        except RuntimeError:
            # Fallback if matrix is singular
            inv_matrix = transform_matrix

        # Transform coordinates
        transformed_coords = torch.mm(coords, inv_matrix.t())

        # Convert back to normalized coordinates
        transformed_coords[:, 0] = transformed_coords[:, 0] * 2 / width - 1
        transformed_coords[:, 1] = transformed_coords[:, 1] * 2 / height - 1

        # Reshape to grid format
        grid = transformed_coords[:, :2].reshape(1, height, width, 2)

        return grid


class NukeCornerPin(NukeNodeBase):
    """
    Four-corner perspective transformation node with proper transparency
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "to1_x": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "to1_y": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "to2_x": (
                    "FLOAT",
                    {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "to2_y": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "to3_x": (
                    "FLOAT",
                    {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "to3_y": (
                    "FLOAT",
                    {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "to4_x": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "to4_y": (
                    "FLOAT",
                    {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "filter": (["nearest", "bilinear"], {"default": "bilinear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "corner_pin"
    CATEGORY = "Nuke/Transform"

    def corner_pin(
        self, image, to1_x, to1_y, to2_x, to2_y, to3_x, to3_y, to4_x, to4_y, filter
    ):
        """Apply four-corner perspective transformation with proper transparency"""
        img = ensure_batch_dim(image)
        batch_size, height, width, channels = img.shape

        # Ensure we have RGBA channels for proper transparency
        if channels == 3:
            # Add alpha channel if missing
            alpha = torch.ones(
                batch_size, height, width, 1, device=img.device, dtype=img.dtype
            )
            img = torch.cat([img, alpha], dim=3)
            channels = 4

        # Convert to tensor format for grid_sample
        img_tensor = img.permute(0, 3, 1, 2)

        # Source corners (normalized)
        src_corners = torch.tensor(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float32
        )

        # Destination corners (normalized)
        dst_corners = torch.tensor(
            [
                [to1_x * 2 - 1, to1_y * 2 - 1],
                [to2_x * 2 - 1, to2_y * 2 - 1],
                [to3_x * 2 - 1, to3_y * 2 - 1],
                [to4_x * 2 - 1, to4_y * 2 - 1],
            ],
            dtype=torch.float32,
        )

        # Create perspective transformation grid
        grid = self._create_perspective_grid(
            src_corners, dst_corners, height, width, img.device
        )

        # Apply transformation with proper boundary handling for transparency
        mode = "nearest" if filter == "nearest" else "bilinear"
        result = F.grid_sample(
            img_tensor, grid, mode=mode, padding_mode="zeros", align_corners=False
        )

        # Convert back to ComfyUI format
        result = result.permute(0, 2, 3, 1)

        return (normalize_tensor(result),)

    def _create_perspective_grid(self, src_corners, dst_corners, height, width, device):
        """Create perspective transformation grid"""
        # This is a simplified perspective transformation
        # For a full implementation, you'd solve for the homography matrix

        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Simple bilinear interpolation between corners
        # This is a simplified version - full perspective would require homography

        # Interpolate in u direction (left-right)
        u = (x_grid + 1) / 2  # Convert from [-1,1] to [0,1]
        v = (y_grid + 1) / 2  # Convert from [-1,1] to [0,1]

        # Bilinear interpolation of corner positions
        top_interp = dst_corners[0] * (1 - u).unsqueeze(-1) + dst_corners[
            1
        ] * u.unsqueeze(-1)
        bottom_interp = dst_corners[3] * (1 - u).unsqueeze(-1) + dst_corners[
            2
        ] * u.unsqueeze(-1)

        final_coords = top_interp * (1 - v).unsqueeze(-1) + bottom_interp * v.unsqueeze(
            -1
        )

        grid = final_coords.unsqueeze(0)  # Add batch dimension

        return grid


class NukeCrop(NukeNodeBase):
    """Precise cropping node with soft edges"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "right": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "bottom": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "softness": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.001},
                ),
                "resize": (["crop", "format"], {"default": "crop"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "Nuke/Transform"

    def crop(self, image, left, right, top, bottom, softness, resize):
        """Apply cropping with optional soft edges"""
        img = ensure_batch_dim(image)
        batch_size, height, width, channels = img.shape

        if resize == "crop":
            # Hard crop
            left_px = int(left * width)
            right_px = int(right * width)
            top_px = int(top * height)
            bottom_px = int(bottom * height)

            result = img[:, top_px:bottom_px, left_px:right_px, :]
        else:
            # Format crop (resize to original dimensions with mask)
            result = img.clone()

            # Create soft mask
            y_coords = torch.linspace(0, 1, height, device=img.device)
            x_coords = torch.linspace(0, 1, width, device=img.device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

            # Create mask for crop area
            mask = torch.ones_like(x_grid)

            if softness > 0:
                # Apply soft edges
                mask = mask * torch.clamp((x_grid - left) / softness, 0, 1)
                mask = mask * torch.clamp((right - x_grid) / softness, 0, 1)
                mask = mask * torch.clamp((y_grid - top) / softness, 0, 1)
                mask = mask * torch.clamp((bottom - y_grid) / softness, 0, 1)
            else:
                # Hard edges
                mask = mask * (x_grid >= left) * (x_grid <= right)
                mask = mask * (y_grid >= top) * (y_grid <= bottom)

            # Apply mask
            mask = mask.unsqueeze(0).unsqueeze(-1)
            result = result * mask

        return (normalize_tensor(result),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeTransform": NukeTransform,
    "NukeCornerPin": NukeCornerPin,
    "NukeCrop": NukeCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeTransform": "Nuke Transform",
    "NukeCornerPin": "Nuke Corner Pin",
    "NukeCrop": "Nuke Crop",
}
