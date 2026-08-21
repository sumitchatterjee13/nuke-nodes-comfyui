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

Resampling goes through ``utils.remap_image``: for every OUTPUT pixel centre
(x + 0.5, y + 0.5) the inverse transform gives the SOURCE coordinate to sample,
in pixel units with centres at +0.5. An identity transform therefore
reproduces the input bit-exactly (no half-pixel drift).
"""

import logging
import math

import numpy as np
import torch

from .utils import (
    FILTER_NAMES,
    NukeNodeBase,
    ensure_batch_dim,
    identity_maps,
    normalize_tensor,
    remap_image,
)

logger = logging.getLogger(__name__)


def _to_rgba_numpy(img):
    """[B,H,W,3|4] tensor -> contiguous float32 [B,H,W,4] array (alpha = 1 if absent)."""
    arr = img.detach().cpu().numpy().astype(np.float32, copy=False)
    if arr.shape[3] == 3:
        alpha = np.ones(arr.shape[:3] + (1,), dtype=np.float32)
        arr = np.concatenate([arr, alpha], axis=3)
    elif arr.shape[3] > 4:
        arr = arr[:, :, :, :4]
    return np.ascontiguousarray(arr)


def _remap_batch(arr, map_x, map_y, filter_name):
    """Resample every item of a [B,H,W,C] array through the same source maps."""
    out = np.empty(
        (arr.shape[0], map_x.shape[0], map_x.shape[1], arr.shape[3]), dtype=np.float32
    )
    for i in range(arr.shape[0]):
        out[i] = remap_image(arr[i], map_x, map_y, filter_name, black_outside=True)
    return out


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
                "filter": (list(FILTER_NAMES), {"default": "cubic"}),
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
        Output is always RGBA; the alpha channel carries the coverage of the
        transformed image (1 inside, 0 outside).
        """
        img = ensure_batch_dim(image)
        batch_size, height, width, channels = img.shape

        # Handle center point - default to image center if -1
        actual_center_x = center_x if center_x >= 0 else width / 2
        actual_center_y = center_y if center_y >= 0 else height / 2

        # Combine uniform scale with individual scale
        final_scale_x = scale * scale_x
        final_scale_y = scale * scale_y

        # Forward matrix: source pixel coordinate -> output pixel coordinate
        matrix = self._create_transform_matrix(
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

        map_x, map_y = self._source_maps(matrix, height, width)

        arr = _to_rgba_numpy(img)
        out = _remap_batch(arr, map_x, map_y, filter)

        result = torch.from_numpy(out).to(img.device)
        return (normalize_tensor(result),)

    def _create_transform_matrix(
        self,
        tx,
        ty,
        rotate,
        sx,
        sy,
        skx,
        sky,
        skew_order,
        cx,
        cy,
        width,
        height,
        invert,
    ):
        """
        Create the forward 3x3 transformation matrix matching Nuke's order:
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
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)

        # Scale matrix
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)

        # Skew matrices
        SKX = np.array(
            [[1, math.tan(skew_x_rad), 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        SKY = np.array(
            [[1, 0, 0], [math.tan(skew_y_rad), 1, 0], [0, 0, 1]], dtype=np.float64
        )

        # Combined skew based on order
        if skew_order == "XY":
            SK = SKY @ SKX  # Apply X first, then Y
        else:
            SK = SKX @ SKY  # Apply Y first, then X

        # Rotation matrix: positive degrees rotate COUNTER-clockwise on screen
        # (Nuke convention). Image coordinates are y-down, so the standard
        # [[cos,-sin],[sin,cos]] form would appear clockwise - hence the
        # sign flip on sin.
        cos_r, sin_r = math.cos(rotate_rad), math.sin(rotate_rad)
        R = np.array(
            [[cos_r, sin_r, 0], [-sin_r, cos_r, 0], [0, 0, 1]], dtype=np.float64
        )

        # Translation back from center + user translation
        # Note: In Nuke, positive Y translation moves the image up
        # In image coordinates (top-left origin), we need to negate Y
        T2 = np.array([[1, 0, cx + tx], [0, 1, cy - ty], [0, 0, 1]], dtype=np.float64)

        # Combine transformations in Nuke's order:
        # T2 * R * SK * S * T1
        matrix = T2 @ R @ SK @ S @ T1

        if invert:
            matrix = self._safe_inverse(matrix, "invert")

        return matrix

    @staticmethod
    def _safe_inverse(matrix, what):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            logger.warning(
                f"[NukeTransform] Singular transform matrix ({what}); "
                f"falling back to the un-inverted matrix"
            )
            return matrix

    def _source_maps(self, matrix, height, width):
        """Source-coordinate maps (pixel units, centres at +0.5) for remap_image.

        For each output pixel centre, apply the inverse of the forward matrix
        to find where in the source to sample.
        """
        inv = self._safe_inverse(matrix, "sampling")
        xs, ys = identity_maps(height, width)  # output pixel centres
        xs64 = xs.astype(np.float64)
        ys64 = ys.astype(np.float64)
        src_x = inv[0, 0] * xs64 + inv[0, 1] * ys64 + inv[0, 2]
        src_y = inv[1, 0] * xs64 + inv[1, 1] * ys64 + inv[1, 2]
        return src_x.astype(np.float32), src_y.astype(np.float32)


class NukeCornerPin(NukeNodeBase):
    """
    Four-corner perspective transformation node with proper transparency.

    The four source image corners are pinned to the ``to1..to4`` destination
    corners (normalised 0..1, Nuke's bottom-left origin: to1 = bottom-left,
    to2 = bottom-right, to3 = top-right, to4 = top-left). A true homography
    is solved from the four corner pairs, inverted, and every output pixel
    centre is resampled from its source position through ``remap_image``.
    Pixels outside the pinned quad are transparent black.
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
                "filter": (list(FILTER_NAMES), {"default": "cubic"}),
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

        # Source corners in pixel coordinates (top-row-first arrays), in Nuke
        # order: bottom-left, bottom-right, top-right, top-left.
        src = np.array(
            [[0.0, height], [width, height], [width, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        )
        # Destination corners: normalised, bottom-left origin -> pixel coords
        dst = np.array(
            [
                [to1_x * width, (1.0 - to1_y) * height],
                [to2_x * width, (1.0 - to2_y) * height],
                [to3_x * width, (1.0 - to3_y) * height],
                [to4_x * width, (1.0 - to4_y) * height],
            ],
            dtype=np.float64,
        )

        arr = _to_rgba_numpy(img)

        homography = self._solve_homography(src, dst)
        if homography is None:
            logger.warning(
                "[NukeCornerPin] Degenerate corner configuration; "
                "returning the input unchanged"
            )
            result = torch.from_numpy(arr).to(img.device)
            return (normalize_tensor(result),)

        map_x, map_y = self._source_maps(homography, height, width)
        out = _remap_batch(arr, map_x, map_y, filter)

        result = torch.from_numpy(out).to(img.device)
        return (normalize_tensor(result),)

    @staticmethod
    def _solve_homography(src, dst):
        """3x3 homography H with dst ~ H @ src for the four corner pairs.

        Returns None when the system is singular (collinear/coincident corners).
        """
        if np.allclose(src, dst):
            return np.eye(3, dtype=np.float64)
        A = []
        b = []
        for (x, y), (u, v) in zip(src, dst):
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
            b.append(u)
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
            b.append(v)
        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        try:
            h = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(h)):
            return None
        return np.append(h, 1.0).reshape(3, 3)

    @staticmethod
    def _source_maps(homography, height, width):
        """Source-coordinate maps for remap_image via the inverse homography."""
        try:
            inv = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            inv = np.eye(3, dtype=np.float64)
        xs, ys = identity_maps(height, width)  # output pixel centres
        xs64 = xs.astype(np.float64)
        ys64 = ys.astype(np.float64)
        w = inv[2, 0] * xs64 + inv[2, 1] * ys64 + inv[2, 2]
        src_x = inv[0, 0] * xs64 + inv[0, 1] * ys64 + inv[0, 2]
        src_y = inv[1, 0] * xs64 + inv[1, 1] * ys64 + inv[1, 2]
        # Points with w <= 0 lie "behind" the projection; push them far
        # outside the source so they resolve to transparent black.
        valid = w > 1e-12
        safe_w = np.where(valid, w, 1.0)
        src_x = np.where(valid, src_x / safe_w, -1.0e6)
        src_y = np.where(valid, src_y / safe_w, -1.0e6)
        return src_x.astype(np.float32), src_y.astype(np.float32)


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
