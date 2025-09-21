"""
Merge and compositing nodes that replicate Nuke's merge functionality
"""

import numpy as np
import torch
import torch.nn.functional as F

from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor


class NukeMerge(NukeNodeBase):
    """
    Advanced merge node with multiple blend modes, similar to Nuke's Merge node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "operation": (
                    [
                        "over",
                        "under",
                        "plus",
                        "from",
                        "multiply",
                        "screen",
                        "overlay",
                        "soft_light",
                        "hard_light",
                        "color_dodge",
                        "color_burn",
                        "darken",
                        "lighten",
                        "difference",
                        "exclusion",
                        "average",
                        "subtract",
                        "divide",
                        "min",
                        "max",
                        "mask",
                        "stencil",
                        "hypot",
                        "in",
                        "out",
                        "atop",
                        "xor",
                        "conjoint_over",
                        "disjoint_over",
                        "copy",
                    ],
                    {"default": "over"},
                ),
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
    FUNCTION = "merge"
    CATEGORY = "Nuke/Merge"

    def merge(self, image_a, image_b, operation, mix, mask=None):
        """
        Merge two images using specified blend mode
        """
        # Ensure both images have batch dimension
        a = ensure_batch_dim(image_a)
        b = ensure_batch_dim(image_b)

        # Get dimensions
        batch_size = max(a.shape[0], b.shape[0])

        # Resize images to match if needed
        if a.shape[1:3] != b.shape[1:3]:
            target_h, target_w = max(a.shape[1], b.shape[1]), max(
                a.shape[2], b.shape[2]
            )
            a = F.interpolate(
                a.permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            b = F.interpolate(
                b.permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        # Extract alpha channels if they exist
        if a.shape[3] >= 4:
            a_alpha = a[:, :, :, 3:4]
            a_rgb = a[:, :, :, :3]
        else:
            a_alpha = torch.ones_like(a[:, :, :, :1])
            a_rgb = a

        if b.shape[3] >= 4:
            b_alpha = b[:, :, :, 3:4]
            b_rgb = b[:, :, :, :3]
        else:
            b_alpha = torch.ones_like(b[:, :, :, :1])
            b_rgb = b

        # Apply blend mode
        result_rgb = self._apply_blend_mode(a_rgb, b_rgb, operation)

        # Composite with alpha based on operation
        if operation == "over":
            # Standard over operation: A over B
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)
            result_rgb = (
                a_rgb * a_alpha + b_rgb * b_alpha * (1 - a_alpha)
            ) / torch.clamp(result_alpha, min=1e-7)
        elif operation == "under":
            # B under A (swap A and B)
            result_alpha = b_alpha + a_alpha * (1 - b_alpha)
            result_rgb = (
                b_rgb * b_alpha + a_rgb * a_alpha * (1 - b_alpha)
            ) / torch.clamp(result_alpha, min=1e-7)
        elif operation == "in":
            # A in B: show A where B has alpha
            result_alpha = a_alpha * b_alpha
            result_rgb = result_rgb * b_alpha
        elif operation == "out":
            # A out B: show A where B has no alpha
            result_alpha = a_alpha * (1 - b_alpha)
            result_rgb = result_rgb * (1 - b_alpha)
        elif operation == "atop":
            # A atop B: A over B, but only where B exists
            result_alpha = b_alpha
            result_rgb = (
                result_rgb * a_alpha + b_rgb * (1 - a_alpha)
            ) * b_alpha
        elif operation == "xor":
            # A xor B: A and B where they don't overlap
            result_alpha = a_alpha * (1 - b_alpha) + b_alpha * (1 - a_alpha)
            result_rgb = (
                a_rgb * a_alpha * (1 - b_alpha) + 
                b_rgb * b_alpha * (1 - a_alpha)
            ) / torch.clamp(result_alpha, min=1e-7)
        elif operation == "mask":
            # A masked by B: A where B has alpha
            result_alpha = a_alpha * b_alpha
            result_rgb = result_rgb
        elif operation == "stencil":
            # A stenciled by B: A where B has no alpha
            result_alpha = a_alpha * (1 - b_alpha)
            result_rgb = result_rgb
        elif operation == "conjoint_over":
            # Conjoint over: A over B with alpha clamping
            fa = torch.clamp(a_alpha, 0, 1 - b_alpha + 1e-7)
            result_alpha = fa + b_alpha
            result_rgb = (
                result_rgb * fa + b_rgb * b_alpha
            ) / torch.clamp(result_alpha, min=1e-7)
        elif operation == "disjoint_over":
            # Disjoint over: A over B with alpha separation
            fa = torch.clamp(a_alpha, 1 - b_alpha, 1)
            result_alpha = fa + b_alpha
            result_rgb = (
                result_rgb * fa + b_rgb * b_alpha
            ) / torch.clamp(result_alpha, min=1e-7)
        elif operation == "copy":
            # Copy A: just return A
            result_alpha = a_alpha
            result_rgb = result_rgb
        else:
            # For mathematical operations, use simple alpha blending
            result_alpha = torch.max(a_alpha, b_alpha)

        # Apply mask if provided
        if mask is not None:
            mask = ensure_batch_dim(mask)
            if mask.shape[1:3] != result_rgb.shape[1:3]:
                mask = F.interpolate(
                    mask.permute(0, 3, 1, 2),
                    size=result_rgb.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

            # Use first channel of mask as alpha
            mask_alpha = mask[:, :, :, :1]
            result_rgb = a_rgb + (result_rgb - a_rgb) * mask_alpha * mix
            result_alpha = a_alpha + (result_alpha - a_alpha) * mask_alpha * mix
        else:
            # Apply mix factor
            result_rgb = a_rgb + (result_rgb - a_rgb) * mix
            result_alpha = a_alpha + (result_alpha - a_alpha) * mix

        # Combine RGB and alpha
        if a.shape[3] >= 4 or b.shape[3] >= 4:
            result = torch.cat([result_rgb, result_alpha], dim=3)
        else:
            result = result_rgb

        return (normalize_tensor(result),)

    def _apply_blend_mode(self, a, b, mode):
        """Apply specified blend mode to RGB channels"""
        if mode == "over":
            return b  # Will be handled in compositing section
        elif mode == "under":
            return a  # B under A (swap inputs)
        elif mode == "plus":
            return a + b
        elif mode == "from":
            return torch.clamp(a - b, 0, 1)
        elif mode == "multiply":
            return a * b
        elif mode == "screen":
            return 1 - (1 - a) * (1 - b)
        elif mode == "overlay":
            return torch.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
        elif mode == "soft_light":
            return torch.where(
                b < 0.5,
                2 * a * b + a * a * (1 - 2 * b),
                2 * a * (1 - b) + torch.sqrt(a) * (2 * b - 1),
            )
        elif mode == "hard_light":
            return torch.where(b < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
        elif mode == "color_dodge":
            return torch.where(
                b >= 1, torch.ones_like(a), torch.clamp(a / (1 - b + 1e-7), 0, 1)
            )
        elif mode == "color_burn":
            return torch.where(
                b <= 0, torch.zeros_like(a), 1 - torch.clamp((1 - a) / (b + 1e-7), 0, 1)
            )
        elif mode == "darken":
            return torch.min(a, b)
        elif mode == "lighten":
            return torch.max(a, b)
        elif mode == "difference":
            return torch.abs(a - b)
        elif mode == "exclusion":
            return a + b - 2 * a * b
        elif mode == "average":
            return (a + b) / 2
        elif mode == "subtract":
            return torch.clamp(a - b, 0, 1)
        elif mode == "divide":
            return torch.clamp(a / (b + 1e-7), 0, 1)
        elif mode == "min":
            return torch.min(a, b)
        elif mode == "max":
            return torch.max(a, b)
        elif mode == "mask":
            return a  # Will be handled with alpha compositing
        elif mode == "stencil":
            return a  # Will be handled with alpha compositing
        elif mode == "hypot":
            return torch.sqrt(a * a + b * b)
        elif mode == "in":
            return a  # Will be handled with alpha compositing
        elif mode == "out":
            return a  # Will be handled with alpha compositing
        elif mode == "atop":
            return a  # Will be handled with alpha compositing
        elif mode == "xor":
            return a  # Will be handled with alpha compositing
        elif mode == "conjoint_over":
            return a  # Will be handled with alpha compositing
        elif mode == "disjoint_over":
            return a  # Will be handled with alpha compositing
        elif mode == "copy":
            return a
        else:
            return b


class NukeMix(NukeNodeBase):
    """
    Simple mix node for blending two images with a factor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mix": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix"
    CATEGORY = "Nuke/Merge"

    def mix(self, image_a, image_b, mix):
        """
        Linear mix between two images
        """
        a = ensure_batch_dim(image_a)
        b = ensure_batch_dim(image_b)

        # Resize if needed
        if a.shape[1:3] != b.shape[1:3]:
            target_h, target_w = max(a.shape[1], b.shape[1]), max(
                a.shape[2], b.shape[2]
            )
            a = F.interpolate(
                a.permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            b = F.interpolate(
                b.permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        result = a * (1 - mix) + b * mix
        return (normalize_tensor(result),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeMerge": NukeMerge,
    "NukeMix": NukeMix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMerge": "Nuke Merge",
    "NukeMix": "Nuke Mix",
}
