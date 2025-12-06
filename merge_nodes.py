"""
Merge and compositing nodes that replicate Nuke's merge functionality.

In Nuke's Merge node:
- A input is the foreground (top layer)
- B input is the background (bottom layer)
- "A over B" means A composited on top of B

All Porter-Duff operations follow standard compositing formulas.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor


class NukeMerge(NukeNodeBase):
    """
    Advanced merge node with multiple blend modes, matching Nuke's Merge node behavior.

    A = Foreground (top layer)
    B = Background (bottom layer)

    For "over" operation: A is composited over B (A on top of B)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "A": ("IMAGE",),  # Foreground
                "B": ("IMAGE",),  # Background
                "operation": (
                    [
                        "over",
                        "under",
                        "plus",
                        "minus",
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
                        "divide",
                        "min",
                        "max",
                        "mask",
                        "stencil",
                        "matte",
                        "hypot",
                        "in",
                        "out",
                        "atop",
                        "xor",
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
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Nuke/Merge"

    def merge(self, A, B, operation, mix, mask=None):
        """
        Merge two images using specified blend mode.

        A = Foreground (the layer on top)
        B = Background (the layer behind)

        For "over": Result = A over B (A composited on top of B)
        """
        # Ensure both images have batch dimension
        a = ensure_batch_dim(A)
        b = ensure_batch_dim(B)

        # Resize images to match if needed (use B's size as reference, like Nuke)
        if a.shape[1:3] != b.shape[1:3]:
            target_h, target_w = b.shape[1], b.shape[2]
            a = F.interpolate(
                a.permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        # Extract RGB and alpha channels
        # A = foreground
        if a.shape[3] >= 4:
            a_alpha = a[:, :, :, 3:4]
            a_rgb = a[:, :, :, :3]
        else:
            a_alpha = torch.ones_like(a[:, :, :, :1])
            a_rgb = a[:, :, :, :3] if a.shape[3] >= 3 else a.repeat(1, 1, 1, 3)

        # B = background
        if b.shape[3] >= 4:
            b_alpha = b[:, :, :, 3:4]
            b_rgb = b[:, :, :, :3]
        else:
            b_alpha = torch.ones_like(b[:, :, :, :1])
            b_rgb = b[:, :, :, :3] if b.shape[3] >= 3 else b.repeat(1, 1, 1, 3)

        # Apply the merge operation
        # All formulas follow Nuke/Porter-Duff conventions
        if operation == "over":
            # A over B: A composited on top of B
            # Formula: A + B * (1 - Aα)
            result_rgb = a_rgb * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "under":
            # A under B: equivalent to B over A
            # Formula: B + A * (1 - Bα)
            result_rgb = b_rgb * b_alpha + a_rgb * (1 - b_alpha)
            result_alpha = b_alpha + a_alpha * (1 - b_alpha)

        elif operation == "plus":
            # Additive blend: A + B
            result_rgb = a_rgb + b_rgb
            result_alpha = a_alpha + b_alpha

        elif operation == "minus":
            # Subtractive blend: B - A (Nuke convention)
            result_rgb = b_rgb - a_rgb
            result_alpha = b_alpha

        elif operation == "multiply":
            # Multiply: A * B, composited over B
            blended = a_rgb * b_rgb
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "screen":
            # Screen: 1 - (1-A) * (1-B), composited over B
            blended = 1 - (1 - a_rgb) * (1 - b_rgb)
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "overlay":
            # Overlay blend mode
            blended = torch.where(
                b_rgb < 0.5,
                2 * a_rgb * b_rgb,
                1 - 2 * (1 - a_rgb) * (1 - b_rgb)
            )
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "soft_light":
            # Soft light blend
            blended = torch.where(
                a_rgb < 0.5,
                b_rgb - (1 - 2 * a_rgb) * b_rgb * (1 - b_rgb),
                b_rgb + (2 * a_rgb - 1) * (torch.sqrt(b_rgb) - b_rgb)
            )
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "hard_light":
            # Hard light blend
            blended = torch.where(
                a_rgb < 0.5,
                2 * a_rgb * b_rgb,
                1 - 2 * (1 - a_rgb) * (1 - b_rgb)
            )
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "color_dodge":
            # Color dodge
            blended = torch.where(
                a_rgb >= 1,
                torch.ones_like(b_rgb),
                torch.clamp(b_rgb / (1 - a_rgb + 1e-7), 0, 1)
            )
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "color_burn":
            # Color burn
            blended = torch.where(
                a_rgb <= 0,
                torch.zeros_like(b_rgb),
                1 - torch.clamp((1 - b_rgb) / (a_rgb + 1e-7), 0, 1)
            )
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "darken":
            # Darken: min(A, B)
            blended = torch.min(a_rgb, b_rgb)
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "lighten":
            # Lighten: max(A, B)
            blended = torch.max(a_rgb, b_rgb)
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "difference":
            # Difference: |A - B|
            blended = torch.abs(a_rgb - b_rgb)
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "exclusion":
            # Exclusion: A + B - 2*A*B
            blended = a_rgb + b_rgb - 2 * a_rgb * b_rgb
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "average":
            # Average: (A + B) / 2
            blended = (a_rgb + b_rgb) / 2
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "divide":
            # Divide: B / A
            blended = torch.clamp(b_rgb / (a_rgb + 1e-7), 0, 1)
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "min":
            # Min: min(A, B)
            blended = torch.min(a_rgb, b_rgb)
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "max":
            # Max: max(A, B)
            blended = torch.max(a_rgb, b_rgb)
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        elif operation == "hypot":
            # Hypotenuse: sqrt(A² + B²)
            blended = torch.clamp(torch.sqrt(a_rgb * a_rgb + b_rgb * b_rgb), 0, 1)
            result_rgb = blended * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        # Porter-Duff operations
        elif operation == "in":
            # A in B: A masked by B's alpha
            result_rgb = a_rgb * b_alpha
            result_alpha = a_alpha * b_alpha

        elif operation == "out":
            # A out B: A where B is transparent
            result_rgb = a_rgb * (1 - b_alpha)
            result_alpha = a_alpha * (1 - b_alpha)

        elif operation == "atop":
            # A atop B: A where B exists, B elsewhere
            result_rgb = a_rgb * a_alpha * b_alpha + b_rgb * (1 - a_alpha)
            result_alpha = b_alpha

        elif operation == "xor":
            # A xor B: A and B where they don't overlap
            result_rgb = a_rgb * (1 - b_alpha) + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha * (1 - b_alpha) + b_alpha * (1 - a_alpha)

        elif operation == "mask":
            # Mask: Use A's color with A's alpha multiplied by B's alpha
            result_rgb = a_rgb
            result_alpha = a_alpha * b_alpha

        elif operation == "stencil":
            # Stencil: A where B is transparent
            result_rgb = a_rgb
            result_alpha = a_alpha * (1 - b_alpha)

        elif operation == "matte":
            # Matte: B with A's alpha as matte
            result_rgb = b_rgb * a_alpha
            result_alpha = b_alpha * a_alpha

        elif operation == "copy":
            # Copy: Just A
            result_rgb = a_rgb
            result_alpha = a_alpha

        else:
            # Default to over
            result_rgb = a_rgb * a_alpha + b_rgb * (1 - a_alpha)
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)

        # Apply mask if provided
        if mask is not None:
            # ComfyUI MASK type is typically (B, H, W) - 3D tensor
            # Handle different mask formats
            if len(mask.shape) == 2:
                # (H, W) -> (1, H, W, 1)
                mask_alpha = mask.unsqueeze(0).unsqueeze(-1)
            elif len(mask.shape) == 3:
                # (B, H, W) -> (B, H, W, 1)
                mask_alpha = mask.unsqueeze(-1)
            elif len(mask.shape) == 4:
                # (B, H, W, C) - take first channel
                mask_alpha = mask[:, :, :, :1]
            else:
                mask_alpha = mask

            # Resize mask if needed
            if mask_alpha.shape[1:3] != result_rgb.shape[1:3]:
                # Need to permute for interpolate: (B, H, W, 1) -> (B, 1, H, W)
                mask_alpha = mask_alpha.permute(0, 3, 1, 2)
                mask_alpha = F.interpolate(
                    mask_alpha,
                    size=(result_rgb.shape[1], result_rgb.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                )
                # Back to (B, H, W, 1)
                mask_alpha = mask_alpha.permute(0, 2, 3, 1)

            # Blend between B (original) and result based on mask
            effective_mix = mask_alpha * mix
            result_rgb = b_rgb + (result_rgb - b_rgb) * effective_mix
            result_alpha = b_alpha + (result_alpha - b_alpha) * effective_mix
        else:
            # Apply mix factor - blend between B and result
            if mix < 1.0:
                result_rgb = b_rgb + (result_rgb - b_rgb) * mix
                result_alpha = b_alpha + (result_alpha - b_alpha) * mix

        # Clamp results
        result_rgb = torch.clamp(result_rgb, 0, 1)
        result_alpha = torch.clamp(result_alpha, 0, 1)

        # Combine RGB and alpha
        if A.shape[-1] >= 4 or B.shape[-1] >= 4:
            result = torch.cat([result_rgb, result_alpha], dim=3)
        else:
            result = result_rgb

        return (result,)


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


class NukeConstant(NukeNodeBase):
    """
    Constant node that generates a solid color image, similar to Nuke's Constant node.

    Creates a solid color image with configurable RGBA values, width, and height.
    The output can be used as a background, matte, or connected to any node expecting an image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "red": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "green": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "blue": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "output_alpha": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
            "optional": {
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 64, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Nuke/Generate"

    def generate(self, width, height, red, green, blue, alpha, output_alpha, batch_size=1):
        """
        Generate a solid color image with the specified RGBA values.

        Args:
            output_alpha: If True, output RGBA (4 channels). If False, output RGB (3 channels).
                         RGB is more compatible with other ComfyUI nodes.

        Returns:
            Tensor of shape (batch_size, height, width, 3 or 4)
        """
        if output_alpha:
            # Shape: (batch_size, height, width, 4) for RGBA
            constant = torch.zeros((batch_size, height, width, 4), dtype=torch.float32)
            constant[:, :, :, 0] = red
            constant[:, :, :, 1] = green
            constant[:, :, :, 2] = blue
            constant[:, :, :, 3] = alpha
        else:
            # Shape: (batch_size, height, width, 3) for RGB - more compatible
            constant = torch.zeros((batch_size, height, width, 3), dtype=torch.float32)
            constant[:, :, :, 0] = red
            constant[:, :, :, 1] = green
            constant[:, :, :, 2] = blue

        return (constant,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeMerge": NukeMerge,
    "NukeMix": NukeMix,
    "NukeConstant": NukeConstant,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMerge": "Nuke Merge",
    "NukeMix": "Nuke Mix",
    "NukeConstant": "Nuke Constant",
}
