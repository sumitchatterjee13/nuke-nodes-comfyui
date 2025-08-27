"""
Color grading and correction nodes that replicate Nuke's color tools
"""

import numpy as np
import torch
import torch.nn.functional as F

from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor


class NukeGrade(NukeNodeBase):
    """
    Professional color grading node with lift, gamma, gain controls (similar to Nuke's Grade node)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Master controls (apply to all channels)
                "lift": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01},
                ),
                "gain": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                # Global controls
                "multiply": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "mix": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "mask": ("IMAGE",),
                # Individual channel adjustments (collapsed by default)
                "lift_r_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "lift_g_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "lift_b_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "gamma_r_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "gamma_g_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "gamma_b_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "gain_r_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "gain_g_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "gain_b_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grade"
    CATEGORY = "Nuke/Color"

    def grade(
        self,
        image,
        lift,
        gamma,
        gain,
        multiply,
        offset,
        mix,
        mask=None,
        lift_r_offset=0.0,
        lift_g_offset=0.0,
        lift_b_offset=0.0,
        gamma_r_offset=0.0,
        gamma_g_offset=0.0,
        gamma_b_offset=0.0,
        gain_r_offset=0.0,
        gain_g_offset=0.0,
        gain_b_offset=0.0,
    ):
        """
        Apply professional color grading using master lift/gamma/gain controls with optional per-channel offsets
        """
        img = ensure_batch_dim(image)

        # Separate RGB and alpha channels
        if img.shape[3] >= 4:
            rgb = img[:, :, :, :3]
            alpha = img[:, :, :, 3:]
        else:
            rgb = img
            alpha = None

        # Calculate final values: master + per-channel offset
        final_lift_r = lift + lift_r_offset
        final_lift_g = lift + lift_g_offset
        final_lift_b = lift + lift_b_offset
        
        final_gamma_r = gamma + gamma_r_offset
        final_gamma_g = gamma + gamma_g_offset
        final_gamma_b = gamma + gamma_b_offset
        
        final_gain_r = gain + gain_r_offset
        final_gain_g = gain + gain_g_offset
        final_gain_b = gain + gain_b_offset

        # Create color correction vectors
        lift_vec = torch.tensor(
            [final_lift_r, final_lift_g, final_lift_b], device=rgb.device, dtype=rgb.dtype
        ).view(1, 1, 1, 3)
        gamma_vec = torch.tensor(
            [final_gamma_r, final_gamma_g, final_gamma_b], device=rgb.device, dtype=rgb.dtype
        ).view(1, 1, 1, 3)
        gain_vec = torch.tensor(
            [final_gain_r, final_gain_g, final_gain_b], device=rgb.device, dtype=rgb.dtype
        ).view(1, 1, 1, 3)

        # Apply lift/gamma/gain formula: ((rgb + lift) ^ (1/gamma)) * gain
        # Ensure positive values for gamma correction
        rgb_lifted = torch.clamp(rgb + lift_vec, min=1e-7)

        # Apply gamma correction
        rgb_gamma = torch.pow(rgb_lifted, 1.0 / torch.clamp(gamma_vec, min=0.1))

        # Apply gain
        rgb_graded = rgb_gamma * gain_vec

        # Apply master multiply and offset
        rgb_graded = rgb_graded * multiply + offset

        # Apply mask if provided
        if mask is not None:
            mask = ensure_batch_dim(mask)
            if mask.shape[1:3] != rgb.shape[1:3]:
                mask = F.interpolate(
                    mask.permute(0, 3, 1, 2),
                    size=rgb.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

            mask_alpha = mask[:, :, :, :1]
            rgb_graded = rgb + (rgb_graded - rgb) * mask_alpha * mix
        else:
            rgb_graded = rgb + (rgb_graded - rgb) * mix

        # Recombine with alpha
        if alpha is not None:
            result = torch.cat([rgb_graded, alpha], dim=3)
        else:
            result = rgb_graded

        return (normalize_tensor(result),)


class NukeColorCorrect(NukeNodeBase):
    """
    HSV-based color correction node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "value": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "mix": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_correct"
    CATEGORY = "Nuke/Color"

    def color_correct(self, image, hue, saturation, value, contrast, mix):
        """
        Apply HSV-based color correction
        """
        img = ensure_batch_dim(image)

        # Separate RGB and alpha
        if img.shape[3] >= 4:
            rgb = img[:, :, :, :3]
            alpha = img[:, :, :, 3:]
        else:
            rgb = img
            alpha = None

        # Convert RGB to HSV
        hsv = self._rgb_to_hsv(rgb)

        # Apply adjustments
        # Hue shift
        hsv[:, :, :, 0] = (hsv[:, :, :, 0] + hue / 360.0) % 1.0

        # Saturation adjustment
        hsv[:, :, :, 1] = torch.clamp(hsv[:, :, :, 1] * saturation, 0.0, 1.0)

        # Value adjustment
        hsv[:, :, :, 2] = torch.clamp(hsv[:, :, :, 2] * value, 0.0, 1.0)

        # Convert back to RGB
        rgb_corrected = self._hsv_to_rgb(hsv)

        # Apply contrast
        rgb_corrected = torch.clamp((rgb_corrected - 0.5) * contrast + 0.5, 0.0, 1.0)

        # Mix with original
        rgb_corrected = rgb + (rgb_corrected - rgb) * mix

        # Recombine with alpha
        if alpha is not None:
            result = torch.cat([rgb_corrected, alpha], dim=3)
        else:
            result = rgb_corrected

        return (normalize_tensor(result),)

    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space"""
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]

        max_val = torch.max(torch.max(r, g), b)
        min_val = torch.min(torch.min(r, g), b)
        delta = max_val - min_val

        # Hue calculation
        hue = torch.zeros_like(max_val)
        mask = delta != 0

        # Red is max
        red_mask = (max_val == r) & mask
        hue[red_mask] = ((g[red_mask] - b[red_mask]) / delta[red_mask]) % 6

        # Green is max
        green_mask = (max_val == g) & mask
        hue[green_mask] = (b[green_mask] - r[green_mask]) / delta[green_mask] + 2

        # Blue is max
        blue_mask = (max_val == b) & mask
        hue[blue_mask] = (r[blue_mask] - g[blue_mask]) / delta[blue_mask] + 4

        hue = hue / 6.0

        # Saturation
        saturation = torch.where(
            max_val != 0, delta / max_val, torch.zeros_like(max_val)
        )

        # Value
        value = max_val

        return torch.stack([hue, saturation, value], dim=3)

    def _hsv_to_rgb(self, hsv):
        """Convert HSV to RGB color space"""
        h, s, v = hsv[:, :, :, 0], hsv[:, :, :, 1], hsv[:, :, :, 2]

        h = h * 6.0
        i = torch.floor(h).long()
        f = h - i

        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        rgb = torch.zeros_like(hsv)

        # i == 0
        mask = (i % 6) == 0
        rgb[mask, 0] = v[mask]
        rgb[mask, 1] = t[mask]
        rgb[mask, 2] = p[mask]

        # i == 1
        mask = (i % 6) == 1
        rgb[mask, 0] = q[mask]
        rgb[mask, 1] = v[mask]
        rgb[mask, 2] = p[mask]

        # i == 2
        mask = (i % 6) == 2
        rgb[mask, 0] = p[mask]
        rgb[mask, 1] = v[mask]
        rgb[mask, 2] = t[mask]

        # i == 3
        mask = (i % 6) == 3
        rgb[mask, 0] = p[mask]
        rgb[mask, 1] = q[mask]
        rgb[mask, 2] = v[mask]

        # i == 4
        mask = (i % 6) == 4
        rgb[mask, 0] = t[mask]
        rgb[mask, 1] = p[mask]
        rgb[mask, 2] = v[mask]

        # i == 5
        mask = (i % 6) == 5
        rgb[mask, 0] = v[mask]
        rgb[mask, 1] = p[mask]
        rgb[mask, 2] = q[mask]

        return rgb


class NukeLevels(NukeNodeBase):
    """
    Input/output levels adjustment similar to Nuke's ColorLookup node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_black": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "input_white": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01},
                ),
                "output_black": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "output_white": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mix": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "levels"
    CATEGORY = "Nuke/Color"

    def levels(
        self, image, input_black, input_white, gamma, output_black, output_white, mix
    ):
        """
        Apply levels adjustment
        """
        img = ensure_batch_dim(image)

        # Separate RGB and alpha
        if img.shape[3] >= 4:
            rgb = img[:, :, :, :3]
            alpha = img[:, :, :, 3:]
        else:
            rgb = img
            alpha = None

        # Apply input levels
        input_range = input_white - input_black
        input_range = torch.clamp(input_range, min=1e-7)  # Avoid division by zero

        rgb_normalized = torch.clamp((rgb - input_black) / input_range, 0.0, 1.0)

        # Apply gamma
        rgb_gamma = torch.pow(rgb_normalized, 1.0 / torch.clamp(gamma, min=0.1))

        # Apply output levels
        output_range = output_white - output_black
        rgb_final = rgb_gamma * output_range + output_black

        # Mix with original
        rgb_final = rgb + (rgb_final - rgb) * mix

        # Recombine with alpha
        if alpha is not None:
            result = torch.cat([rgb_final, alpha], dim=3)
        else:
            result = rgb_final

        return (normalize_tensor(result),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeGrade": NukeGrade,
    "NukeColorCorrect": NukeColorCorrect,
    "NukeLevels": NukeLevels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeGrade": "Nuke Grade",
    "NukeColorCorrect": "Nuke Color Correct",
    "NukeLevels": "Nuke Levels",
}
