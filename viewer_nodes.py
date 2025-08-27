"""
Viewer and utility nodes that replicate Nuke's viewer functionality
"""

import numpy as np
import torch
import torch.nn.functional as F

from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor


class NukeViewer(NukeNodeBase):
    """
    Nuke-style viewer node with channel display options and shortcuts
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (
                    ["rgba", "rgb", "red", "green", "blue", "alpha", "luminance"],
                    {"default": "rgba"},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1},
                ),
                "gain": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
                ),
                "show_overlay": ("BOOLEAN", {"default": False}),
                "overlay_text": ("STRING", {"default": ""}),
            },
            "optional": {
                "mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "view"
    CATEGORY = "Nuke/Viewer"

    def view(self, image, channel, gamma, gain, show_overlay, overlay_text, mask=None):
        """
        Display image with channel selection and viewing controls
        """
        img = ensure_batch_dim(image)

        # Apply gamma and gain first
        img_processed = torch.pow(torch.clamp(img * gain, 0, 1), 1.0 / gamma)

        # Extract channels based on selection
        if img_processed.shape[3] >= 4:
            r = img_processed[:, :, :, 0:1]
            g = img_processed[:, :, :, 1:2]
            b = img_processed[:, :, :, 2:3]
            a = img_processed[:, :, :, 3:4]
        else:
            r = img_processed[:, :, :, 0:1]
            g = (
                img_processed[:, :, :, 1:2]
                if img_processed.shape[3] > 1
                else torch.zeros_like(r)
            )
            b = (
                img_processed[:, :, :, 2:3]
                if img_processed.shape[3] > 2
                else torch.zeros_like(r)
            )
            a = torch.ones_like(r)

        # Channel selection
        if channel == "rgba":
            if img_processed.shape[3] >= 4:
                result = img_processed
            else:
                result = torch.cat(
                    [img_processed, torch.ones_like(img_processed[:, :, :, :1])], dim=3
                )
        elif channel == "rgb":
            result = torch.cat([r, g, b], dim=3)
        elif channel == "red":
            result = torch.cat([r, r, r], dim=3)
        elif channel == "green":
            result = torch.cat([g, g, g], dim=3)
        elif channel == "blue":
            result = torch.cat([b, b, b], dim=3)
        elif channel == "alpha":
            result = torch.cat([a, a, a], dim=3)
        elif channel == "luminance":
            # Calculate luminance using standard weights
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            result = torch.cat([lum, lum, lum], dim=3)
        else:
            result = img_processed

        # Apply mask overlay if provided
        if mask is not None and show_overlay:
            mask = ensure_batch_dim(mask)
            if mask.shape[1:3] != result.shape[1:3]:
                mask = F.interpolate(
                    mask.permute(0, 3, 1, 2),
                    size=result.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

            # Create mask overlay (red tint where mask is active)
            mask_alpha = mask[:, :, :, :1]
            overlay_color = torch.tensor([1.0, 0.0, 0.0], device=result.device).view(
                1, 1, 1, 3
            )
            result = result + mask_alpha * overlay_color * 0.3

        return (normalize_tensor(result),)


class NukeChannelShuffle(NukeNodeBase):
    """
    Channel shuffle node for rearranging RGBA channels
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "red_from": (
                    ["red", "green", "blue", "alpha", "zero", "one"],
                    {"default": "red"},
                ),
                "green_from": (
                    ["red", "green", "blue", "alpha", "zero", "one"],
                    {"default": "green"},
                ),
                "blue_from": (
                    ["red", "green", "blue", "alpha", "zero", "one"],
                    {"default": "blue"},
                ),
                "alpha_from": (
                    ["red", "green", "blue", "alpha", "zero", "one"],
                    {"default": "alpha"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shuffle"
    CATEGORY = "Nuke/Viewer"

    def shuffle(self, image, red_from, green_from, blue_from, alpha_from):
        """
        Shuffle channels according to user selection
        """
        img = ensure_batch_dim(image)

        # Extract source channels
        if img.shape[3] >= 4:
            channels = {
                "red": img[:, :, :, 0:1],
                "green": img[:, :, :, 1:2],
                "blue": img[:, :, :, 2:3],
                "alpha": img[:, :, :, 3:4],
            }
        else:
            channels = {
                "red": img[:, :, :, 0:1],
                "green": (
                    img[:, :, :, 1:2]
                    if img.shape[3] > 1
                    else torch.zeros_like(img[:, :, :, :1])
                ),
                "blue": (
                    img[:, :, :, 2:3]
                    if img.shape[3] > 2
                    else torch.zeros_like(img[:, :, :, :1])
                ),
                "alpha": torch.ones_like(img[:, :, :, :1]),
            }

        channels["zero"] = torch.zeros_like(img[:, :, :, :1])
        channels["one"] = torch.ones_like(img[:, :, :, :1])

        # Build output channels
        out_r = channels[red_from]
        out_g = channels[green_from]
        out_b = channels[blue_from]
        out_a = channels[alpha_from]

        result = torch.cat([out_r, out_g, out_b, out_a], dim=3)

        return (normalize_tensor(result),)


class NukeRamp(NukeNodeBase):
    """
    Generate ramps and gradients for testing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "ramp_type": (
                    ["horizontal", "vertical", "radial", "diagonal", "checkerboard"],
                    {"default": "horizontal"},
                ),
                "color_start": ("STRING", {"default": "0,0,0"}),  # RGB values 0-1
                "color_end": ("STRING", {"default": "1,1,1"}),
                "invert": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Nuke/Viewer"

    def generate(
        self, width, height, ramp_type, color_start, color_end, invert, batch_size
    ):
        """
        Generate various types of ramps and patterns
        """
        # Parse color strings
        try:
            start_rgb = [float(x.strip()) for x in color_start.split(",")]
            end_rgb = [float(x.strip()) for x in color_end.split(",")]

            if len(start_rgb) != 3 or len(end_rgb) != 3:
                raise ValueError("Colors must have 3 values (R,G,B)")

        except:
            # Fallback to black and white
            start_rgb = [0.0, 0.0, 0.0]
            end_rgb = [1.0, 1.0, 1.0]

        # Create coordinate grids
        y_coords = torch.linspace(0, 1, height).view(-1, 1).expand(height, width)
        x_coords = torch.linspace(0, 1, width).view(1, -1).expand(height, width)

        # Generate ramp based on type
        if ramp_type == "horizontal":
            ramp = x_coords
        elif ramp_type == "vertical":
            ramp = y_coords
        elif ramp_type == "diagonal":
            ramp = (x_coords + y_coords) / 2
        elif ramp_type == "radial":
            center_x, center_y = width // 2, height // 2
            y_centered = (
                torch.arange(height, dtype=torch.float32).view(-1, 1) - center_y
            )
            x_centered = torch.arange(width, dtype=torch.float32).view(1, -1) - center_x
            distance = torch.sqrt(x_centered**2 + y_centered**2)
            max_distance = torch.sqrt(
                torch.tensor(center_x**2 + center_y**2, dtype=torch.float32)
            )
            ramp = distance / max_distance
            ramp = torch.clamp(ramp, 0, 1)
        elif ramp_type == "checkerboard":
            # Create checkerboard pattern
            checker_size = min(width, height) // 8
            x_checker = (x_coords * width // checker_size) % 2
            y_checker = (y_coords * height // checker_size) % 2
            ramp = (x_checker + y_checker) % 2
        else:
            ramp = x_coords

        # Invert if requested
        if invert:
            ramp = 1.0 - ramp

        # Convert to RGB
        ramp = ramp.unsqueeze(-1)  # Add channel dimension

        # Interpolate colors
        start_tensor = torch.tensor(start_rgb, dtype=torch.float32).view(1, 1, 3)
        end_tensor = torch.tensor(end_rgb, dtype=torch.float32).view(1, 1, 3)

        rgb = start_tensor + ramp * (end_tensor - start_tensor)

        # Add alpha channel
        alpha = torch.ones_like(rgb[:, :, :1])
        result = torch.cat([rgb, alpha], dim=2)

        # Add batch dimension and repeat for batch size
        result = result.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        return (normalize_tensor(result),)


class NukeColorBars(NukeNodeBase):
    """
    Generate standard color bars for testing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "pattern": (
                    ["smpte", "rgb_bars", "primary_colors", "grayscale"],
                    {"default": "smpte"},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_bars"
    CATEGORY = "Nuke/Viewer"

    def generate_bars(self, width, height, pattern, batch_size):
        """
        Generate color bar patterns
        """
        if pattern == "smpte":
            # SMPTE color bars
            colors = [
                [0.75, 0.75, 0.75],  # White (75%)
                [0.75, 0.75, 0.0],  # Yellow
                [0.0, 0.75, 0.75],  # Cyan
                [0.0, 0.75, 0.0],  # Green
                [0.75, 0.0, 0.75],  # Magenta
                [0.75, 0.0, 0.0],  # Red
                [0.0, 0.0, 0.75],  # Blue
                [0.0, 0.0, 0.0],  # Black
            ]
        elif pattern == "rgb_bars":
            # RGB primary bars
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 1.0, 1.0],  # White
                [0.0, 0.0, 0.0],  # Black
            ]
        elif pattern == "primary_colors":
            # Basic primary colors
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 1.0],  # White
                [0.0, 0.0, 0.0],  # Black
            ]
        elif pattern == "grayscale":
            # Grayscale steps
            colors = []
            for i in range(8):
                val = i / 7.0
                colors.append([val, val, val])
        else:
            colors = [[1.0, 1.0, 1.0]]  # Fallback to white

        # Create the pattern
        num_bars = len(colors)
        bar_width = width // num_bars

        result = torch.zeros(height, width, 3, dtype=torch.float32)

        for i, color in enumerate(colors):
            start_x = i * bar_width
            end_x = (i + 1) * bar_width if i < num_bars - 1 else width

            color_tensor = torch.tensor(color, dtype=torch.float32).view(1, 1, 3)
            result[:, start_x:end_x, :] = color_tensor

        # Add alpha channel
        alpha = torch.ones(height, width, 1, dtype=torch.float32)
        result = torch.cat([result, alpha], dim=2)

        # Add batch dimension and repeat
        result = result.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        return (normalize_tensor(result),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeViewer": NukeViewer,
    "NukeChannelShuffle": NukeChannelShuffle,
    "NukeRamp": NukeRamp,
    "NukeColorBars": NukeColorBars,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeViewer": "Nuke Viewer",
    "NukeChannelShuffle": "Nuke Channel Shuffle",
    "NukeRamp": "Nuke Ramp",
    "NukeColorBars": "Nuke Color Bars",
}
