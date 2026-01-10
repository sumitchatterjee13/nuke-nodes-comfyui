"""
Color space transformation nodes using OpenColorIO (OCIO).

=================================================================
USING: ACES 2.0 Studio Config (Hardcoded Colorspaces)
Config: studio-config-v4.0.0_aces-v2.0_ocio-v2.5
Total Colorspaces: 55 (All hardcoded for reliability)
=================================================================

Includes full camera IDT support:
- ARRI: LogC3 (EI800), LogC4, Linear ARRI Wide Gamut 3/4
- Sony: S-Log3 S-Gamut3, S-Log3 S-Gamut3.Cine, Venice variants
- RED: Log3G10 REDWideGamutRGB, Linear REDWideGamutRGB
- Canon: CanonLog2/3 CinemaGamut D55
- Panasonic: V-Log V-Gamut
- Blackmagic: BMDFilm WideGamut Gen5, DaVinci Intermediate WideGamut
- Apple: Apple Log
- DJI: D-Log D-Gamut

All colorspaces are hardcoded from ACES 2.0 Studio Config to ensure
reliability and eliminate the need for config path inputs.

Requires: pip install opencolorio (version 2.2+ recommended)
"""

import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor

# Try to import OpenColorIO
OCIO_AVAILABLE = False
OCIO_VERSION = "0.0.0"
OCIO_CONFIG = None  # Cached config loaded at startup

# Hardcoded ACES 2.0 Studio Config colorspaces (55 total)
# From: studio-config-v4.0.0_aces-v2.0_ocio-v2.5
ACES_STUDIO_COLORSPACES = [
    "ACES2065-1",
    "ACEScc",
    "ACEScct",
    "ACEScg",
    "ADX10",
    "ADX16",
    "ARRI LogC3 (EI800)",
    "ARRI LogC4",
    "Apple Log",
    "BMDFilm WideGamut Gen5",
    "Camera Rec.709",
    "CanonLog2 CinemaGamut D55",
    "CanonLog3 CinemaGamut D55",
    "D-Log D-Gamut",
    "DaVinci Intermediate WideGamut",
    "Display P3 - Display",
    "Display P3 HDR - Display",
    "Gamma 1.8 Encoded Rec.709",
    "Gamma 2.2 Encoded AP1",
    "Gamma 2.2 Encoded AdobeRGB",
    "Gamma 2.2 Encoded Rec.709",
    "Gamma 2.2 Rec.709 - Display",
    "Gamma 2.4 Encoded Rec.709",
    "Linear ARRI Wide Gamut 3",
    "Linear ARRI Wide Gamut 4",
    "Linear AdobeRGB",
    "Linear BMD WideGamut Gen5",
    "Linear CinemaGamut D55",
    "Linear D-Gamut",
    "Linear DaVinci WideGamut",
    "Linear P3-D65",
    "Linear REDWideGamutRGB",
    "Linear Rec.2020",
    "Linear Rec.709 (sRGB)",
    "Linear S-Gamut3",
    "Linear S-Gamut3.Cine",
    "Linear V-Gamut",
    "Linear Venice S-Gamut3",
    "Linear Venice S-Gamut3.Cine",
    "Log3G10 REDWideGamutRGB",
    "P3-D65 - Display",
    "Raw",
    "Rec.1886 Rec.709 - Display",
    "Rec.2100-HLG - Display",
    "Rec.2100-PQ - Display",
    "S-Log3 S-Gamut3",
    "S-Log3 S-Gamut3.Cine",
    "S-Log3 Venice S-Gamut3",
    "S-Log3 Venice S-Gamut3.Cine",
    "ST2084-P3-D65 - Display",
    "V-Log V-Gamut",
    "sRGB - Display",
    "sRGB Encoded AP1",
    "sRGB Encoded P3-D65",
    "sRGB Encoded Rec.709 (sRGB)",
]

try:
    import PyOpenColorIO as OCIO
    OCIO_AVAILABLE = True
    OCIO_VERSION = OCIO.GetVersion()
    print(f"[NukeOCIO] OpenColorIO version: {OCIO_VERSION}")
except ImportError:
    OCIO = None
    print("[NukeOCIO] OpenColorIO not installed. Install with: pip install opencolorio")

# Studio config names to try (in order of preference)
# These include all camera IDTs (ARRI, Sony, RED, Canon, etc.)
STUDIO_CONFIGS = [
    "studio-config-v4.0.0_aces-v2.0_ocio-v2.5",  # ACES 2.0 (OCIO 2.5+)
    "studio-config-v2.2.0_aces-v1.3_ocio-v2.4",  # ACES 1.3 (OCIO 2.4+)
    "studio-config-v2.1.0_aces-v1.3_ocio-v2.3",  # ACES 1.3 (OCIO 2.3+)
    "studio-config-v1.0.0_aces-v1.3_ocio-v2.1",  # ACES 1.3 (OCIO 2.1+)
]


def load_studio_config():
    """
    Load ACES Studio Config at startup.
    Tries multiple versions for compatibility with different OCIO versions.
    """
    global OCIO_CONFIG

    if not OCIO_AVAILABLE:
        return None

    # Try each Studio Config version until one works
    for config_name in STUDIO_CONFIGS:
        try:
            config = OCIO.Config.CreateFromBuiltinConfig(config_name)
            if config:
                OCIO_CONFIG = config
                print(f"[NukeOCIO] Loaded {config_name}")
                print(f"[NukeOCIO] Using hardcoded colorspaces: {len(ACES_STUDIO_COLORSPACES)} available")
                
                # Print some camera colorspaces to verify
                camera_cs = [cs for cs in ACES_STUDIO_COLORSPACES if any(kw in cs for kw in ['ARRI', 'Sony', 'RED', 'Canon', 'LogC', 'S-Log'])]
                if camera_cs:
                    print(f"[NukeOCIO] Camera colorspaces available: {len(camera_cs)}")

                return config
        except Exception as e:
            print(f"[NukeOCIO] Could not load {config_name}: {e}")
            continue

    print("[NukeOCIO] WARNING: Could not load any ACES Studio Config")
    return None


# Load config at module import time
if OCIO_AVAILABLE:
    load_studio_config()


def get_ocio_config(config_path: Optional[str] = None) -> Optional["OCIO.Config"]:
    """
    Get OCIO config. Uses cached Studio Config or loads from custom path.

    Args:
        config_path: Optional path to custom .ocio config file

    Returns:
        OCIO Config object or None
    """
    if not OCIO_AVAILABLE:
        return None

    # If custom config path provided, load it
    if config_path and os.path.exists(config_path):
        try:
            return OCIO.Config.CreateFromFile(config_path)
        except Exception as e:
            print(f"[NukeOCIO] Could not load custom config '{config_path}': {e}")

    # Return cached Studio Config
    return OCIO_CONFIG


def get_colorspace_names(config: "OCIO.Config") -> List[str]:
    """Get list of color space names from config."""
    if not config:
        return ACES_STUDIO_COLORSPACES
    try:
        # Use getColorSpaces() method for OCIO 2.x
        colorspaces = [cs.getName() for cs in config.getColorSpaces()]
        return colorspaces if colorspaces else ACES_STUDIO_COLORSPACES
    except:
        return ACES_STUDIO_COLORSPACES


def get_camera_colorspaces(colorspaces: List[str]) -> List[str]:
    """
    Filter colorspace list to return only camera-related colorspaces.

    Args:
        colorspaces: Full list of colorspaces

    Returns:
        List of camera-related colorspaces (ARRI, Sony, RED, Canon, etc.)
    """
    camera_keywords = [
        "ARRI", "LogC", "Wide Gamut 3", "Wide Gamut 4",
        "Sony", "S-Log", "S-Gamut",
        "RED", "Log3G10", "REDWideGamut",
        "Canon", "CanonLog", "CinemaGamut",
        "Panasonic", "V-Log", "V-Gamut",
        "Blackmagic", "BMD", "DaVinci",
        "Apple Log",
        "Fuji", "F-Log",
        "GoPro", "Protune",
        "DJI", "D-Log", "D-Gamut",
    ]

    camera_spaces = []
    for cs in colorspaces:
        for keyword in camera_keywords:
            if keyword.lower() in cs.lower():
                camera_spaces.append(cs)
                break

    return camera_spaces


def apply_ocio_transform(
    image_np: np.ndarray,
    src_colorspace: str,
    dst_colorspace: str,
    config: "OCIO.Config"
) -> np.ndarray:
    """
    Apply OCIO color space transformation to numpy image array.

    Args:
        image_np: Image as numpy array (H, W, C) in float32
        src_colorspace: Source color space name
        dst_colorspace: Destination color space name
        config: OCIO config object

    Returns:
        Transformed image as numpy array
    """
    if not OCIO_AVAILABLE or config is None:
        return image_np

    try:
        # Get processor for the transformation
        processor = config.getProcessor(src_colorspace, dst_colorspace)
        cpu_processor = processor.getDefaultCPUProcessor()

        # OCIO expects float32
        img = image_np.astype(np.float32)

        # Get original shape
        original_shape = img.shape
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1

        # Handle alpha channel separately if present
        if channels == 4:
            rgb = img[:, :, :3].copy()
            alpha = img[:, :, 3:4].copy()
        elif channels == 3:
            rgb = img.copy()
            alpha = None
        else:
            # Single channel, duplicate to RGB
            rgb = np.stack([img[:, :, 0]] * 3, axis=-1)
            alpha = None

        # Flatten for OCIO processing
        rgb_flat = rgb.reshape(-1, 3)

        # Apply transformation using OCIO's PackedImageDesc
        # Note: OCIO 2.5+ changed API - only one bitDepth parameter
        img_desc = OCIO.PackedImageDesc(
            rgb_flat,
            width,
            height,
            3,  # numChannels
            OCIO.BIT_DEPTH_F32,
            rgb_flat.strides[1],  # chanStrideBytes
            rgb_flat.strides[0],  # xStrideBytes
            width * rgb_flat.strides[0]  # yStrideBytes
        )

        cpu_processor.apply(img_desc)

        # Reshape back
        rgb_transformed = rgb_flat.reshape(height, width, 3)

        # Recombine with alpha if present
        if alpha is not None:
            result = np.concatenate([rgb_transformed, alpha], axis=-1)
        else:
            result = rgb_transformed

        return result

    except Exception as e:
        print(f"[NukeOCIO] Error applying transform: {e}")
        return image_np


# Common color spaces for quick access (when no config is loaded)
COMMON_COLORSPACES = [
    "ACES2065-1",
    "ACEScg",
    "ACEScct",
    "ACEScc",
    "Linear Rec.709 (sRGB)",
    "Linear Rec.2020",
    "Linear P3-D65",
    "sRGB - Texture",
    "sRGB",
    "Rec.709",
    "Rec.2020",
    "P3-D65",
    "Raw",
]


class NukeOCIOColorSpace(NukeNodeBase):
    """
    OCIO Color Space transformation node, similar to Nuke's OCIOColorSpace.

    Transforms images between color spaces using OpenColorIO.
    Supports ACES 2.0 (built-in) and custom OCIO configurations.

    Requirements:
        pip install opencolorio

    OpenColorIO 2.5+ includes built-in ACES configs:
    - ACES 2.0 CG Config (recommended for VFX/CG)
    - ACES 2.0 Studio Config (full camera/display support)
    - ACES 1.3 configs also available

    No external config files needed!
    """

    # Cache for config and color spaces
    _cached_config = None
    _cached_config_type = None
    _cached_colorspaces = None

    @classmethod
    def INPUT_TYPES(cls):
        # Use hardcoded ACES 2.0 Studio colorspaces
        colorspaces = ACES_STUDIO_COLORSPACES

        # Find good defaults for common workflow
        default_in = "ACEScg"
        default_out = "sRGB Encoded Rec.709 (sRGB)"

        return {
            "required": {
                "image": ("IMAGE",),
                "config": (["ACES 2.0 Studio Config"], {"default": "ACES 2.0 Studio Config"}),
                "in_colorspace": (colorspaces, {"default": default_in}),
                "out_colorspace": (colorspaces, {"default": default_out}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_colorspace"
    CATEGORY = "Nuke/Color"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def transform_colorspace(self, image, config, in_colorspace, out_colorspace):
        """Transform image from one color space to another using OCIO."""

        if not OCIO_AVAILABLE:
            print("[NukeOCIO] OpenColorIO not installed. Install with: pip install opencolorio")
            return (image,)

        # Skip if same color space
        if in_colorspace == out_colorspace:
            return (image,)

        # Use cached Studio Config
        config = OCIO_CONFIG

        if config is None:
            print("[NukeOCIO] No OCIO config available. Upgrade OpenColorIO: pip install opencolorio --upgrade")
            return (image,)

        # Ensure batch dimension
        img = ensure_batch_dim(image)
        batch_size = img.shape[0]

        # Process each image in batch
        results = []
        for i in range(batch_size):
            # Convert to numpy
            img_np = img[i].cpu().numpy()

            # Apply OCIO transform
            transformed = apply_ocio_transform(
                img_np,
                in_colorspace,
                out_colorspace,
                config
            )

            results.append(transformed)

        # Stack results back to batch
        result_np = np.stack(results, axis=0)

        # Convert back to torch tensor
        result = torch.from_numpy(result_np).to(image.device)

        return (result,)


class NukeOCIODisplay(NukeNodeBase):
    """
    OCIO Display/View transformation node.

    Applies a display transform (similar to Nuke's viewer process).
    Useful for converting scene-referred images to display-referred for viewing.

    Supports ACES 2.0 built-in configs (no external config files needed).

    The 'invert' parameter allows round-tripping:
    - forward: linear -> tonemapped (for viewing)
    - inverse: tonemapped -> linear (to reverse a display transform)
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Use hardcoded colorspaces and get displays/views from cached config
        displays = ["sRGB - Display", "Rec.1886 Rec.709 - Display", "P3-D65 - Display", "Rec.2100-PQ - Display"]
        views = ["ACES 2.0 - SDR Video", "Raw"]
        input_colorspaces = ACES_STUDIO_COLORSPACES

        if OCIO_CONFIG:
            try:
                displays = list(OCIO_CONFIG.getDisplays())
                if displays:
                    views = list(OCIO_CONFIG.getViews(displays[0]))
            except:
                pass

        return {
            "required": {
                "image": ("IMAGE",),
                "config": (["ACES 2.0 Studio Config"], {"default": "ACES 2.0 Studio Config"}),
                "display": (displays, {"default": displays[0] if displays else "sRGB - Display"}),
                "view": (views, {"default": views[0] if views else "ACES 2.0 - SDR Video"}),
                "input_colorspace": (input_colorspaces, {"default": "ACEScg"}),
                "invert": (["forward", "inverse"], {"default": "forward"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_display"
    CATEGORY = "Nuke/Color"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply_display(self, image, config, display, view, input_colorspace, invert="forward"):
        """Apply display/view transform using OCIO.

        Args:
            invert: Direction of transform. "forward" applies the display transform
                   (linear -> tonemapped), "inverse" reverses it (tonemapped -> linear).
        """

        if not OCIO_AVAILABLE:
            print("[NukeOCIO] OpenColorIO not installed. Install with: pip install opencolorio")
            return (image,)

        # Use cached Studio Config
        config = OCIO_CONFIG

        if config is None:
            print("[NukeOCIO] No OCIO config available.")
            return (image,)

        try:
            # Create display transform
            transform = OCIO.DisplayViewTransform()
            transform.setSrc(input_colorspace)
            transform.setDisplay(display)
            transform.setView(view)

            # Set transform direction
            direction = OCIO.TRANSFORM_DIR_INVERSE if invert == "inverse" else OCIO.TRANSFORM_DIR_FORWARD
            processor = config.getProcessor(transform, direction)
            cpu_processor = processor.getDefaultCPUProcessor()

            # Ensure batch dimension
            img = ensure_batch_dim(image)
            batch_size = img.shape[0]

            results = []
            for i in range(batch_size):
                img_np = img[i].cpu().numpy().astype(np.float32)
                height, width = img_np.shape[:2]
                channels = img_np.shape[2] if len(img_np.shape) > 2 else 1

                # Handle alpha
                if channels == 4:
                    rgb = img_np[:, :, :3].copy()
                    alpha = img_np[:, :, 3:4].copy()
                else:
                    rgb = img_np[:, :, :3].copy() if channels >= 3 else np.stack([img_np] * 3, axis=-1)
                    alpha = None

                # Flatten and apply
                rgb_flat = rgb.reshape(-1, 3)

                img_desc = OCIO.PackedImageDesc(
                    rgb_flat,
                    width,
                    height,
                    3,
                    OCIO.BIT_DEPTH_F32,
                    rgb_flat.strides[1],
                    rgb_flat.strides[0],
                    width * rgb_flat.strides[0]
                )

                cpu_processor.apply(img_desc)

                rgb_transformed = rgb_flat.reshape(height, width, 3)

                if alpha is not None:
                    result = np.concatenate([rgb_transformed, alpha], axis=-1)
                else:
                    result = rgb_transformed

                results.append(result)

            result_np = np.stack(results, axis=0)
            result = torch.from_numpy(result_np).to(image.device)

            return (result,)

        except Exception as e:
            print(f"[NukeOCIO] Error applying display transform: {e}")
            return (image,)


class NukeOCIOInfo(NukeNodeBase):
    """
    OCIO Info node - displays information about the current OCIO configuration.

    Useful for debugging and understanding what color spaces are available.
    Uses ACES 2.0 Studio Config with full camera support.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "Nuke/Color"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def get_info(self):
        """Get OCIO configuration information."""

        if not OCIO_AVAILABLE:
            return ("OpenColorIO not installed.\n\nInstall with:\npip install opencolorio",)

        # Use cached Studio Config
        config = OCIO_CONFIG

        if config is None:
            info = "No OCIO config loaded.\n\n"
            info += "OpenColorIO 2.2+ includes built-in ACES configs.\n"
            info += "Make sure you have opencolorio >= 2.2 installed:\n"
            info += "  pip install opencolorio --upgrade\n"
            return (info,)

        try:
            info = f"OCIO Version: {OCIO_VERSION}\n"
            info += f"Config: ACES 2.0 Studio Config (Hardcoded)\n"
            info += f"Description: {config.getDescription()}\n\n"

            # Color spaces (hardcoded)
            info += f"Color Spaces ({len(ACES_STUDIO_COLORSPACES)}):\n"
            for cs in ACES_STUDIO_COLORSPACES[:20]:  # Limit to first 20
                info += f"  - {cs}\n"
            if len(ACES_STUDIO_COLORSPACES) > 20:
                info += f"  ... and {len(ACES_STUDIO_COLORSPACES) - 20} more\n"

            info += "\n"

            # Camera colorspaces
            camera_spaces = get_camera_colorspaces(ACES_STUDIO_COLORSPACES)
            if camera_spaces:
                info += f"Camera Colorspaces ({len(camera_spaces)}):\n"
                for cs in camera_spaces:
                    info += f"  - {cs}\n"
                info += "\n"

            # Displays
            displays = list(config.getDisplays())
            info += f"Displays ({len(displays)}):\n"
            for d in displays:
                info += f"  - {d}\n"
                views = list(config.getViews(d))
                for v in views[:5]:
                    info += f"      View: {v}\n"
                if len(views) > 5:
                    info += f"      ... and {len(views) - 5} more views\n"

            # Show available built-in configs
            info += "\n=== Using ACES 2.0 Studio Config ===\n"
            info += "Hardcoded colorspaces from:\n"
            info += "  studio-config-v4.0.0_aces-v2.0_ocio-v2.5\n"
            info += "\nIncludes all camera IDTs:\n"
            info += "  - ARRI LogC3/4, Linear Wide Gamut 3/4\n"
            info += "  - Sony S-Log3 S-Gamut3/Cine\n"
            info += "  - RED Log3G10, Linear REDWideGamutRGB\n"
            info += "  - Canon CanonLog2/3 CinemaGamut\n"
            info += "  - Panasonic V-Log V-Gamut\n"
            info += "  - Blackmagic BMDFilm, DaVinci Intermediate\n"
            info += "  - Apple Log\n"
            info += "  - DJI D-Log D-Gamut\n"

            return (info,)

        except Exception as e:
            return (f"Error reading config: {e}",)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeOCIOColorSpace": NukeOCIOColorSpace,
    "NukeOCIODisplay": NukeOCIODisplay,
    "NukeOCIOInfo": NukeOCIOInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeOCIOColorSpace": "Nuke OCIO ColorSpace",
    "NukeOCIODisplay": "Nuke OCIO Display",
    "NukeOCIOInfo": "Nuke OCIO Info",
}
