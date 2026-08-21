"""
Shared OpenColorIO configuration for all Nuke nodes.

Resolution order (mirrors Nuke):
  1. ``$OCIO`` environment variable pointing at a .ocio file (show / studio
     config) - resolved once when ComfyUI starts.
  2. OCIO's built-in ACES Studio configs, newest first.

Every colour-aware node (OCIO nodes, NukeRead / NukeWrite) takes its
colourspace, role, display and view lists from here, so the dropdowns always
reflect the config actually in use. Because ComfyUI evaluates INPUT_TYPES at
startup, changing ``$OCIO`` requires a ComfyUI restart.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

OCIO_AVAILABLE = False
OCIO_VERSION = "0.0.0"
try:
    import PyOpenColorIO as OCIO

    OCIO_AVAILABLE = True
    OCIO_VERSION = OCIO.GetVersion()
    logger.info(f"[NukeOCIO] OpenColorIO version: {OCIO_VERSION}")
except ImportError:  # pragma: no cover - depends on the host environment
    OCIO = None
    logger.warning(
        "[NukeOCIO] OpenColorIO not installed. Install with: pip install opencolorio"
    )

# Built-in ACES Studio configs to try, newest first (used when $OCIO is unset)
BUILTIN_CONFIGS = [
    "studio-config-v4.0.0_aces-v2.0_ocio-v2.5",  # ACES 2.0 (OCIO 2.5+)
    "studio-config-v2.2.0_aces-v1.3_ocio-v2.4",  # ACES 1.3 (OCIO 2.4+)
    "studio-config-v2.1.0_aces-v1.3_ocio-v2.3",  # ACES 1.3 (OCIO 2.3+)
    "studio-config-v1.0.0_aces-v1.3_ocio-v2.1",  # ACES 1.3 (OCIO 2.1+)
]

# Used ONLY when OpenColorIO cannot be imported, so the nodes still register
# (and saved workflows still load) with a sensible dropdown. These are the
# colourspaces of studio-config-v4.0.0_aces-v2.0_ocio-v2.5.
FALLBACK_COLORSPACES = [
    "ACES2065-1", "ACEScc", "ACEScct", "ACEScg", "ADX10", "ADX16",
    "ARRI LogC3 (EI800)", "ARRI LogC4", "Apple Log", "BMDFilm WideGamut Gen5",
    "Camera Rec.709", "CanonLog2 CinemaGamut D55", "CanonLog3 CinemaGamut D55",
    "D-Log D-Gamut", "DaVinci Intermediate WideGamut", "Display P3 - Display",
    "Display P3 HDR - Display", "Gamma 1.8 Encoded Rec.709", "Gamma 2.2 Encoded AP1",
    "Gamma 2.2 Encoded AdobeRGB", "Gamma 2.2 Encoded Rec.709",
    "Gamma 2.2 Rec.709 - Display", "Gamma 2.4 Encoded Rec.709",
    "Linear ARRI Wide Gamut 3", "Linear ARRI Wide Gamut 4", "Linear AdobeRGB",
    "Linear BMD WideGamut Gen5", "Linear CinemaGamut D55", "Linear D-Gamut",
    "Linear DaVinci WideGamut", "Linear P3-D65", "Linear REDWideGamutRGB",
    "Linear Rec.2020", "Linear Rec.709 (sRGB)", "Linear S-Gamut3",
    "Linear S-Gamut3.Cine", "Linear V-Gamut", "Linear Venice S-Gamut3",
    "Linear Venice S-Gamut3.Cine", "Log3G10 REDWideGamutRGB", "P3-D65 - Display",
    "Raw", "Rec.1886 Rec.709 - Display", "Rec.2100-HLG - Display",
    "Rec.2100-PQ - Display", "S-Log3 S-Gamut3", "S-Log3 S-Gamut3.Cine",
    "S-Log3 Venice S-Gamut3", "S-Log3 Venice S-Gamut3.Cine", "ST2084-P3-D65 - Display",
    "V-Log V-Gamut", "sRGB - Display", "sRGB Encoded AP1", "sRGB Encoded P3-D65",
    "sRGB Encoded Rec.709 (sRGB)",
]
FALLBACK_DISPLAYS = ["sRGB - Display", "Rec.1886 Rec.709 - Display", "P3-D65 - Display"]
FALLBACK_VIEWS = ["ACES 2.0 - SDR Video", "Un-tone-mapped", "Raw"]
FALLBACK_SCENE_LINEAR = "ACEScg"

_CONFIG = None
_SOURCE = "none"


def resolve_ocio_config() -> Tuple[Optional["OCIO.Config"], str]:
    """Return (config, human-readable source) following the resolution order."""
    if not OCIO_AVAILABLE:
        return None, "OpenColorIO not installed"

    env_path = os.environ.get("OCIO", "").strip()
    if env_path:
        path = os.path.normpath(os.path.expandvars(os.path.expanduser(env_path)))
        if os.path.isfile(path):
            try:
                config = OCIO.Config.CreateFromFile(path)
                logger.info(f"[NukeOCIO] Loaded config from $OCIO: {path}")
                return config, f"$OCIO={path}"
            except Exception as e:
                logger.warning(
                    f"[NukeOCIO] $OCIO points at '{path}' but it failed to load ({e}); "
                    f"falling back to the built-in ACES config"
                )
        else:
            logger.warning(
                f"[NukeOCIO] $OCIO is set to '{env_path}' but no such file exists; "
                f"falling back to the built-in ACES config"
            )

    for name in BUILTIN_CONFIGS:
        try:
            config = OCIO.Config.CreateFromBuiltinConfig(name)
            if config:
                logger.info(f"[NukeOCIO] Loaded built-in config {name}")
                return config, f"built-in {name}"
        except Exception as e:
            logger.warning(f"[NukeOCIO] Could not load built-in {name}: {e}")

    logger.warning("[NukeOCIO] Could not load any OCIO config")
    return None, "no config could be loaded"


def reload() -> Optional["OCIO.Config"]:
    """(Re)resolve the config. Called once at import; tests may call it again."""
    global _CONFIG, _SOURCE
    _CONFIG, _SOURCE = resolve_ocio_config()
    return _CONFIG


def get_config() -> Optional["OCIO.Config"]:
    return _CONFIG


def config_source() -> str:
    return _SOURCE


def load_config_file(path: str) -> Optional["OCIO.Config"]:
    """Load an explicit .ocio file (used by nodes that take a config path)."""
    if not OCIO_AVAILABLE or not path:
        return None
    path = os.path.normpath(os.path.expandvars(os.path.expanduser(path)))
    if not os.path.isfile(path):
        return None
    try:
        return OCIO.Config.CreateFromFile(path)
    except Exception as e:
        logger.warning(f"[NukeOCIO] Could not load config '{path}': {e}")
        return None


def colorspace_names(config: Optional["OCIO.Config"] = None) -> List[str]:
    """All colourspace names of the active config (or the fallback list)."""
    config = config or _CONFIG
    if config is None:
        return list(FALLBACK_COLORSPACES)
    try:
        names = list(config.getColorSpaceNames())
    except Exception:
        names = [cs.getName() for cs in config.getColorSpaces()]
    return names or list(FALLBACK_COLORSPACES)


def role_names(config: Optional["OCIO.Config"] = None) -> List[str]:
    config = config or _CONFIG
    if config is None:
        return []
    try:
        # OCIO 2 exposes roles as (role, colourspace) pairs
        return [pair[0] for pair in config.getRoles()]
    except Exception:
        try:
            return [config.getRoleName(i) for i in range(config.getNumRoles())]
        except Exception:
            return []


def display_names(config: Optional["OCIO.Config"] = None) -> List[str]:
    config = config or _CONFIG
    if config is None:
        return list(FALLBACK_DISPLAYS)
    try:
        names = list(config.getDisplays())
    except Exception:
        names = []
    return names or list(FALLBACK_DISPLAYS)


def view_names(display: Optional[str] = None, config: Optional["OCIO.Config"] = None) -> List[str]:
    """Views for a display; with no display, the union over all displays."""
    config = config or _CONFIG
    if config is None:
        return list(FALLBACK_VIEWS)
    try:
        displays = [display] if display else list(config.getDisplays())
        views: List[str] = []
        for d in displays:
            for v in config.getViews(d):
                if v not in views:
                    views.append(v)
        return views or list(FALLBACK_VIEWS)
    except Exception:
        return list(FALLBACK_VIEWS)


def scene_linear_name(config: Optional["OCIO.Config"] = None) -> str:
    """The working space: the config's scene_linear role, resolved to a name."""
    config = config or _CONFIG
    if config is None:
        return FALLBACK_SCENE_LINEAR
    try:
        cs = config.getColorSpace(OCIO.ROLE_SCENE_LINEAR)
        if cs is not None:
            return cs.getName()
    except Exception:
        pass
    names = colorspace_names(config)
    for candidate in ("ACEScg", "scene_linear", "linear", "Linear Rec.709 (sRGB)"):
        if candidate in names:
            return candidate
    return names[0] if names else FALLBACK_SCENE_LINEAR


def apply_transform(
    image_np: np.ndarray,
    src_colorspace: str,
    dst_colorspace: str,
    config: Optional["OCIO.Config"] = None,
) -> np.ndarray:
    """Apply an OCIO colourspace transform to a (H, W, C) float array.

    Alpha (4th channel) is passed through untouched. On any failure the input
    is returned unchanged and the error is logged.
    """
    config = config or _CONFIG
    if not OCIO_AVAILABLE or config is None or src_colorspace == dst_colorspace:
        return image_np
    try:
        processor = config.getProcessor(src_colorspace, dst_colorspace)
        cpu = processor.getDefaultCPUProcessor()
        img = np.ascontiguousarray(image_np, dtype=np.float32)
        height, width = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1
        if channels >= 4:
            rgb = np.ascontiguousarray(img[:, :, :3])
            alpha = img[:, :, 3:]
        elif channels == 3:
            rgb = img.copy()
            alpha = None
        else:
            rgb = np.ascontiguousarray(np.repeat(img.reshape(height, width, 1), 3, axis=2))
            alpha = None
        rgb_flat = rgb.reshape(-1, 3)
        desc = OCIO.PackedImageDesc(
            rgb_flat, width, height, 3, OCIO.BIT_DEPTH_F32,
            rgb_flat.strides[1], rgb_flat.strides[0], width * rgb_flat.strides[0],
        )
        cpu.apply(desc)
        out = rgb_flat.reshape(height, width, 3)
        if alpha is not None:
            out = np.concatenate([out, alpha], axis=-1)
        return out
    except Exception as e:
        logger.error(
            f"[NukeOCIO] Transform '{src_colorspace}' -> '{dst_colorspace}' failed: {e}"
        )
        return image_np


def apply_transform_tensor(
    image: torch.Tensor,
    src_colorspace: str,
    dst_colorspace: str,
    config: Optional["OCIO.Config"] = None,
) -> torch.Tensor:
    """Apply an OCIO transform to a [B,H,W,C] torch tensor, frame by frame."""
    if not OCIO_AVAILABLE or src_colorspace == dst_colorspace:
        return image
    frames = [
        torch.from_numpy(
            apply_transform(image[i].detach().cpu().numpy(), src_colorspace, dst_colorspace, config)
        )
        for i in range(image.shape[0])
    ]
    return torch.stack(frames, dim=0).to(image.device, image.dtype)


# Resolve once at import (ComfyUI startup)
reload()
