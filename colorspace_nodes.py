"""
Colour space nodes using OpenColorIO (OCIO).

All OCIO state lives in ``ocio_config``: the active config is resolved once at
ComfyUI startup (``$OCIO`` if set, otherwise OCIO's built-in ACES Studio
config) and every dropdown here is built from that config, so the lists
always match what the transforms actually use. Changing ``$OCIO`` requires a
ComfyUI restart. ``NukeOCIOInfo`` reports which config is in use.

Nodes:
- NukeOCIOColorSpace    - colourspace -> colourspace (Nuke OCIOColorSpace)
- NukeOCIODisplay       - display/view transform, forward or inverse (OCIODisplay)
- NukeOCIOFileTransform - LUT files via OCIO FileTransform (OCIOFileTransform)
- NukeOCIOInfo          - report on the active config and the luts/ folder

Requires: pip install opencolorio (2.2+; 2.5+ for ACES 2.0)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from . import ocio_config
from .ocio_config import OCIO, OCIO_AVAILABLE, OCIO_VERSION
from .utils import NukeNodeBase, ensure_batch_dim

logger = logging.getLogger(__name__)

_RESTART_NOTE = (
    "Lists come from the active OCIO config ($OCIO or built-in ACES); "
    "changing $OCIO needs a ComfyUI restart."
)
_ROLE_PREFIX = "role:"

# ---------------------------------------------------------------------------
# Dropdown helpers
# ---------------------------------------------------------------------------


def _colorspace_options() -> List[str]:
    """Colourspace names of the active config followed by ``role:<name>`` entries."""
    options = list(ocio_config.colorspace_names())
    options += [f"{_ROLE_PREFIX}{role}" for role in ocio_config.role_names()]
    return options


def _resolve_colorspace(name: str) -> str:
    """Map a dropdown value to what OCIO expects.

    ``role:x`` becomes ``x`` - OCIO accepts role names wherever a colourspace
    name is expected. Plain colourspace names pass through unchanged.
    """
    if name.startswith(_ROLE_PREFIX):
        return name[len(_ROLE_PREFIX):]
    return name


def _pick_default(options: List[str], *preferred: str) -> str:
    """First preferred entry present in options, else the first option."""
    for candidate in preferred:
        if candidate and candidate in options:
            return candidate
    return options[0] if options else ""


# ---------------------------------------------------------------------------
# CPU processor application (shared by Display and FileTransform)
# ---------------------------------------------------------------------------


def _apply_cpu_processor(image: torch.Tensor, cpu_processor, mix: float = 1.0) -> torch.Tensor:
    """Run an OCIO CPU processor over a [B,H,W,C] tensor, frame by frame.

    Alpha (4th channel) is passed through untouched; single-channel input is
    replicated to RGB. ``mix`` < 1 blends the result back towards the input.
    """
    img = ensure_batch_dim(image)
    frames = []
    for i in range(img.shape[0]):
        frame = np.ascontiguousarray(img[i].detach().cpu().numpy(), dtype=np.float32)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        height, width, channels = frame.shape
        if channels >= 4:
            rgb = np.ascontiguousarray(frame[:, :, :3])
            alpha = frame[:, :, 3:]
        elif channels == 3:
            rgb = frame.copy()
            alpha = None
        else:
            rgb = np.ascontiguousarray(np.repeat(frame[:, :, :1], 3, axis=2))
            alpha = None

        rgb_in = rgb.copy() if mix < 1.0 else None
        rgb_flat = rgb.reshape(-1, 3)
        desc = OCIO.PackedImageDesc(
            rgb_flat, width, height, 3, OCIO.BIT_DEPTH_F32,
            rgb_flat.strides[1], rgb_flat.strides[0], width * rgb_flat.strides[0],
        )
        cpu_processor.apply(desc)
        out = rgb_flat.reshape(height, width, 3)

        if rgb_in is not None:
            out = rgb_in + mix * (out - rgb_in)
        if alpha is not None:
            out = np.concatenate([out, alpha], axis=-1)
        frames.append(out)

    return torch.from_numpy(np.stack(frames, axis=0)).to(image.device, image.dtype)


def _warn_no_ocio(node: str) -> None:
    logger.warning(
        f"[NukeOCIO] {node}: OpenColorIO not installed - passing image through. "
        f"Install with: pip install opencolorio"
    )


# ---------------------------------------------------------------------------
# LUT folder (OCIOFileTransform)
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent
_LUTS_DIR = _MODULE_DIR / "luts"

# Formats OCIO's FileTransform can read
_OCIO_LUT_EXTENSIONS = {
    ".3dl", ".cc", ".ccc", ".cdl", ".clf", ".csp", ".ctf", ".cub", ".cube",
    ".lut", ".spi1d", ".spi3d", ".spimtx", ".vf",
}


def _get_available_ocio_luts() -> List[str]:
    """Scan the luts directory for OCIO-compatible LUT files."""
    if not _LUTS_DIR.exists():
        _LUTS_DIR.mkdir(parents=True, exist_ok=True)
        return ["No LUTs found"]

    lut_files = []
    for file in _LUTS_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in _OCIO_LUT_EXTENSIONS:
            lut_files.append(file.name)

    if not lut_files:
        return ["No LUTs found"]

    return sorted(lut_files)


def _expand_custom_lut_path(custom_lut_path: str) -> str:
    """Expand ~ and environment variables in a user-supplied LUT path."""
    if not custom_lut_path:
        return ""
    return os.path.normpath(os.path.expandvars(os.path.expanduser(custom_lut_path)))


def _resolve_lut_path(lut_file: str, custom_lut_path: str) -> Optional[str]:
    """Shared by IS_CHANGED and apply_file_transform so both agree on the file.

    A custom path wins only when the file exists; otherwise the dropdown LUT is
    used (apply_file_transform logs a warning in that case - this helper stays
    silent because IS_CHANGED calls it on every cache check).
    """
    expanded = _expand_custom_lut_path(custom_lut_path)
    if expanded and os.path.isfile(expanded):
        return expanded
    if lut_file and lut_file != "No LUTs found":
        return str(_LUTS_DIR / lut_file)
    return None


# OCIO interpolation mapping
_OCIO_INTERPOLATION = {
    "default": "INTERP_DEFAULT",
    "nearest": "INTERP_NEAREST",
    "linear": "INTERP_LINEAR",
    "tetrahedral": "INTERP_TETRAHEDRAL",
    "best": "INTERP_BEST",
}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class NukeOCIOColorSpace(NukeNodeBase):
    """
    OCIO colourspace transform, similar to Nuke's OCIOColorSpace.

    Converts between any two colourspaces (or roles, listed as ``role:<name>``)
    of the active OCIO config - ``$OCIO`` if set, otherwise OCIO's built-in
    ACES Studio config. Alpha is passed through untouched.

    Requirements:
        pip install opencolorio
    """

    @classmethod
    def INPUT_TYPES(cls):
        options = _colorspace_options()
        default_in = _pick_default(options, ocio_config.scene_linear_name())
        default_out = _pick_default(options, "sRGB Encoded Rec.709 (sRGB)")

        return {
            "required": {
                "image": ("IMAGE",),
                "in_colorspace": (
                    options,
                    {"default": default_in, "tooltip": f"Source colourspace. {_RESTART_NOTE}"},
                ),
                "out_colorspace": (
                    options,
                    {"default": default_out, "tooltip": f"Destination colourspace. {_RESTART_NOTE}"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_colorspace"
    CATEGORY = "Nuke/Color"

    def transform_colorspace(self, image, in_colorspace, out_colorspace):
        """Transform image from one colourspace to another using OCIO."""
        if not OCIO_AVAILABLE:
            _warn_no_ocio("ColorSpace")
            return (image,)

        config = ocio_config.get_config()
        if config is None:
            logger.warning(
                f"[NukeOCIO] ColorSpace: no OCIO config available "
                f"({ocio_config.config_source()}) - passing image through"
            )
            return (image,)

        src = _resolve_colorspace(in_colorspace)
        dst = _resolve_colorspace(out_colorspace)
        if src == dst:
            return (image,)

        img = ensure_batch_dim(image)
        result = ocio_config.apply_transform_tensor(img, src, dst, config)
        return (result,)


class NukeOCIODisplay(NukeNodeBase):
    """
    OCIO display/view transform, similar to Nuke's OCIODisplay / viewer process.

    Converts a scene-referred image to display-referred using a display and
    view of the active OCIO config.

    The 'invert' parameter allows round-tripping:
    - forward: linear -> display (for viewing)
    - inverse: display -> linear (to reverse a display transform)
    """

    @classmethod
    def INPUT_TYPES(cls):
        displays = ocio_config.display_names()
        views = ocio_config.view_names()
        options = _colorspace_options()

        default_display = _pick_default(displays, "sRGB - Display")
        default_view = _pick_default(views, *ocio_config.view_names(default_display))
        default_input = _pick_default(options, ocio_config.scene_linear_name())

        return {
            "required": {
                "image": ("IMAGE",),
                "display": (
                    displays,
                    {"default": default_display, "tooltip": f"Target display. {_RESTART_NOTE}"},
                ),
                "view": (
                    views,
                    {
                        "default": default_view,
                        "tooltip": "View transform (must belong to the chosen display). "
                        + _RESTART_NOTE,
                    },
                ),
                "input_colorspace": (
                    options,
                    {
                        "default": default_input,
                        "tooltip": f"Colourspace of the input image. {_RESTART_NOTE}",
                    },
                ),
                "invert": (["forward", "inverse"], {"default": "forward"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_display"
    CATEGORY = "Nuke/Color"

    def apply_display(self, image, display, view, input_colorspace, invert="forward"):
        """Apply a display/view transform using OCIO.

        Args:
            invert: "forward" applies the display transform (linear -> display),
                    "inverse" reverses it (display -> linear).
        """
        if not OCIO_AVAILABLE:
            _warn_no_ocio("Display")
            return (image,)

        config = ocio_config.get_config()
        if config is None:
            logger.warning(
                f"[NukeOCIO] Display: no OCIO config available "
                f"({ocio_config.config_source()}) - passing image through"
            )
            return (image,)

        valid_views = ocio_config.view_names(display, config)
        if view not in valid_views:
            logger.warning(
                f"[NukeOCIO] Display: view '{view}' is not defined for display "
                f"'{display}' (valid: {', '.join(valid_views)}) - passing image through"
            )
            return (image,)

        try:
            transform = OCIO.DisplayViewTransform()
            transform.setSrc(_resolve_colorspace(input_colorspace))
            transform.setDisplay(display)
            transform.setView(view)

            direction = (
                OCIO.TRANSFORM_DIR_INVERSE
                if invert == "inverse"
                else OCIO.TRANSFORM_DIR_FORWARD
            )
            processor = config.getProcessor(transform, direction)
            cpu_processor = processor.getDefaultCPUProcessor()
            return (_apply_cpu_processor(image, cpu_processor),)

        except Exception as e:
            logger.error(
                f"[NukeOCIO] Display: transform '{input_colorspace}' -> "
                f"{display} / {view} ({invert}) failed: {e}"
            )
            return (image,)


class NukeOCIOInfo(NukeNodeBase):
    """
    OCIO Info node - reports the active OCIO configuration.

    Shows where the config came from ($OCIO or built-in ACES), the working
    space, every colourspace / role / display / view, and the LUT files found
    in the package's luts/ folder (used by NukeOCIOFileTransform).
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

    @staticmethod
    def _lut_section() -> List[str]:
        lines = ["LUT files in luts/ (NukeOCIOFileTransform):", f"  Folder: {_LUTS_DIR}"]
        luts = _get_available_ocio_luts()
        if luts == ["No LUTs found"]:
            lines.append("  No LUT files found.")
            lines.append("  Supported: " + ", ".join(sorted(_OCIO_LUT_EXTENSIONS)))
        else:
            lines.append(f"  Found {len(luts)} file(s):")
            lines += [f"    - {name}" for name in luts]
        return lines

    def get_info(self):
        """Describe the active OCIO configuration."""
        lines: List[str] = []

        if not OCIO_AVAILABLE:
            lines.append("OpenColorIO not installed.")
            lines.append("Install with: pip install opencolorio")
            lines.append("")
            lines.append(f"Config source: {ocio_config.config_source()}")
            lines.append("")
            lines += self._lut_section()
            return ("\n".join(lines),)

        config = ocio_config.get_config()
        lines.append(f"OCIO version: {OCIO_VERSION}")
        lines.append(f"Config source: {ocio_config.config_source()}")

        if config is None:
            lines.append("")
            lines.append("No OCIO config could be loaded.")
            lines.append("OpenColorIO 2.2+ ships built-in ACES configs; upgrade with:")
            lines.append("  pip install opencolorio --upgrade")
            lines.append("")
            lines += self._lut_section()
            return ("\n".join(lines),)

        try:
            description = (config.getDescription() or "").strip()
            if description:
                lines.append(f"Description: {description.splitlines()[0]}")
            lines.append(f"Working space (scene_linear): {ocio_config.scene_linear_name(config)}")
            lines.append("")

            colorspaces = ocio_config.colorspace_names(config)
            lines.append(f"Colourspaces ({len(colorspaces)}):")
            lines += [f"  - {name}" for name in colorspaces]
            lines.append("")

            roles = ocio_config.role_names(config)
            lines.append(f"Roles ({len(roles)}):")
            for role in roles:
                try:
                    cs = config.getColorSpace(role)
                    target = cs.getName() if cs is not None else "?"
                except Exception:
                    target = "?"
                lines.append(f"  - role:{role} -> {target}")
            lines.append("")

            displays = ocio_config.display_names(config)
            lines.append(f"Displays ({len(displays)}):")
            for display in displays:
                views = ocio_config.view_names(display, config)
                lines.append(f"  - {display} ({len(views)} views)")
                lines += [f"      View: {view}" for view in views]
            lines.append("")

            lines += self._lut_section()
            return ("\n".join(lines),)

        except Exception as e:
            logger.error(f"[NukeOCIO] Info: error reading config: {e}")
            return ("\n".join(lines + [f"Error reading config: {e}"]),)


class NukeOCIOFileTransform(NukeNodeBase):
    """
    OCIO File Transform node - loads and applies LUT files via OpenColorIO.

    Similar to Nuke's OCIOFileTransform node, this uses OCIO's FileTransform
    to apply color transforms from LUT files. Supports forward and inverse
    directions, multiple interpolation modes, and a mix slider.

    Supported formats:
    - .cube (Resolve, Adobe)
    - .3dl (Autodesk Flame, Lustre)
    - .csp (Cinespace)
    - .spi1d / .spi3d / .spimtx (Sony Pictures Imageworks)
    - .clf / .ctf (Common LUT Format, Autodesk CTF)
    - .lut (Houdini)
    - .cub (Truelight)
    - .vf (Nuke)
    - .cdl / .ccc / .cc (ASC CDL)

    Place LUT files in the 'luts' folder within the nuke-nodes package.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lut_files = _get_available_ocio_luts()

        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": (
                    lut_files,
                    {
                        "default": lut_files[0] if lut_files else "No LUTs found",
                    },
                ),
                "direction": (["forward", "inverse"], {"default": "forward"}),
                "interpolation": (
                    ["default", "nearest", "linear", "tetrahedral", "best"],
                    {
                        "default": "default",
                    },
                ),
                "mix": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            },
            "optional": {
                "custom_lut_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional: absolute path to a LUT file",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_file_transform"
    CATEGORY = "Nuke/Color"

    @classmethod
    def IS_CHANGED(
        cls,
        image,
        lut_file,
        direction="forward",
        interpolation="default",
        mix=1.0,
        custom_lut_path="",
    ):
        """Fingerprint the resolved LUT file so caching invalidates on file changes.

        Mirrors the path resolution in apply_file_transform.
        """
        lut_path = _resolve_lut_path(lut_file, custom_lut_path)
        if lut_path is None:
            return "no-lut"

        try:
            stat = os.stat(lut_path)
            return f"{lut_path}:{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            return f"missing:{lut_path}"

    def apply_file_transform(
        self,
        image,
        lut_file,
        direction="forward",
        interpolation="default",
        mix=1.0,
        custom_lut_path="",
    ):
        """Apply an OCIO FileTransform LUT to the input image."""
        if not OCIO_AVAILABLE:
            _warn_no_ocio("FileTransform")
            return (image,)

        lut_path = _resolve_lut_path(lut_file, custom_lut_path)
        expanded_custom = _expand_custom_lut_path(custom_lut_path)
        if expanded_custom and lut_path != expanded_custom:
            # A typo in custom_lut_path must not silently apply another LUT
            logger.warning(
                f"[NukeOCIO] FileTransform: custom_lut_path '{custom_lut_path}' "
                f"not found - falling back to the dropdown LUT '{lut_path}'"
            )
        if lut_path is None:
            logger.warning("[NukeOCIO] FileTransform: no LUT file specified - passing image through")
            return (image,)

        if not os.path.exists(lut_path):
            logger.error(f"[NukeOCIO] FileTransform: LUT file not found: {lut_path}")
            return (image,)

        try:
            file_transform = OCIO.FileTransform()
            file_transform.setSrc(lut_path)

            interp_name = _OCIO_INTERPOLATION.get(interpolation, "INTERP_DEFAULT")
            file_transform.setInterpolation(getattr(OCIO, interp_name, OCIO.INTERP_DEFAULT))

            ocio_direction = (
                OCIO.TRANSFORM_DIR_INVERSE
                if direction == "inverse"
                else OCIO.TRANSFORM_DIR_FORWARD
            )

            # The active config supplies search paths / context for the LUT;
            # a raw config is enough when no config could be resolved.
            config = ocio_config.get_config()
            if config is None:
                config = OCIO.Config.CreateRaw()
            processor = config.getProcessor(file_transform, ocio_direction)
            cpu_processor = processor.getDefaultCPUProcessor()

            return (_apply_cpu_processor(image, cpu_processor, mix=mix),)

        except Exception as e:
            logger.error(f"[NukeOCIO] FileTransform: '{lut_path}' ({direction}) failed: {e}")
            return (image,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeOCIOColorSpace": NukeOCIOColorSpace,
    "NukeOCIODisplay": NukeOCIODisplay,
    "NukeOCIOInfo": NukeOCIOInfo,
    "NukeOCIOFileTransform": NukeOCIOFileTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeOCIOColorSpace": "Nuke OCIO ColorSpace",
    "NukeOCIODisplay": "Nuke OCIO Display",
    "NukeOCIOInfo": "Nuke OCIO Info",
    "NukeOCIOFileTransform": "Nuke OCIO FileTransform",
}
