"""
Reformat node that replicates Nuke's Reformat functionality.

Nuke's Reformat node resizes/repositions an image onto a new output format
(canvas), with pixel-aspect-aware scaling. Semantics follow the Foundry
reference (Transform Nodes > Reformat):

- Orientation is applied first: turn (90 degrees counter-clockwise), then
  flip (vertical mirror), then flop (horizontal mirror).
- The output canvas (width, height, pixel aspect) comes from `type`:
  "to format" uses a named format, "to box" uses explicit box fields,
  "scale" multiplies the input dimensions (pixel aspect preserved).
- Scaling happens in DISPLAY space (display width = pixel width x pixel
  aspect). Raw factors:
      sx_raw = (W_out * pa_out) / (W_in * pa_in)
      sy_raw = H_out / H_in
  resize_type picks how those are combined:
      none    -> no resampling; image is placed on the canvas as-is
      width   -> uniform display-space scale s = sx_raw
      height  -> uniform display-space scale s = sy_raw
      fit     -> s = min(sx_raw, sy_raw)  (smallest side fills)
      fill    -> s = max(sx_raw, sy_raw)
      distort -> sx_raw / sy_raw applied independently per axis
  Aspect-preserving modes resample pixel dimensions by s * (pa_in / pa_out)
  horizontally and s vertically.
- The resampled image is placed on the canvas: centered when `center=True`,
  otherwise lower-left aligned (Nuke's origin is bottom-left, so in
  top-row-first array terms the image is aligned to the bottom rows and the
  left columns). Overflow is cropped; underflow is padded black
  (`black_outside=True`) or with replicated edge pixels (False).

Custom formats can be persisted: with format="custom" and a non-empty
`save_format_as`, the format is written to
`<user directory>/nuke_nodes/user_formats.json` and appears in the format
dropdown after the frontend refreshes node definitions.

Note: ComfyUI IMAGE tensors carry no pixel-aspect metadata, so the input
pixel aspect is provided via the optional `input_pixel_aspect` widget
(default 1.0 = square pixels).
"""

import json
import logging
import math
import os

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor

logger = logging.getLogger(__name__)


# Built-in Nuke formats: name -> (width, height, pixel_aspect)
BUILTIN_FORMATS = {
    "PAL": (720, 576, 1.09),
    "NTSC": (720, 486, 0.91),
    "PAL_16:9": (720, 576, 1.46),
    "NTSC_16:9": (720, 486, 1.21),
    "HD_720": (1280, 720, 1.0),
    "HD_1080": (1920, 1080, 1.0),
    "UHD_4K": (3840, 2160, 1.0),
    "2K_DCP": (2048, 1080, 1.0),
    "4K_DCP": (4096, 2160, 1.0),
    "1K_Super_35(full-ap)": (1024, 778, 1.0),
    "2K_Super_35(full-ap)": (2048, 1556, 1.0),
    "4K_Super_35(full-ap)": (4096, 3112, 1.0),
    "square_256": (256, 256, 1.0),
    "square_512": (512, 512, 1.0),
    "square_1K": (1024, 1024, 1.0),
    "square_2K": (2048, 2048, 1.0),
}

# Documented approximation of Nuke's filter set using OpenCV interpolators
FILTER_MAP = {
    "impulse": 0,  # cv2.INTER_NEAREST
    "cubic": 2,  # cv2.INTER_CUBIC
    "lanczos": 4,  # cv2.INTER_LANCZOS4
    "area": 3,  # cv2.INTER_AREA
}


def _user_formats_path():
    """Path of the persisted user formats JSON.

    Uses ComfyUI's user directory when available; falls back to a
    `nuke_nodes` folder under this pack's directory so the node also works
    standalone (tests, other hosts) where `folder_paths` does not exist.
    """
    try:
        import folder_paths

        base = folder_paths.get_user_directory()
    except Exception:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "nuke_nodes", "user_formats.json")


def load_user_formats():
    """Load persisted user formats as {name: (width, height, pixel_aspect)}.

    Invalid entries and names colliding with built-ins are skipped.
    """
    path = _user_formats_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}

    formats = {}
    if isinstance(data, dict):
        for name, value in data.items():
            try:
                width, height, pixel_aspect = value
                width = int(width)
                height = int(height)
                pixel_aspect = float(pixel_aspect)
            except (TypeError, ValueError):
                logger.warning(
                    f"[NukeReformat] Skipping invalid user format entry: {name!r}"
                )
                continue
            if width < 1 or height < 1 or pixel_aspect <= 0:
                logger.warning(
                    f"[NukeReformat] Skipping degenerate user format: {name!r}"
                )
                continue
            name = str(name)
            if name in BUILTIN_FORMATS or name == "custom":
                logger.warning(
                    f"[NukeReformat] Skipping user format shadowing built-in: {name!r}"
                )
                continue
            formats[name] = (width, height, pixel_aspect)
    return formats


def save_user_format(name, width, height, pixel_aspect):
    """Persist a custom format to the user formats JSON (merge + atomic write)."""
    if name in BUILTIN_FORMATS or name == "custom":
        logger.warning(
            f"[NukeReformat] Not saving format '{name}': "
            f"name collides with a built-in format"
        )
        return

    path = _user_formats_path()
    existing = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            existing = data
    except (OSError, ValueError):
        pass

    existing[name] = [int(width), int(height), float(pixel_aspect)]

    tmp_path = f"{path}.__tmp{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
        logger.info(
            f"[NukeReformat] Saved custom format '{name}' "
            f"({int(width)}x{int(height)}, pixel aspect {float(pixel_aspect)}) "
            f"to {path}"
        )
    except OSError as e:
        logger.error(f"[NukeReformat] Failed to save format '{name}': {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _round_half_up(value):
    """Deterministic round-half-up (avoids Python's banker's rounding)."""
    return int(math.floor(value + 0.5))


class NukeReformat(NukeNodeBase):
    """
    Reformat node matching Nuke's Reformat node behavior.

    Resizes and repositions the image onto a new output canvas defined by a
    named format, an explicit box, or a scale factor — with pixel-aspect
    aware resize modes (none/width/height/fit/fill/distort), orientation
    controls (turn/flip/flop), centering, and black or replicated borders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # User formats are read from disk here so a format saved during a
        # session shows up in the dropdown once the frontend refreshes the
        # node definitions.
        format_names = (
            list(BUILTIN_FORMATS.keys())
            + sorted(load_user_formats().keys())
            + ["custom"]
        )
        return {
            "required": {
                "image": ("IMAGE",),
                "type": (
                    ["to format", "to box", "scale"],
                    {"default": "to format"},
                ),
                "format": (format_names, {"default": "HD_1080"}),
                "custom_width": (
                    "INT",
                    {"default": 1920, "min": 1, "max": 16384, "step": 1},
                ),
                "custom_height": (
                    "INT",
                    {"default": 1080, "min": 1, "max": 16384, "step": 1},
                ),
                "custom_pixel_aspect": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01},
                ),
                "save_format_as": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "When format is 'custom' and this is non-empty, "
                            "the custom width/height/pixel aspect are saved "
                            "under this name and appear in the format "
                            "dropdown after a frontend refresh."
                        ),
                    },
                ),
                "box_width": (
                    "INT",
                    {"default": 1920, "min": 1, "max": 16384, "step": 1},
                ),
                "box_height": (
                    "INT",
                    {"default": 1080, "min": 1, "max": 16384, "step": 1},
                ),
                "box_pixel_aspect": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01},
                ),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01},
                ),
                "resize_type": (
                    ["none", "width", "height", "fit", "fill", "distort"],
                    {"default": "width"},
                ),
                "center": ("BOOLEAN", {"default": True}),
                "flip": ("BOOLEAN", {"default": False}),
                "flop": ("BOOLEAN", {"default": False}),
                "turn": ("BOOLEAN", {"default": False}),
                "filter": (
                    ["impulse", "cubic", "lanczos", "area"],
                    {"default": "cubic"},
                ),
                "black_outside": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "input_pixel_aspect": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": (
                            "Pixel aspect ratio of the incoming image "
                            "(1.0 = square pixels). ComfyUI images carry no "
                            "format metadata, so declare it here when the "
                            "source is anamorphic (e.g. PAL 1.09)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reformat"
    CATEGORY = "Nuke/Transform"

    @classmethod
    def VALIDATE_INPUTS(cls, format):
        # The format dropdown is built from a JSON file that can change
        # mid-session (a format saved in this or another session/machine).
        # Declaring `format` here makes ComfyUI skip the combo-membership
        # check, so such values don't fail validation; unknown names are
        # reported at execution time instead.
        return True

    def reformat(
        self,
        image,
        type,
        format,
        custom_width,
        custom_height,
        custom_pixel_aspect,
        save_format_as,
        box_width,
        box_height,
        box_pixel_aspect,
        scale,
        resize_type,
        center,
        flip,
        flop,
        turn,
        filter,
        black_outside,
        input_pixel_aspect=1.0,
    ):
        """Reformat the image batch onto the requested output canvas."""
        if cv2 is None:
            raise RuntimeError(
                "[NukeReformat] OpenCV (opencv-python-headless) is required "
                "for the Reformat node but is not installed"
            )

        # Persist custom format if requested
        save_name = save_format_as.strip() if save_format_as else ""
        if format == "custom" and save_name:
            save_user_format(
                save_name, custom_width, custom_height, custom_pixel_aspect
            )

        img = ensure_batch_dim(image)
        arr = img.detach().cpu().numpy().astype(np.float32)

        # Orientation first (Nuke's transform order for Reformat):
        # turn (90 CCW in display terms), then flip (vertical mirror),
        # then flop (horizontal mirror). Arrays are top-row-first, where
        # np.rot90 over (H, W) axes is a counter-clockwise rotation of the
        # displayed image.
        if turn:
            arr = np.rot90(arr, k=1, axes=(1, 2))
        if flip:
            arr = arr[:, ::-1, :, :]
        if flop:
            arr = arr[:, :, ::-1, :]
        arr = np.ascontiguousarray(arr)

        batch, h_in, w_in, channels = arr.shape
        pa_in = float(input_pixel_aspect)
        if pa_in <= 0:
            pa_in = 1.0

        # Output canvas (W_out, H_out, pa_out) from type
        if type == "to format":
            w_out, h_out, pa_out = self._resolve_format(
                format, custom_width, custom_height, custom_pixel_aspect
            )
        elif type == "to box":
            w_out = int(box_width)
            h_out = int(box_height)
            pa_out = float(box_pixel_aspect)
        elif type == "scale":
            s = max(float(scale), 0.01)
            w_out = _round_half_up(w_in * s)
            h_out = _round_half_up(h_in * s)
            pa_out = pa_in  # pixel aspect preserved
        else:
            raise ValueError(f"[NukeReformat] Unknown type: {type}")

        # Guard degenerate canvases
        w_out = max(1, w_out)
        h_out = max(1, h_out)
        if pa_out <= 0:
            pa_out = 1.0

        # Resampled pixel dimensions from resize_type (display-space math)
        if resize_type == "none":
            # Pure crop/pad placement, no resampling
            new_w, new_h = w_in, h_in
        else:
            sx_raw = (w_out * pa_out) / (w_in * pa_in)
            sy_raw = h_out / h_in
            if resize_type == "width":
                sx = sy = sx_raw
            elif resize_type == "height":
                sx = sy = sy_raw
            elif resize_type == "fit":
                sx = sy = min(sx_raw, sy_raw)
            elif resize_type == "fill":
                sx = sy = max(sx_raw, sy_raw)
            elif resize_type == "distort":
                sx, sy = sx_raw, sy_raw
            else:
                raise ValueError(
                    f"[NukeReformat] Unknown resize_type: {resize_type}"
                )
            # Display-space scale s maps to pixel dims via s * (pa_in/pa_out)
            # horizontally and s vertically
            new_w = max(1, _round_half_up(w_in * sx * (pa_in / pa_out)))
            new_h = max(1, _round_half_up(h_in * sy))

        interpolation = FILTER_MAP.get(filter, FILTER_MAP["cubic"])

        out = np.zeros((batch, h_out, w_out, channels), dtype=np.float32)
        for i in range(batch):
            item = arr[i]
            if (new_w, new_h) != (w_in, h_in):
                item = cv2.resize(
                    item, (new_w, new_h), interpolation=interpolation
                )
                if item.ndim == 2:  # cv2 drops a singleton channel dim
                    item = item[:, :, np.newaxis]
            out[i] = self._place_on_canvas(
                item, w_out, h_out, center, black_outside
            )

        result = torch.from_numpy(out)
        return (normalize_tensor(result),)

    def _resolve_format(
        self, format, custom_width, custom_height, custom_pixel_aspect
    ):
        """Resolve a format name to (width, height, pixel_aspect)."""
        if format == "custom":
            return (
                max(1, int(custom_width)),
                max(1, int(custom_height)),
                float(custom_pixel_aspect),
            )
        if format in BUILTIN_FORMATS:
            return BUILTIN_FORMATS[format]
        user_formats = load_user_formats()
        if format in user_formats:
            return user_formats[format]
        raise ValueError(
            f"[NukeReformat] Unknown format '{format}'. It may have been "
            f"removed from {_user_formats_path()}"
        )

    @staticmethod
    def _place_on_canvas(item, w_out, h_out, center, black_outside):
        """Place a single (H, W, C) image onto a (h_out, w_out) canvas.

        Centered when `center` is True, otherwise lower-left aligned in
        Nuke's bottom-left-origin display coordinates — which in
        top-row-first array terms means aligned to the bottom rows and left
        columns. Overflow is cropped; underflow is padded black
        (black_outside=True) or with replicated edge pixels (False).
        """
        h, w = item.shape[:2]
        diff_x = w_out - w
        diff_y = h_out - h
        if center:
            off_x = diff_x // 2
            # Center in display (bottom-first) space, then convert the
            # bottom offset to a top-row array offset so odd leftovers land
            # where Nuke puts them.
            off_y = diff_y - (diff_y // 2)
        else:
            off_x = 0  # left aligned
            off_y = diff_y  # bottom aligned (arrays are top-row-first)

        src_x0 = max(0, -off_x)
        src_x1 = min(w, w_out - off_x)
        src_y0 = max(0, -off_y)
        src_y1 = min(h, h_out - off_y)

        if src_x1 <= src_x0 or src_y1 <= src_y0:
            # No overlap between image and canvas
            return np.zeros((h_out, w_out, item.shape[2]), dtype=np.float32)

        cropped = item[src_y0:src_y1, src_x0:src_x1]

        pad_top = max(0, off_y)
        pad_left = max(0, off_x)
        pad_bottom = h_out - pad_top - (src_y1 - src_y0)
        pad_right = w_out - pad_left - (src_x1 - src_x0)

        if pad_top or pad_bottom or pad_left or pad_right:
            border = (
                cv2.BORDER_CONSTANT if black_outside else cv2.BORDER_REPLICATE
            )
            cropped = cv2.copyMakeBorder(
                np.ascontiguousarray(cropped),
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                border,
                value=0.0,
            )
            if cropped.ndim == 2:
                cropped = cropped[:, :, np.newaxis]
        return cropped


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeReformat": NukeReformat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeReformat": "Nuke Reformat",
}
