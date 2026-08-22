"""
Read and Write nodes for loading and saving images, similar to Nuke's Read/Write nodes.

Pixel I/O goes through OpenImageIO only (see ``image_io.py``); frame-sequence
path handling lives in ``sequence.py`` and thumbnail previews in
``preview.py``. Colour management is driven by the shared OpenColorIO config
(``ocio_config.py``): the ``colorspace`` dropdowns list the colourspaces of
the active config ($OCIO or the built-in ACES Studio config) and the working
space is the config's ``scene_linear`` role.

Supported formats (OIIO): EXR, TIFF, PNG, JPEG, DPX, Cineon, HDR, TGA, BMP,
PSD (read-only), camera RAW via LibRaw, and many more.

Sequence patterns supported:
- %04d style (printf format): image.%04d.exr
- #### style (hash padding): image.####.exr
- Frame ranges: 1-100, 1-100x2 (every 2nd frame)
"""

import logging
import os
import re
from typing import List

import folder_paths
import numpy as np
import torch
import torch.nn.functional as F

from . import ocio_config
from .image_io import (
    OIIO_AVAILABLE,
    get_supported_formats,
    oiio,
    oiio_version,
    read_image,
    write_image,
)
from .preview import create_preview_images
from .sequence import (
    _file_exists_case_aware,
    _versioned_sequence_pattern,
    _warn_stale_frames,
    auto_detect_sequence,
    detect_sequence,
    expand_frame_pattern,
    file_change_token,
    parse_frame_pattern,
)
from .utils import NukeNodeBase, cv2, ensure_batch_dim

logger = logging.getLogger(__name__)


# ============================================================================
# Colour helpers (OpenColorIO)
# ============================================================================

_OCIO_RESTART_NOTE = (
    "The list comes from the active OCIO config ($OCIO, or the built-in "
    "ACES Studio config) and is built when ComfyUI starts - restart "
    "ComfyUI after changing $OCIO."
)


def _colorspace_choices() -> List[str]:
    """Combo options for the Read/Write ``colorspace`` input."""
    if not ocio_config.OCIO_AVAILABLE:
        return ["raw"]
    return ["raw"] + ocio_config.colorspace_names()


def _resolve_colorspace(colorspace: str, node_tag: str) -> str:
    """Return the colourspace to use, downgrading to "raw" when OCIO is missing."""
    if colorspace == "raw":
        return "raw"
    if not ocio_config.OCIO_AVAILABLE:
        logger.warning(
            f"[{node_tag}] colorspace '{colorspace}' requested but OpenColorIO is "
            f"not installed; treating as 'raw' (no conversion)"
        )
        return "raw"
    return colorspace


def _to_working_space(img: np.ndarray, colorspace: str) -> np.ndarray:
    """File colourspace -> working space (scene_linear role). Alpha untouched."""
    if colorspace == "raw":
        return img
    return ocio_config.apply_transform(img, colorspace, ocio_config.scene_linear_name())


def _from_working_space(img: np.ndarray, colorspace: str) -> np.ndarray:
    """Working space (scene_linear role) -> file colourspace. Alpha untouched."""
    if colorspace == "raw":
        return img
    return ocio_config.apply_transform(img, ocio_config.scene_linear_name(), colorspace)


def _file_colorspace_tag(colorspace: str, file_type: str) -> str:
    """Value for the ``oiio:ColorSpace`` attribute, or "" to leave it unset.

    Files written in a named colourspace are tagged with it. "raw" EXRs are
    tagged with the working-space name (linear by convention); other "raw"
    formats are left untagged so viewers do not misinterpret them.
    """
    if colorspace != "raw":
        return colorspace
    if file_type == "exr":
        return ocio_config.scene_linear_name()
    return ""


# ============================================================================
# Output path helpers
# ============================================================================

_FRAME_TOKEN_RX = re.compile(r"%\d*d|#+")


def _apply_file_type(file_path: str, file_type: str) -> str:
    """Force ``file_path`` to end in ``.{file_type}`` without eating a frame token.

    ``os.path.splitext`` would treat the token of ``out.####`` / ``out.%04d``
    as the extension, so the token is located in the basename first and
    only an extension that starts AFTER the token counts as a real one:

    - ``out.####``      -> ``out.####.exr``   (token kept, extension appended)
    - ``out.%04d``      -> ``out.%04d.exr``
    - ``out_####``      -> ``out_####.exr``
    - ``out.####.png``  -> ``out.####.exr``   (real extension replaced)
    - ``out.####.exr``  -> unchanged
    - ``out``           -> ``out.exr``;  ``x.tif`` (tiff) -> ``x.tiff``

    The comparison is on the bare lower-cased extension, so ``.EXR`` already
    matches ``exr``. Separators are left exactly as given.
    """
    directory, basename = os.path.split(file_path)
    token = _FRAME_TOKEN_RX.search(basename)
    stem, ext = os.path.splitext(basename)
    if token is not None and len(stem) < token.end():
        # The only "." is before/inside the token: no real extension.
        new_basename = f"{basename}.{file_type}"
    elif ext.lower().lstrip(".") != file_type:
        new_basename = f"{stem}.{file_type}"
    else:
        return file_path
    return os.path.join(directory, new_basename) if directory else new_basename


# ============================================================================
# ComfyUI Nodes
# ============================================================================


class NukeRead(NukeNodeBase):
    """
    Read node - loads images or image sequences from disk.

    Similar to Nuke's Read node, supports:
    - Single images or image sequences
    - Frame pattern matching (%04d, ####)
    - Frame range specification (first/last frame)
    - Wide format support via OpenImageIO
    - Optional thumbnail preview

    Supported formats: EXR, TIFF, PNG, JPEG, DPX, HDR, TGA, BMP, and more.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Path to image or sequence (e.g., /path/image.%04d.exr)",
                    },
                ),
                "frame": (
                    "INT",
                    {
                        "default": 1,
                        "min": -999999,
                        "max": 999999,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "load_as_sequence": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Treat path as image sequence (enable for patterns like ####, %04d)",
                    },
                ),
                "first_frame": (
                    "INT",
                    {
                        "default": 1,
                        "min": -999999,
                        "max": 999999,
                        "step": 1,
                    },
                ),
                "last_frame": (
                    "INT",
                    {
                        "default": 1,
                        "min": -999999,
                        "max": 999999,
                        "step": 1,
                    },
                ),
                "frame_mode": (["single", "range", "all"], {"default": "single"}),
                "missing_frames": (
                    ["error", "black", "hold", "nearest"],
                    {"default": "black"},
                ),
                "colorspace": (
                    _colorspace_choices(),
                    {
                        "default": "raw",
                        "tooltip": (
                            "Colorspace the FILE IS IN on disk (matches Nuke's "
                            "Read node convention). The node converts from this "
                            "into the OCIO working space (the config's "
                            "scene_linear role, e.g. ACEScg).\n\n"
                            "  raw - no conversion (pixels are passed through "
                            "as stored)\n"
                            "  any other entry - OCIO colourspace the file is "
                            "encoded in (e.g. 'sRGB Encoded Rec.709 (sRGB)' for "
                            "PNG / JPG, 'ACEScg' for linear EXR)\n\n"
                            + _OCIO_RESTART_NOTE
                        ),
                    },
                ),
                "show_preview": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Show thumbnail preview in node"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "read_image"
    CATEGORY = "Nuke/IO"

    @classmethod
    def IS_CHANGED(
        cls,
        file_path="",
        frame=1,
        load_as_sequence=True,
        first_frame=1,
        last_frame=1,
        frame_mode="single",
        missing_frames="black",
        **kwargs,
    ):
        """
        Fingerprint the file(s) that read_image WOULD load so ComfyUI only
        re-runs the node when something on disk actually changes.

        Mirrors read_image's path/frame resolution, then stats each resolved
        file into a (path, mtime_ns, size) token. Missing files contribute a
        deterministic "missing:<path>" token, so the node re-runs when they
        appear. Only stat()/glob are used — file contents are never read.
        """
        if not file_path:
            # Deterministic: output is a constant black image.
            return ""

        # Mirror read_image's path resolution
        file_path = os.path.expandvars(os.path.expanduser(file_path))
        pattern, frame_spec, padding = parse_frame_pattern(file_path)

        # Mirror read_image's sequence auto-detection (stat-only)
        auto_frames = None
        if load_as_sequence and frame_spec is None:
            detected = auto_detect_sequence(file_path)
            if detected is not None:
                pattern, auto_frames, padding = detected
                frame_spec = "auto"

        is_sequence = load_as_sequence and (frame_spec is not None and padding > 0)

        tokens = []
        available_frames = []

        # Mirror read_image's frame resolution
        if frame_mode == "single":
            frames_to_load = [frame]
        elif frame_mode == "range":
            frames_to_load = list(range(first_frame, last_frame + 1))
        elif frame_mode == "all" and is_sequence:
            if auto_frames is not None:
                available_frames = auto_frames
            else:
                _, available_frames, _ = detect_sequence(file_path)
            frames_to_load = available_frames if available_frames else [frame]
            # Added/removed frames must invalidate the cache
            tokens.append(
                "frames:" + ",".join(str(f) for f in available_frames)
            )
        else:
            frames_to_load = [frame]

        # Nearest mode can substitute another frame's file for a missing one
        if (
            is_sequence
            and missing_frames == "nearest"
            and not available_frames
        ):
            if auto_frames is not None:
                available_frames = auto_frames
            else:
                _, available_frames, _ = detect_sequence(file_path)

        for f in frames_to_load:
            if is_sequence:
                actual_path = expand_frame_pattern(pattern, f, padding)
            else:
                actual_path = file_path

            token = file_change_token(actual_path)
            tokens.append(token)

            # If this frame is missing and nearest mode would substitute
            # another frame's file, fingerprint that file too.
            if (
                token.startswith("missing:")
                and is_sequence
                and missing_frames == "nearest"
                and available_frames
            ):
                nearest = min(available_frames, key=lambda x: abs(x - f))
                nearest_path = expand_frame_pattern(pattern, nearest, padding)
                tokens.append(file_change_token(nearest_path))

        return ";".join(tokens)

    def read_image(
        self,
        file_path,
        frame,
        load_as_sequence=True,
        first_frame=1,
        last_frame=1,
        frame_mode="single",
        missing_frames="black",
        colorspace="raw",
        show_preview=True,
    ):
        """Read image(s) from disk."""

        if not file_path:
            logger.warning("[NukeRead] No file path specified")
            # Return black image
            result = torch.zeros((1, 512, 512, 3))
            return {"ui": {"images": []}, "result": (result,)}

        # Expand environment variables and user home
        file_path = os.path.expandvars(os.path.expanduser(file_path))

        logger.info(f"[NukeRead] Loading path: {file_path}")
        logger.info(f"[NukeRead] Load as sequence: {load_as_sequence}")

        # Resolve colour handling once (warns if OCIO is missing)
        colorspace = _resolve_colorspace(colorspace, "NukeRead")
        if colorspace != "raw":
            logger.info(
                f"[NukeRead] Colorspace: '{colorspace}' -> "
                f"'{ocio_config.scene_linear_name()}' (working space)"
            )

        # Parse frame pattern
        pattern, frame_spec, padding = parse_frame_pattern(file_path)

        # No explicit frame token: try auto-detecting an on-disk sequence
        auto_frames = None
        if load_as_sequence and frame_spec is None:
            detected = auto_detect_sequence(file_path)
            if detected is not None:
                pattern, auto_frames, padding = detected
                frame_spec = "auto"
                logger.info(
                    f"[NukeRead] Auto-detected sequence: {pattern} "
                    f"({len(auto_frames)} frames)"
                )

        is_sequence = load_as_sequence and (frame_spec is not None and padding > 0)

        if frame_spec and padding > 0:
            logger.info(f"[NukeRead] Detected pattern: {pattern}, padding: {padding}")
        else:
            logger.info(f"[NukeRead] No sequence pattern detected, treating as single file")

        # Determine frames to load
        if frame_mode == "single":
            frames_to_load = [frame]
        elif frame_mode == "range":
            frames_to_load = list(range(first_frame, last_frame + 1))
        elif frame_mode == "all" and is_sequence:
            if auto_frames is not None:
                available_frames = auto_frames
            else:
                _, available_frames, _ = detect_sequence(file_path)
            frames_to_load = available_frames if available_frames else [frame]
        else:
            frames_to_load = [frame]

        # Get available frames for nearest/hold modes
        if is_sequence and missing_frames in ["hold", "nearest"]:
            if auto_frames is not None:
                available_frames = auto_frames
            else:
                _, available_frames, _ = detect_sequence(file_path)
        else:
            available_frames = []

        # Load images. A black placeholder is recorded as None and filled in
        # AFTER the loop with the shape of the first frame that actually
        # loaded, so a missing leading frame no longer forces 512x512.
        images = []
        reference_shape = None

        for f in frames_to_load:
            if is_sequence:
                actual_path = expand_frame_pattern(pattern, f, padding)
            else:
                actual_path = file_path

            img = None
            # True when `img` did not come from disk (a held copy of an
            # already-converted frame) and must not be converted again.
            in_working_space = False

            if os.path.exists(actual_path):
                img = read_image(actual_path)
            else:
                # Handle missing frames
                if missing_frames == "error":
                    # Nuke semantics: a missing frame is an error.
                    raise RuntimeError(f"[NukeRead] Frame not found: {actual_path}")
                elif missing_frames == "hold" and images:
                    # Use previous frame (already in working space). A held
                    # placeholder stays a placeholder (black).
                    if images[-1] is not None:
                        img = images[-1].copy()
                        in_working_space = True
                elif missing_frames == "nearest" and available_frames:
                    # Find nearest available frame
                    nearest = min(available_frames, key=lambda x: abs(x - f))
                    nearest_path = expand_frame_pattern(pattern, nearest, padding)
                    img = read_image(nearest_path)
                # "black" - placeholder, filled below

            if img is None:
                images.append(None)
                continue

            # Ensure consistent channel count (minimum 3 channels for ComfyUI)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 1:
                img = np.concatenate([img, img, img], axis=-1)

            # Convention (matches Nuke's Read node): `colorspace` describes
            # what colourspace the FILE IS IN on disk; OCIO converts from
            # that into the config's scene_linear working space.
            if not in_working_space:
                img = _to_working_space(img, colorspace)

            if reference_shape is None:
                reference_shape = img.shape
            images.append(img)

        # Fill black placeholders with the resolution / channel count of the
        # first frame that loaded; 512x512x3 only when nothing loaded at all.
        fill_shape = reference_shape or (512, 512, 3)
        images = [
            np.zeros(fill_shape, dtype=np.float32) if im is None else im
            for im in images
        ]

        # Stack into batch
        if images:
            result = np.stack(images, axis=0)
        else:
            result = np.zeros((1, 512, 512, 3), dtype=np.float32)

        result = torch.from_numpy(result)

        # Create preview if enabled
        ui_images = []
        if show_preview and result.shape[0] > 0:
            ui_images = create_preview_images(result)

        # Return UI data
        return {"ui": {"images": ui_images}, "result": (result,)}


class NukeWrite(NukeNodeBase):
    """
    Write node - saves images or image sequences to disk.

    Similar to Nuke's Write node, supports:
    - Single images or image sequences
    - Frame pattern matching (%04d, ####)
    - Multiple file formats with format-specific options
    - Bit depth control (8, 16, 16f, 32f)
    - EXR compression options
    - Optional thumbnail preview

    Supported formats: EXR, TIFF, PNG, JPEG, DPX, HDR, TGA, BMP, and more.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Common EXR compression types
        exr_compressions = [
            "none",
            "rle",
            "zip",
            "zips",
            "piz",
            "pxr24",
            "b44",
            "b44a",
            "dwaa",
            "dwab",
        ]

        # Common bit depths
        bit_depths = ["8", "16", "16f", "32f"]

        return {
            "required": {
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Output path (e.g., output.%04d.exr or output.#### or output)",
                    },
                ),
                "frame_start": (
                    "INT",
                    {
                        "default": 1,
                        "min": -999999,
                        "max": 999999,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image input. Used when channels is 'rgb' or 'rgba'. "
                                   "Not needed when channels='all_channels'."
                    },
                ),
                "channels": (
                    ["rgba", "rgb", "all_channels"],
                    {
                        "default": "rgba",
                        "tooltip": (
                            "Channel layout for the output file:\n"
                            "  rgba - write 4 channels (RGB + alpha). Uses the IMAGE input.\n"
                            "  rgb - write 3 channels (RGB only). Uses the IMAGE input.\n"
                            "  all_channels - write all passes as multi-channel EXR. "
                            "Uses the PASSES input. Colorspace conversion is applied "
                            "only to light passes, not to data passes (normal, depth, "
                            "position, motion, IDs, mattes, UVs)."
                        ),
                    },
                ),
                "passes": (
                    "NUKE_PASSES",
                    {
                        "tooltip": "Multi-pass bundle from Nuke Read MultiPass. "
                                   "Used when channels='all_channels'."
                    },
                ),
                "file_type": (
                    ["exr", "tiff", "png", "jpg", "dpx", "hdr", "tga", "bmp", "webp"],
                    {"default": "exr"},
                ),
                "bit_depth": (bit_depths, {"default": "16f"}),
                "compression": (exr_compressions, {"default": "dwaa"}),
                "frame_padding": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of digits for frame numbers (e.g., 4 = 0001, 0002...)",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "True: overwrite existing files at the exact "
                            "target paths (each frame is written atomically "
                            "via a temp file + rename).\n"
                            "False: never clobber - if any target frame "
                            "already exists, the WHOLE sequence is written "
                            "with a versioned base name (name_1.0001.exr, "
                            "name_1.0002.exr, ...) so one render always "
                            "shares one consistent name."
                        ),
                    },
                ),
                "create_directories": ("BOOLEAN", {"default": True}),
                "colorspace": (
                    _colorspace_choices(),
                    {
                        "default": "raw",
                        "tooltip": (
                            "Colorspace to WRITE the file as on disk (matches "
                            "Nuke's Write node convention). Input is assumed to "
                            "be in the OCIO working space (the config's "
                            "scene_linear role, e.g. ACEScg); the node converts "
                            "from there into the chosen space.\n\n"
                            "  raw - write input as-is, no conversion (EXRs are "
                            "tagged with the working-space name)\n"
                            "  any other entry - OCIO colourspace to encode to "
                            "(e.g. 'sRGB Encoded Rec.709 (sRGB)' for PNG / JPG, "
                            "'ACEScg' for linear EXR)\n\n"
                            + _OCIO_RESTART_NOTE
                        ),
                    },
                ),
                "show_preview": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Show thumbnail preview in node"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file_paths")
    FUNCTION = "write_image"
    CATEGORY = "Nuke/IO"
    OUTPUT_NODE = True

    def write_image(
        self,
        file_path,
        frame_start=1,
        image=None,
        channels="rgba",
        passes=None,
        file_type="exr",
        bit_depth="16f",
        compression="dwaa",
        frame_padding=4,
        overwrite=False,
        create_directories=True,
        colorspace="raw",
        show_preview=True,
    ):
        """Write image(s) to disk. Supports rgb / rgba / multi-pass EXR."""

        if not file_path:
            logger.warning("[NukeWrite] No file path specified")
            return {"ui": {"images": []}, "result": (image, "")}

        # Resolve colour handling once (warns if OCIO is missing)
        colorspace = _resolve_colorspace(colorspace, "NukeWrite")

        # Branch to multi-pass writer if requested
        if channels == "all_channels":
            return self._write_multipass(
                passes=passes,
                image=image,
                file_path=file_path,
                frame_start=frame_start,
                file_type=file_type,
                bit_depth=bit_depth,
                compression=compression,
                frame_padding=frame_padding,
                overwrite=overwrite,
                create_directories=create_directories,
                colorspace=colorspace,
                show_preview=show_preview,
            )

        # rgb / rgba mode requires an image input
        if image is None:
            logger.error(f"[NukeWrite] channels='{channels}' requires the 'image' input "
                         f"to be connected")
            return {"ui": {"images": []}, "result": (None, "")}

        # Ensure batch dimension
        img = ensure_batch_dim(image)

        # Clamp to rgb / rgba per user selection
        if channels == "rgb" and img.shape[-1] > 3:
            img = img[..., :3]
        elif channels == "rgba":
            if img.shape[-1] == 3:
                # Pad with opaque alpha
                alpha = torch.ones_like(img[..., :1])
                img = torch.cat([img, alpha], dim=-1)
            elif img.shape[-1] > 4:
                img = img[..., :4]

        batch_size = img.shape[0]

        # Get ComfyUI output directory
        output_base = folder_paths.get_output_directory()

        # Process file path - if it's absolute, use as-is; if relative, make it relative to output_base
        file_path = os.path.expandvars(os.path.expanduser(file_path))

        # Check if path is absolute (Windows: C:\path or \\path, Unix: /path)
        is_absolute = os.path.isabs(file_path)

        if not is_absolute:
            # Relative path - join with ComfyUI output directory
            file_path = os.path.join(output_base, file_path)

        # Ensure the correct extension WITHOUT eating a frame token
        # (out.#### -> out.####.exr), then parse the frame pattern once.
        file_path = _apply_file_type(file_path, file_type)
        pattern, frame_spec, padding = parse_frame_pattern(file_path)
        is_sequence = frame_spec is not None and padding > 0
        if not is_sequence:
            # No token in path: Nuke-style numbering with frame_padding digits
            pattern = file_path
            padding = frame_padding

        # Create output directory if needed
        if create_directories:
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        written_paths = []

        base_no_ext, ext_final = os.path.splitext(file_path)

        def build_target_paths(version: int) -> List[str]:
            """Target paths for the whole batch, with optional _<version>
            base-name suffix. Frame numbers are always appended, Nuke-style
            ({base}.{frame}{ext}) when the path has no explicit token.
            Paths are normalised to the native separator."""
            paths = []
            if is_sequence:
                # Explicit frame pattern in path (e.g., %04d or ####)
                pat = (
                    pattern
                    if version == 0
                    else _versioned_sequence_pattern(pattern, version)
                )
                for i in range(batch_size):
                    paths.append(expand_frame_pattern(pat, frame_start + i, padding))
            else:
                vbase = base_no_ext if version == 0 else f"{base_no_ext}_{version}"
                for i in range(batch_size):
                    frame_str = str(frame_start + i).zfill(padding)
                    paths.append(f"{vbase}.{frame_str}{ext_final}")
            return [os.path.normpath(p) for p in paths]

        # Compute the full target set up front
        target_paths = build_target_paths(0)

        if not overwrite and any(_file_exists_case_aware(p) for p in target_paths):
            # No-clobber mode: version the WHOLE sequence with one _N base
            # suffix until the entire set of target paths is collision-free.
            version = 1
            candidate = build_target_paths(version)
            while any(_file_exists_case_aware(p) for p in candidate):
                version += 1
                if version >= 100000:
                    # Ultimate fallback: use timestamp
                    import time

                    version = int(time.time() * 1000)
                    candidate = build_target_paths(version)
                    break
                candidate = build_target_paths(version)
            target_paths = candidate
            logger.info(
                f"[NukeWrite] Existing frame(s) detected; writing whole "
                f"sequence with base suffix _{version}"
            )

        # Prepare metadata. "raw" EXRs are tagged with the working-space
        # name (linear by convention); other "raw" formats stay untagged so
        # viewers do not misinterpret them (e.g. a PNG tagged linear).
        metadata = {
            "Software": "ComfyUI Nuke Nodes",
        }
        colorspace_tag = _file_colorspace_tag(colorspace, file_type)
        if colorspace_tag:
            metadata["oiio:ColorSpace"] = colorspace_tag

        if colorspace != "raw":
            logger.info(
                f"[NukeWrite] Colorspace: '{ocio_config.scene_linear_name()}' "
                f"(working space) -> '{colorspace}'"
            )

        for i in range(batch_size):
            output_path = target_paths[i]

            # Get pixel data
            pixels = img[i].cpu().numpy()

            # Convention (matches Nuke's Write node): `colorspace` describes
            # what colourspace to write the FILE AS on disk. Input is in the
            # OCIO working space; OCIO converts to the chosen output space.
            pixels = _from_working_space(pixels, colorspace)

            # Write the image
            if overwrite:
                # Atomic overwrite: write to a temp file in the same
                # directory (real extension kept last so format inference
                # still works), then os.replace onto the target.
                tmp_base, tmp_ext = os.path.splitext(output_path)
                tmp_path = f"{tmp_base}.__tmp{os.getpid()}{tmp_ext}"
                success = write_image(
                    tmp_path, pixels, bit_depth, compression, metadata
                )
                if success:
                    try:
                        os.replace(tmp_path, output_path)
                    except OSError as e:
                        logger.error(
                            f"[NukeWrite] Atomic replace failed for "
                            f"{output_path}: {e}"
                        )
                        success = False
                if not success and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            else:
                success = write_image(
                    output_path, pixels, bit_depth, compression, metadata
                )

            if success:
                written_paths.append(output_path)
                logger.info(f"[NukeWrite] Written: {output_path}")
            else:
                logger.error(f"[NukeWrite] Failed to write: {output_path}")

        # Overwrite mode never deletes other files — but warn about stale
        # frames beyond the written range (left over from a longer render).
        if overwrite:
            seq_pattern = (
                pattern
                if is_sequence
                else f"{base_no_ext}.%0{padding}d{ext_final}"
            )
            _warn_stale_frames(
                seq_pattern, frame_start, frame_start + batch_size - 1
            )

        # Return paths as string
        paths_str = "\n".join(written_paths)

        # Create preview if enabled
        ui_images = []
        if show_preview and image.shape[0] > 0:
            ui_images = create_preview_images(image)

        return {"ui": {"images": ui_images}, "result": (image, paths_str)}

    def _write_multipass(
        self,
        passes,
        image,
        file_path,
        frame_start,
        file_type,
        bit_depth,
        compression,
        frame_padding,
        overwrite,
        create_directories,
        colorspace,
        show_preview,
    ):
        """Write a multi-pass NUKE_PASSES bundle as a single multi-channel EXR.

        Colorspace conversion is applied only to light/beauty passes, not to
        data passes (normals, depth, position, motion, IDs, mattes, UVs) —
        those are scene-referred data and must not be colour-transformed.
        """
        if not OIIO_AVAILABLE:
            raise RuntimeError(
                "OpenImageIO is required for multi-pass EXR write. "
                "Install with: pip install OpenImageIO"
            )
        if not passes:
            logger.error("[NukeWrite] channels='all_channels' but no passes bundle "
                         "was connected")
            return {"ui": {"images": []}, "result": (image, "")}

        # Force .exr (multi-channel beyond RGBA is only meaningful for EXR)
        if file_type != "exr":
            logger.warning(f"[NukeWrite] Forcing .exr for multi-pass "
                           f"(ignoring file_type={file_type})")
            file_type = "exr"

        # Import multi-pass helpers lazily to avoid circular imports at module load
        from .multipass_nodes import channel_suffix_for_pass, is_data_pass

        # Resolve output directory base like the single-image path does
        output_base = folder_paths.get_output_directory()
        file_path = os.path.expandvars(os.path.expanduser(file_path))
        if not os.path.isabs(file_path):
            file_path = os.path.join(output_base, file_path)

        # Force the extension without eating a frame token, then parse once
        file_path = _apply_file_type(file_path, file_type)
        pattern, frame_spec, padding = parse_frame_pattern(file_path)
        is_sequence = frame_spec is not None and padding > 0
        if not is_sequence:
            pattern = file_path
            padding = frame_padding

        # Multi-pass is a single frame at a time (bundle has no batch dim)
        base_no_ext, ext_final = os.path.splitext(file_path)

        def build_target_path(version: int) -> str:
            """Target path with optional _<version> base-name suffix.
            Frame number is appended Nuke-style ({base}.{frame}{ext}) when
            the path has no explicit token. Native separators."""
            if is_sequence:
                pat = (
                    pattern
                    if version == 0
                    else _versioned_sequence_pattern(pattern, version)
                )
                path = expand_frame_pattern(pat, frame_start, padding)
            else:
                vbase = base_no_ext if version == 0 else f"{base_no_ext}_{version}"
                frame_str = str(frame_start).zfill(padding)
                path = f"{vbase}.{frame_str}{ext_final}"
            return os.path.normpath(path)

        output_path = build_target_path(0)

        if not overwrite and _file_exists_case_aware(output_path):
            # No-clobber mode: version the base name until the target is free
            version = 1
            candidate = build_target_path(version)
            while _file_exists_case_aware(candidate):
                version += 1
                if version >= 100000:
                    # Ultimate fallback: use timestamp
                    import time

                    version = int(time.time() * 1000)
                    candidate = build_target_path(version)
                    break
                candidate = build_target_path(version)
            output_path = candidate
            logger.info(
                f"[NukeWrite] Target exists; writing multi-pass EXR with "
                f"base suffix _{version}"
            )

        if create_directories:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # --- Build the combined channel array ---
        first = next(iter(passes.values()))
        H, W = first.shape[:2]

        channel_names = []
        channel_arrays = []

        for pass_name, tensor in passes.items():
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            pH, pW, C = arr.shape

            # Resolution mismatch safeguard (shouldn't normally happen)
            if (pH, pW) != (H, W):
                arr = _resize_pass(arr, H, W)

            # Apply colorspace conversion ONLY to color passes, not data passes.
            # Alpha channels within a color pass are also passed through unchanged.
            if colorspace != "raw" and not is_data_pass(pass_name):
                arr = _convert_pass(arr, colorspace)

            # Naming: top-level RGBA, otherwise passname.<suffix>
            suffixes = channel_suffix_for_pass(pass_name, C)
            if pass_name.upper() == "RGBA":
                names = suffixes[:C]
            else:
                names = [f"{pass_name}.{s}" for s in suffixes[:C]]

            channel_names.extend(names)
            channel_arrays.append(arr)

        combined = np.ascontiguousarray(np.concatenate(channel_arrays, axis=-1))
        total_ch = combined.shape[-1]

        # Bit depth
        if bit_depth == "32f":
            out_arr = combined.astype(np.float32)
            fmt = oiio.FLOAT
        elif bit_depth == "16f":
            out_arr = combined.astype(np.float16)
            fmt = oiio.HALF
        else:
            # 8 / 16 integer make little sense for multi-pass EXR — use 16f
            logger.warning(f"[NukeWrite] bit_depth={bit_depth} not suited for multi-pass, "
                           f"using 16f")
            out_arr = combined.astype(np.float16)
            fmt = oiio.HALF

        spec = oiio.ImageSpec(W, H, total_ch, fmt)
        spec.channelnames = tuple(channel_names)
        spec.attribute("compression", compression)
        spec.attribute("Software", "ComfyUI Nuke Nodes")
        spec.attribute("oiio:ColorSpace", _file_colorspace_tag(colorspace, "exr"))

        if overwrite:
            # Atomic overwrite: write to a temp file in the same directory
            # (real extension kept last for format inference), then replace.
            tmp_base, tmp_ext = os.path.splitext(output_path)
            write_path = f"{tmp_base}.__tmp{os.getpid()}{tmp_ext}"
        else:
            write_path = output_path

        try:
            out = oiio.ImageOutput.create(write_path)
            if out is None:
                raise RuntimeError(f"OIIO cannot create: {write_path} "
                                   f"({oiio.geterror()})")
            if not out.open(write_path, spec):
                raise RuntimeError(f"OIIO open failed: {out.geterror()}")
            if not out.write_image(np.ascontiguousarray(out_arr)):
                err = out.geterror()
                out.close()
                raise RuntimeError(f"OIIO write failed: {err}")
            out.close()

            if overwrite:
                os.replace(write_path, output_path)
        except Exception:
            if overwrite and os.path.exists(write_path):
                try:
                    os.remove(write_path)
                except OSError:
                    pass
            raise

        # Overwrite mode never deletes other files — but warn about stale
        # frames beyond the written range (left over from a longer render).
        if overwrite:
            seq_pattern = (
                pattern
                if is_sequence
                else f"{base_no_ext}.%0{padding}d{ext_final}"
            )
            _warn_stale_frames(seq_pattern, frame_start, frame_start)

        logger.info(f"[NukeWrite] Multi-pass EXR written: {output_path}")
        logger.info(f"[NukeWrite]   {W}x{H}, {total_ch} channels, {bit_depth}, "
                    f"compression={compression}")
        logger.info(f"[NukeWrite]   Channels: {', '.join(channel_names)}")

        ui_images = []
        if show_preview and image is not None and image.shape[0] > 0:
            ui_images = create_preview_images(image)

        return {"ui": {"images": ui_images}, "result": (image, output_path)}


def _resize_pass(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear-resize one (H, W, C) pass to (height, width)."""
    if cv2 is not None:
        out = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
        if out.ndim == 2:
            out = out[..., np.newaxis]
        return np.ascontiguousarray(out, dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
    return np.ascontiguousarray(t.squeeze(0).permute(1, 2, 0).numpy(), dtype=np.float32)


def _convert_pass(arr: np.ndarray, colorspace: str) -> np.ndarray:
    """Working space -> ``colorspace`` for one light pass (H, W, C).

    RGB / RGBA passes go through OCIO directly (alpha is preserved). A
    single-channel light pass is converted as a grey triplet and the first
    channel kept. Two-channel passes are left alone (no colour meaning).
    """
    C = arr.shape[-1]
    if C in (3, 4):
        return _from_working_space(arr, colorspace)
    if C == 1:
        grey = np.repeat(arr, 3, axis=-1)
        return np.ascontiguousarray(_from_working_space(grey, colorspace)[..., :1])
    return arr


class NukeReadInfo(NukeNodeBase):
    """
    Read Info node - displays information about an image file or sequence.

    Shows: resolution, channels, bit depth, frame range, file size, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "Nuke/IO"
    OUTPUT_NODE = True

    def get_info(self, file_path):
        """Get image file information."""

        if not file_path:
            return ("No file path specified",)

        # Expand path
        file_path = os.path.expandvars(os.path.expanduser(file_path))

        # Detect sequence
        pattern, frames, padding = detect_sequence(file_path)
        is_sequence = padding > 0 and len(frames) > 1

        info = f"File Path: {file_path}\n"

        if is_sequence:
            info += f"\n=== SEQUENCE INFO ===\n"
            info += f"Pattern: {pattern}\n"
            info += f"Frame Range: {min(frames)}-{max(frames)}\n"
            info += f"Total Frames: {len(frames)}\n"
            info += f"Padding: {padding} digits\n"

            # Check for missing frames
            expected = set(range(min(frames), max(frames) + 1))
            missing = expected - set(frames)
            if missing:
                info += f"Missing Frames: {len(missing)}\n"
                if len(missing) <= 10:
                    info += f"  {sorted(missing)}\n"

            # Use first frame for detailed info
            sample_path = expand_frame_pattern(pattern, frames[0], padding)
        else:
            sample_path = file_path

        # Get file info
        if os.path.exists(sample_path):
            file_size = os.path.getsize(sample_path)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024*1024):.2f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.2f} KB"
            else:
                size_str = f"{file_size} bytes"

            info += f"\n=== FILE INFO ===\n"
            info += f"File Size: {size_str}\n"

            # Try to get image info
            if OIIO_AVAILABLE:
                try:
                    inp = oiio.ImageInput.open(sample_path)
                    if inp:
                        spec = inp.spec()
                        info += f"Resolution: {spec.width} x {spec.height}\n"
                        info += f"Channels: {spec.nchannels} ({', '.join(spec.channelnames)})\n"
                        info += f"Bit Depth: {spec.format}\n"

                        # Get compression for EXR
                        compression = spec.get_string_attribute("compression", "")
                        if compression:
                            info += f"Compression: {compression}\n"

                        # Get colorspace
                        colorspace = spec.get_string_attribute("oiio:ColorSpace", "")
                        if colorspace:
                            info += f"Color Space: {colorspace}\n"

                        inp.close()
                    else:
                        info += f"OpenImageIO could not open file: {oiio.geterror()}\n"
                except Exception as e:
                    info += f"Error reading metadata: {e}\n"
            else:
                info += "Image metadata unavailable: OpenImageIO is not installed\n"
        else:
            info += f"\nFile not found: {sample_path}\n"

        # Show available libraries
        info += f"\n=== I/O LIBRARIES ===\n"
        if OIIO_AVAILABLE:
            version = oiio_version()
            info += f"OpenImageIO: Available{' (' + version + ')' if version else ''}\n"
        else:
            info += "OpenImageIO: Not installed (required by nuke-nodes 2.x)\n"
        if ocio_config.OCIO_AVAILABLE:
            info += f"OpenColorIO: Available ({ocio_config.OCIO_VERSION})\n"
            info += f"OCIO config: {ocio_config.config_source()}\n"
            info += f"Working space: {ocio_config.scene_linear_name()}\n"
        else:
            info += "OpenColorIO: Not installed (colorspace conversion disabled)\n"

        # Show supported formats
        formats = get_supported_formats()
        info += f"\nRead formats: {', '.join(formats['read'][:15])}..."
        info += f"\nWrite formats: {', '.join(formats['write'][:10])}..."

        return (info,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeRead": NukeRead,
    "NukeWrite": NukeWrite,
    "NukeReadInfo": NukeReadInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeRead": "Nuke Read",
    "NukeWrite": "Nuke Write",
    "NukeReadInfo": "Nuke Read Info",
}
