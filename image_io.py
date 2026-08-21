"""
OpenImageIO-backed image reading and writing for the Nuke I/O nodes.

nuke-nodes 2.x is OpenImageIO-only: there is no OpenCV / PIL fallback.
``read_image`` / ``write_image`` raise ``RuntimeError`` when OpenImageIO is
not importable so a missing dependency surfaces as a clear error instead of
silently degraded output.

Supported formats (OIIO):
- EXR (OpenEXR) - 16/32-bit float, multiple compression options
- TIFF - 8/16/32-bit, various compression
- PNG - 8/16-bit with alpha
- JPEG/JPG - 8-bit
- DPX - 10/16-bit (common in film)
- Cineon - 10-bit log
- HDR/RGBE - HDR radiance format
- TGA/Targa - 8-bit with alpha
- BMP - 8-bit
- PSD - Photoshop (read-only)
- RAW formats - via LibRaw plugin
- And many more...
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

OIIO_AVAILABLE = False
try:
    import OpenImageIO as oiio

    OIIO_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the host environment
    oiio = None

OIIO_MISSING_MESSAGE = "OpenImageIO is required by nuke-nodes 2.x - pip install OpenImageIO"


def _require_oiio() -> None:
    if not OIIO_AVAILABLE:
        raise RuntimeError(OIIO_MISSING_MESSAGE)


def oiio_version() -> str:
    """Version string of the imported OpenImageIO, or '' when missing."""
    if not OIIO_AVAILABLE:
        return ""
    return str(getattr(oiio, "__version__", "") or getattr(oiio, "VERSION_STRING", ""))


# ============================================================================
# Readers
# ============================================================================


def read_image_oiio(filepath: str) -> Optional[np.ndarray]:
    """Read an image with OpenImageIO as float32 (H, W, C), or None on error."""
    if not OIIO_AVAILABLE:
        return None

    try:
        inp = oiio.ImageInput.open(filepath)
        if inp is None:
            logger.warning(f"[NukeRead] OIIO error: {oiio.geterror()}")
            return None

        spec = inp.spec()
        pixels = inp.read_image("float")
        inp.close()

        if pixels is None:
            return None

        # Reshape to (H, W, C)
        pixels = np.array(pixels, dtype=np.float32)
        pixels = pixels.reshape(spec.height, spec.width, spec.nchannels)

        return pixels
    except Exception as e:
        logger.warning(f"[NukeRead] OIIO error reading {filepath}: {e}")
        return None


def read_image(filepath: str) -> Optional[np.ndarray]:
    """
    Read an image file with OpenImageIO.

    Returns float32 (H, W, C) pixels, or None when the file is missing or
    cannot be decoded. Raises RuntimeError when OpenImageIO is not installed.
    """
    _require_oiio()

    if not os.path.exists(filepath):
        logger.warning(f"[NukeRead] File not found: {filepath}")
        return None

    img = read_image_oiio(filepath)
    if img is None:
        logger.error(f"[NukeRead] OpenImageIO could not read: {filepath}")
    return img


# ============================================================================
# Writers
# ============================================================================


def write_image_oiio(
    filepath: str,
    pixels: np.ndarray,
    bit_depth: str = "16",
    compression: str = "zip",
    metadata: Optional[Dict] = None,
) -> bool:
    """Write image using OpenImageIO."""
    if not OIIO_AVAILABLE:
        return False

    try:
        height, width = pixels.shape[:2]
        channels = pixels.shape[2] if len(pixels.shape) > 2 else 1

        # Determine output format based on bit depth
        if bit_depth == "8":
            format_type = oiio.UINT8
            pixels_out = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
        elif bit_depth == "16":
            format_type = oiio.UINT16
            pixels_out = (np.clip(pixels, 0, 1) * 65535).astype(np.uint16)
        elif bit_depth == "16f":
            format_type = oiio.HALF
            pixels_out = pixels.astype(np.float16)
        elif bit_depth == "32f":
            format_type = oiio.FLOAT
            pixels_out = pixels.astype(np.float32)
        else:
            format_type = oiio.UINT16
            pixels_out = (np.clip(pixels, 0, 1) * 65535).astype(np.uint16)

        # Ensure array is contiguous in memory for OIIO
        pixels_out = np.ascontiguousarray(pixels_out)

        # Create spec
        spec = oiio.ImageSpec(width, height, channels, format_type)

        # Set compression for EXR
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".exr"]:
            spec.attribute("compression", compression)
        elif ext in [".png"]:
            spec.attribute("png:compressionLevel", 6)
        elif ext in [".jpg", ".jpeg"]:
            spec.attribute("jpeg:quality", 95)
        elif ext in [".tif", ".tiff"]:
            if compression == "none":
                spec.attribute("compression", "none")
            elif compression in ["lzw", "zip", "deflate"]:
                spec.attribute("compression", compression)
        elif ext in [".webp"]:
            # WebP only supports 8-bit, force conversion
            format_type = oiio.UINT8
            pixels_out = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
            pixels_out = np.ascontiguousarray(pixels_out)
            spec = oiio.ImageSpec(width, height, channels, format_type)
            spec.attribute("webp:quality", 90)

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                spec.attribute(key, value)

        # Create output
        out = oiio.ImageOutput.create(filepath)
        if out is None:
            logger.warning(f"[NukeWrite] OIIO error: {oiio.geterror()}")
            return False

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        if not out.open(filepath, spec):
            logger.warning(f"[NukeWrite] OIIO error opening: {out.geterror()}")
            return False

        if not out.write_image(pixels_out):
            logger.warning(f"[NukeWrite] OIIO error writing: {out.geterror()}")
            out.close()
            return False

        out.close()
        return True

    except Exception as e:
        logger.warning(f"[NukeWrite] OIIO error writing {filepath}: {e}")
        return False


def write_image(
    filepath: str,
    pixels: np.ndarray,
    bit_depth: str = "16",
    compression: str = "zip",
    metadata: Optional[Dict] = None,
) -> bool:
    """
    Write an image file with OpenImageIO.

    Returns True on success. Raises RuntimeError when OpenImageIO is not
    installed.
    """
    _require_oiio()

    if write_image_oiio(filepath, pixels, bit_depth, compression, metadata):
        return True

    logger.error(f"[NukeWrite] OpenImageIO could not write: {filepath}")
    return False


# ============================================================================
# Format listing
# ============================================================================


def get_supported_formats() -> Dict[str, List[str]]:
    """Get dictionary of supported image formats."""
    formats: Dict[str, List[str]] = {"read": [], "write": []}

    if OIIO_AVAILABLE:
        # OIIO supports many formats
        formats["read"].extend(
            [
                "exr",
                "tif",
                "tiff",
                "png",
                "jpg",
                "jpeg",
                "dpx",
                "cin",
                "hdr",
                "rgbe",
                "tga",
                "bmp",
                "psd",
                "gif",
                "webp",
                "heic",
                "avif",
                "raw",
                "cr2",
                "nef",
                "arw",
                "dng",
                "fits",
                "sgi",
                "pic",
                "pnm",
                "pbm",
                "pgm",
                "ppm",
                "rla",
                "iff",
                "ico",
            ]
        )
        formats["write"].extend(
            [
                "exr",
                "tif",
                "tiff",
                "png",
                "jpg",
                "jpeg",
                "dpx",
                "hdr",
                "tga",
                "bmp",
                "webp",
                "pnm",
                "pbm",
                "pgm",
                "ppm",
            ]
        )

    # Remove duplicates
    formats["read"] = sorted(set(formats["read"]))
    formats["write"] = sorted(set(formats["write"]))

    return formats
