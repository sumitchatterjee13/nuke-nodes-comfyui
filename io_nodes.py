"""
Read and Write nodes for loading and saving images, similar to Nuke's Read/Write nodes.

Supports a wide variety of image formats through OpenImageIO (OIIO) or fallback to
OpenCV/PIL. Includes full support for image sequences with frame pattern matching.

Supported formats (with OIIO):
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

Sequence patterns supported:
- %04d style (printf format): image.%04d.exr
- #### style (hash padding): image.####.exr
- Frame ranges: 1-100, 1-100x2 (every 2nd frame)
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import torch

import folder_paths
from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor

# Try to import OpenImageIO
OIIO_AVAILABLE = False
try:
    import OpenImageIO as oiio
    OIIO_AVAILABLE = True
except ImportError:
    oiio = None

# Fallback to OpenCV
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None

# Fallback to PIL
PIL_AVAILABLE = False
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PILImage = None


# ============================================================================
# Sequence Pattern Utilities
# ============================================================================

def parse_frame_pattern(filepath: str) -> Tuple[str, Optional[str], int]:
    """
    Parse a filepath to detect frame pattern and padding.

    Supports:
    - %04d style: image.%04d.exr
    - #### style: image.####.exr
    - Literal frame number: image.0001.exr

    Returns:
        (base_pattern, frame_spec, padding)
        - base_pattern: pattern with %0Nd placeholder
        - frame_spec: original frame specifier or None
        - padding: number of digits for padding
    """
    # Normalize path separators for consistent handling
    filepath = filepath.replace('\\', '/')

    # Check for %0Nd pattern
    match = re.search(r'%(\d*)d', filepath)
    if match:
        padding = int(match.group(1)) if match.group(1) else 4
        return filepath, match.group(0), padding

    # Check for #### pattern
    match = re.search(r'(#+)', filepath)
    if match:
        hashes = match.group(1)
        padding = len(hashes)
        pattern = filepath.replace(hashes, f'%0{padding}d')
        return pattern, hashes, padding

    # Check for frame number in filename (e.g., image.0001.exr)
    # Match number before extension
    match = re.search(r'(\d+)(\.[^.]+)$', filepath)
    if match:
        frame_str = match.group(1)
        padding = len(frame_str)
        ext = match.group(2)
        base = filepath[:match.start()]
        pattern = f"{base}%0{padding}d{ext}"
        return pattern, frame_str, padding

    return filepath, None, 0


def expand_frame_pattern(pattern: str, frame: int, padding: int = 4) -> str:
    """
    Expand a frame pattern to an actual filename.

    Args:
        pattern: Pattern with %0Nd or #### placeholder
        frame: Frame number
        padding: Digit padding

    Returns:
        Expanded filename
    """
    # Handle %0Nd pattern
    if '%' in pattern:
        return pattern % frame

    # Handle #### pattern
    if '#' in pattern:
        hashes = re.search(r'#+', pattern).group(0)
        return pattern.replace(hashes, str(frame).zfill(len(hashes)))

    return pattern


def detect_sequence(filepath: str) -> Tuple[str, List[int], int]:
    """
    Detect an image sequence from a single file path.

    Args:
        filepath: Path to one file in the sequence

    Returns:
        (pattern, frames, padding)
        - pattern: Frame pattern string
        - frames: List of available frame numbers
        - padding: Digit padding
    """
    # Normalize path separators
    filepath = filepath.replace('\\', '/')

    pattern, frame_spec, padding = parse_frame_pattern(filepath)

    if frame_spec is None or padding == 0:
        # Not a sequence, single file
        if os.path.exists(filepath):
            return filepath, [0], 0
        return filepath, [], 0

    # Find all matching files
    # Convert pattern to glob pattern
    glob_pattern = re.sub(r'%\d*d', '*', pattern)
    glob_pattern = re.sub(r'#+', '*', glob_pattern)

    print(f"[NukeRead] Searching for sequence with pattern: {glob_pattern}")

    matching_files = glob.glob(glob_pattern)

    if not matching_files:
        print(f"[NukeRead] No files found matching pattern: {glob_pattern}")
        # Check if directory exists
        directory = os.path.dirname(glob_pattern)
        if os.path.exists(directory):
            print(f"[NukeRead] Directory exists: {directory}")
            # List files in directory for debugging
            try:
                files = os.listdir(directory)
                print(f"[NukeRead] Files in directory: {files[:10]}...")  # Show first 10
            except Exception as e:
                print(f"[NukeRead] Error listing directory: {e}")
        else:
            print(f"[NukeRead] Directory does not exist: {directory}")
        return pattern, [], padding

    print(f"[NukeRead] Found {len(matching_files)} files in sequence")

    # Extract frame numbers
    frames = []
    for f in matching_files:
        # Extract number from filename
        match = re.search(r'(\d+)\.[^.]+$', f)
        if match:
            frames.append(int(match.group(1)))

    frames.sort()
    return pattern, frames, padding


def parse_frame_range(range_str: str) -> List[int]:
    """
    Parse a frame range string like "1-100" or "1-100x2" (every 2nd frame).

    Args:
        range_str: Frame range string

    Returns:
        List of frame numbers
    """
    if not range_str or range_str.strip() == "":
        return []

    frames = []

    for part in range_str.split(','):
        part = part.strip()

        # Check for step (x2)
        step = 1
        if 'x' in part:
            part, step_str = part.split('x')
            step = int(step_str)

        # Check for range (-)
        if '-' in part:
            start, end = part.split('-')
            frames.extend(range(int(start), int(end) + 1, step))
        else:
            frames.append(int(part))

    return sorted(set(frames))


# ============================================================================
# File Counter Utilities
# ============================================================================

def _is_windows() -> bool:
    """Check if running on Windows."""
    import sys
    return sys.platform.startswith('win') or sys.platform == 'cygwin' or sys.platform == 'msys'


def _normalize_for_comparison(filename: str) -> str:
    """
    Normalize filename for comparison based on platform.
    Windows filesystem is case-insensitive, Linux/Mac are case-sensitive.
    """
    if _is_windows():
        return filename.lower()
    return filename


def _file_exists_case_aware(filepath: str) -> bool:
    """
    Check if file exists, handling case sensitivity properly for each platform.
    On Windows (case-insensitive), os.path.exists() already handles this.
    On Linux/Mac (case-sensitive), os.path.exists() is already correct.
    """
    # os.path.exists handles platform-specific case sensitivity correctly
    return os.path.exists(filepath)


def get_unique_filepath(filepath: str) -> str:
    """
    Get a unique filepath that doesn't overwrite any existing file.

    Handles various naming patterns and preserves zero-padding:
    - image_0001.exr -> image_0002.exr -> image_0003.exr (preserves padding)
    - image_1.exr -> image_2.exr -> image_3.exr
    - image-1.exr -> image-2.exr -> image-3.exr
    - image.png -> image1.png -> image2.png

    Works correctly on both Windows (case-insensitive) and Linux/Mac (case-sensitive).

    Args:
        filepath: The desired output filepath

    Returns:
        A filepath that is guaranteed not to exist
    """
    # If file doesn't exist, use it as-is
    if not _file_exists_case_aware(filepath):
        return filepath

    directory = os.path.dirname(filepath) or '.'
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)

    # Build a set of existing filenames for efficient lookup
    # Normalize for case-insensitive comparison on Windows
    try:
        existing_files = set()
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                existing_files.add(_normalize_for_comparison(f))
    except (OSError, PermissionError):
        existing_files = set()

    def _exists(name: str) -> bool:
        """Check if a filename exists in the directory (case-aware)."""
        normalized = _normalize_for_comparison(name)
        if normalized in existing_files:
            return True
        # Double-check with filesystem (handles race conditions)
        return _file_exists_case_aware(os.path.join(directory, name))

    # Pattern 1: Check if base already ends with a separator and number (e.g., image_0001, image-3)
    # Match patterns like: name_0001, name-123, name.123
    separator_pattern = re.match(r'^(.+?)([_\-\.])(\d+)$', base)

    if separator_pattern:
        # File already has a separator+number pattern, increment it
        prefix = separator_pattern.group(1)
        separator = separator_pattern.group(2)
        num_str = separator_pattern.group(3)
        current_num = int(num_str)
        # Preserve the original padding (e.g., 0001 has padding of 4)
        padding = len(num_str)

        # Find next available number
        num = current_num + 1
        max_attempts = 100000
        while num < current_num + max_attempts:
            # Use zfill to preserve padding
            new_filename = f"{prefix}{separator}{str(num).zfill(padding)}{ext}"
            if not _exists(new_filename):
                return os.path.join(directory, new_filename)
            num += 1

        # Fallback: use timestamp
        import time
        timestamp = int(time.time() * 1000)
        return os.path.join(directory, f"{prefix}{separator}{timestamp}{ext}")

    # Pattern 2: Check if base ends with a number directly (e.g., image2, render001)
    direct_number_pattern = re.match(r'^(.+?)(\d+)$', base)

    if direct_number_pattern:
        prefix = direct_number_pattern.group(1)
        num_str = direct_number_pattern.group(2)
        current_num = int(num_str)
        # Preserve the original padding
        padding = len(num_str)

        # Find next available number
        num = current_num + 1
        max_attempts = 100000
        while num < current_num + max_attempts:
            # Use zfill to preserve padding
            new_filename = f"{prefix}{str(num).zfill(padding)}{ext}"
            if not _exists(new_filename):
                return os.path.join(directory, new_filename)
            num += 1

        # Fallback: use timestamp
        import time
        timestamp = int(time.time() * 1000)
        return os.path.join(directory, f"{prefix}{timestamp}{ext}")

    # Pattern 3: No number in filename, start with 1
    # Try appending number directly: image.png -> image1.png
    num = 1
    max_attempts = 100000
    while num < max_attempts:
        new_filename = f"{base}{num}{ext}"
        if not _exists(new_filename):
            return os.path.join(directory, new_filename)
        num += 1

    # Ultimate fallback: use timestamp
    import time
    timestamp = int(time.time() * 1000)
    return os.path.join(directory, f"{base}_{timestamp}{ext}")


# ============================================================================
# Image I/O Functions
# ============================================================================

def read_image_oiio(filepath: str) -> Optional[np.ndarray]:
    """Read image using OpenImageIO."""
    if not OIIO_AVAILABLE:
        return None

    try:
        inp = oiio.ImageInput.open(filepath)
        if inp is None:
            print(f"[NukeRead] OIIO error: {oiio.geterror()}")
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
        print(f"[NukeRead] OIIO error reading {filepath}: {e}")
        return None


def read_image_cv2(filepath: str) -> Optional[np.ndarray]:
    """Read image using OpenCV."""
    if not CV2_AVAILABLE:
        return None

    try:
        # Read with alpha channel if present
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        # Convert BGR(A) to RGB(A)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1 range
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)

        return img
    except Exception as e:
        print(f"[NukeRead] OpenCV error reading {filepath}: {e}")
        return None


def read_image_pil(filepath: str) -> Optional[np.ndarray]:
    """Read image using PIL."""
    if not PIL_AVAILABLE:
        return None

    try:
        img = PILImage.open(filepath)
        img = np.array(img, dtype=np.float32)

        # Normalize to 0-1 range
        if img.max() > 1.0:
            if img.max() > 255:
                img /= 65535.0
            else:
                img /= 255.0

        # Ensure 3D array (add channel dim if grayscale)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]

        return img
    except Exception as e:
        print(f"[NukeRead] PIL error reading {filepath}: {e}")
        return None


def read_image(filepath: str) -> Optional[np.ndarray]:
    """
    Read an image file using the best available library.

    Priority: OpenImageIO > OpenCV > PIL
    """
    if not os.path.exists(filepath):
        print(f"[NukeRead] File not found: {filepath}")
        return None

    # Try OIIO first (best format support)
    if OIIO_AVAILABLE:
        img = read_image_oiio(filepath)
        if img is not None:
            return img

    # Fallback to OpenCV
    if CV2_AVAILABLE:
        img = read_image_cv2(filepath)
        if img is not None:
            return img

    # Fallback to PIL
    if PIL_AVAILABLE:
        img = read_image_pil(filepath)
        if img is not None:
            return img

    print(f"[NukeRead] No library available to read: {filepath}")
    return None


def write_image_oiio(filepath: str, pixels: np.ndarray,
                     bit_depth: str = "16", compression: str = "zip",
                     metadata: Optional[Dict] = None) -> bool:
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
        if ext in ['.exr']:
            spec.attribute("compression", compression)
        elif ext in ['.png']:
            spec.attribute("png:compressionLevel", 6)
        elif ext in ['.jpg', '.jpeg']:
            spec.attribute("jpeg:quality", 95)
        elif ext in ['.tif', '.tiff']:
            if compression == "none":
                spec.attribute("compression", "none")
            elif compression in ["lzw", "zip", "deflate"]:
                spec.attribute("compression", compression)

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                spec.attribute(key, value)

        # Create output
        out = oiio.ImageOutput.create(filepath)
        if out is None:
            print(f"[NukeWrite] OIIO error: {oiio.geterror()}")
            return False

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        if not out.open(filepath, spec):
            print(f"[NukeWrite] OIIO error opening: {out.geterror()}")
            return False

        if not out.write_image(pixels_out):
            print(f"[NukeWrite] OIIO error writing: {out.geterror()}")
            out.close()
            return False

        out.close()
        return True

    except Exception as e:
        print(f"[NukeWrite] OIIO error writing {filepath}: {e}")
        return False


def write_image_cv2(filepath: str, pixels: np.ndarray,
                    bit_depth: str = "16") -> bool:
    """Write image using OpenCV."""
    if not CV2_AVAILABLE:
        return False

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        # Convert to output format
        if bit_depth == "8":
            pixels_out = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
        elif bit_depth == "16":
            pixels_out = (np.clip(pixels, 0, 1) * 65535).astype(np.uint16)
        else:
            pixels_out = (np.clip(pixels, 0, 1) * 65535).astype(np.uint16)

        # Convert RGB(A) to BGR(A)
        if len(pixels_out.shape) == 3:
            if pixels_out.shape[2] == 4:
                pixels_out = cv2.cvtColor(pixels_out, cv2.COLOR_RGBA2BGRA)
            elif pixels_out.shape[2] == 3:
                pixels_out = cv2.cvtColor(pixels_out, cv2.COLOR_RGB2BGR)

        return cv2.imwrite(filepath, pixels_out)

    except Exception as e:
        print(f"[NukeWrite] OpenCV error writing {filepath}: {e}")
        return False


def write_image_pil(filepath: str, pixels: np.ndarray,
                    bit_depth: str = "8") -> bool:
    """Write image using PIL."""
    if not PIL_AVAILABLE:
        return False

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        # Convert to 8-bit for PIL
        pixels_out = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)

        # Remove extra dimensions
        if len(pixels_out.shape) == 3 and pixels_out.shape[2] == 1:
            pixels_out = pixels_out[:, :, 0]

        img = PILImage.fromarray(pixels_out)
        img.save(filepath)
        return True

    except Exception as e:
        print(f"[NukeWrite] PIL error writing {filepath}: {e}")
        return False


def write_image(filepath: str, pixels: np.ndarray,
                bit_depth: str = "16", compression: str = "zip",
                metadata: Optional[Dict] = None) -> bool:
    """
    Write an image file using the best available library.

    Priority: OpenImageIO > OpenCV > PIL
    """
    # Try OIIO first (best format support)
    if OIIO_AVAILABLE:
        if write_image_oiio(filepath, pixels, bit_depth, compression, metadata):
            return True

    # Fallback to OpenCV
    if CV2_AVAILABLE:
        if write_image_cv2(filepath, pixels, bit_depth):
            return True

    # Fallback to PIL
    if PIL_AVAILABLE:
        if write_image_pil(filepath, pixels, bit_depth):
            return True

    print(f"[NukeWrite] No library available to write: {filepath}")
    return False


def get_supported_formats() -> Dict[str, List[str]]:
    """Get dictionary of supported image formats."""
    formats = {
        "read": [],
        "write": []
    }

    if OIIO_AVAILABLE:
        # OIIO supports many formats
        formats["read"].extend([
            "exr", "tif", "tiff", "png", "jpg", "jpeg", "dpx", "cin",
            "hdr", "rgbe", "tga", "bmp", "psd", "gif", "webp", "heic",
            "avif", "raw", "cr2", "nef", "arw", "dng", "fits", "sgi",
            "pic", "pnm", "pbm", "pgm", "ppm", "rla", "iff", "ico"
        ])
        formats["write"].extend([
            "exr", "tif", "tiff", "png", "jpg", "jpeg", "dpx",
            "hdr", "tga", "bmp", "webp", "pnm", "pbm", "pgm", "ppm"
        ])

    if CV2_AVAILABLE:
        cv2_formats = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]
        formats["read"].extend(cv2_formats)
        formats["write"].extend(cv2_formats)

    if PIL_AVAILABLE:
        pil_formats = ["png", "jpg", "jpeg", "gif", "bmp", "tga", "webp"]
        formats["read"].extend(pil_formats)
        formats["write"].extend(pil_formats)

    # Remove duplicates
    formats["read"] = sorted(set(formats["read"]))
    formats["write"] = sorted(set(formats["write"]))

    return formats


# ============================================================================
# Preview Utilities
# ============================================================================

def resize_image_oiio(img_np: np.ndarray, max_size: int = 256) -> np.ndarray:
    """
    Resize image using OpenImageIO's ImageBufAlgo.

    Args:
        img_np: Image as numpy array (H, W, C)
        max_size: Maximum dimension

    Returns:
        Resized image as numpy array
    """
    if not OIIO_AVAILABLE:
        # Fallback to simple numpy resize (nearest neighbor)
        height, width = img_np.shape[:2]
        if width <= max_size and height <= max_size:
            return img_np

        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        # Simple nearest neighbor resize
        import cv2 as cv2_resize
        if CV2_AVAILABLE:
            return cv2_resize.resize(img_np, (new_width, new_height), interpolation=cv2_resize.INTER_LANCZOS4)
        else:
            # Very basic resize using numpy
            y_indices = np.linspace(0, height - 1, new_height).astype(int)
            x_indices = np.linspace(0, width - 1, new_width).astype(int)
            return img_np[np.ix_(y_indices, x_indices)]

    height, width = img_np.shape[:2]
    channels = img_np.shape[2] if len(img_np.shape) > 2 else 1

    if width <= max_size and height <= max_size:
        return img_np

    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    # Create ImageBuf from numpy array
    spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    src_buf = oiio.ImageBuf(spec)
    # Ensure array is contiguous in memory for OIIO
    pixels_contiguous = np.ascontiguousarray(img_np.astype(np.float32))
    src_buf.set_pixels(oiio.ROI(0, width, 0, height, 0, 1, 0, channels), pixels_contiguous)

    # Resize using OIIO
    dst_buf = oiio.ImageBufAlgo.resize(src_buf, roi=oiio.ROI(0, new_width, 0, new_height, 0, 1, 0, channels))

    # Get pixels back
    resized = dst_buf.get_pixels(oiio.FLOAT)
    return resized.reshape(new_height, new_width, channels)


def save_preview_oiio(img_np: np.ndarray, filepath: str) -> bool:
    """
    Save preview image using OpenImageIO.

    Args:
        img_np: Image as numpy array (H, W, C) in 0-1 range float or 0-255 uint8
        filepath: Output filepath (should be .png or .jpg)

    Returns:
        True if successful
    """
    if not OIIO_AVAILABLE:
        # Fallback to OpenCV
        if CV2_AVAILABLE:
            # Ensure uint8
            if img_np.dtype != np.uint8:
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            return cv2.imwrite(filepath, img_np)
        return False

    height, width = img_np.shape[:2]
    channels = img_np.shape[2] if len(img_np.shape) > 2 else 1

    # Convert to uint8 for PNG output
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        pixels_out = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    else:
        pixels_out = img_np.astype(np.uint8)

    # Ensure array is contiguous in memory for OIIO
    pixels_out = np.ascontiguousarray(pixels_out)

    # Create spec and output
    spec = oiio.ImageSpec(width, height, channels, oiio.UINT8)
    spec.attribute("png:compressionLevel", 6)

    out = oiio.ImageOutput.create(filepath)
    if out is None:
        return False

    if not out.open(filepath, spec):
        return False

    if not out.write_image(pixels_out):
        out.close()
        return False

    out.close()
    return True


def create_preview_images(images: torch.Tensor, max_size: int = 256, max_frames: int = 1000) -> list:
    """
    Create preview images for display in the node UI.
    Uses OpenImageIO for image processing and saving.

    Args:
        images: Tensor of images (B, H, W, C)
        max_size: Maximum dimension for preview thumbnails
        max_frames: Maximum number of frames to include in preview

    Returns:
        List of preview dictionaries for ComfyUI UI
    """
    previews = []
    batch_size = images.shape[0]

    # Limit number of frames for preview
    frame_step = max(1, batch_size // max_frames)

    # Get temp directory
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(0, batch_size, frame_step):
        if len(previews) >= max_frames:
            break

        img_tensor = images[i]
        img_np = img_tensor.cpu().numpy()

        # Ensure we have 3 channels (RGB)
        if img_np.shape[-1] == 4:
            # RGBA -> RGB (discard alpha for preview)
            img_np = img_np[:, :, :3]
        elif img_np.shape[-1] == 1:
            # Grayscale -> RGB
            img_np = np.concatenate([img_np, img_np, img_np], axis=-1)

        # Resize using OIIO
        img_np = resize_image_oiio(img_np, max_size)

        # Save to temporary file
        preview_filename = f"nuke_preview_{id(images)}_{i}.png"
        preview_path = os.path.join(temp_dir, preview_filename)

        if save_preview_oiio(img_np, preview_path):
            previews.append({
                "filename": preview_filename,
                "subfolder": "",
                "type": "temp",
                "frame": i + 1
            })

    return previews


def save_preview_to_temp(img_np: np.ndarray, suffix: str = "") -> dict:
    """
    Save a single numpy image to temp directory for preview.
    Uses OpenImageIO for image processing and saving.

    Args:
        img_np: Image as numpy array (H, W, C) in 0-1 range
        suffix: Optional suffix for filename

    Returns:
        Preview dictionary for ComfyUI UI
    """
    import uuid

    # Ensure we have 3 channels (RGB)
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np, img_np, img_np], axis=-1)
    elif img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    elif img_np.shape[-1] == 1:
        img_np = np.concatenate([img_np, img_np, img_np], axis=-1)

    # Resize for preview (max 256px)
    img_np = resize_image_oiio(img_np, max_size=256)

    # Save to temp directory
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    preview_filename = f"nuke_preview_{uuid.uuid4().hex[:8]}{suffix}.png"
    preview_path = os.path.join(temp_dir, preview_filename)

    if save_preview_oiio(img_np, preview_path):
        return {
            "filename": preview_filename,
            "subfolder": "",
            "type": "temp"
        }

    return None


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
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to image or sequence (e.g., /path/image.%04d.exr)",
                }),
                "frame": ("INT", {
                    "default": 1,
                    "min": -999999,
                    "max": 999999,
                    "step": 1,
                }),
            },
            "optional": {
                "load_as_sequence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Treat path as image sequence (enable for patterns like ####, %04d)"
                }),
                "first_frame": ("INT", {
                    "default": 1,
                    "min": -999999,
                    "max": 999999,
                    "step": 1,
                }),
                "last_frame": ("INT", {
                    "default": 1,
                    "min": -999999,
                    "max": 999999,
                    "step": 1,
                }),
                "frame_mode": (["single", "range", "all"], {"default": "single"}),
                "missing_frames": (["error", "black", "hold", "nearest"], {"default": "black"}),
                "colorspace": (["raw", "sRGB", "linear", "ACEScg"], {"default": "raw"}),
                "show_preview": ("BOOLEAN", {"default": True, "tooltip": "Show thumbnail preview in node"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "read_image"
    CATEGORY = "Nuke/IO"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def read_image(self, file_path, frame, load_as_sequence=True, first_frame=1, last_frame=1,
                   frame_mode="single", missing_frames="black", colorspace="raw",
                   show_preview=True):
        """Read image(s) from disk."""

        if not file_path:
            print("[NukeRead] No file path specified")
            # Return black image
            result = torch.zeros((1, 512, 512, 3))
            return {"ui": {"images": []}, "result": (result,)}

        # Expand environment variables and user home
        file_path = os.path.expandvars(os.path.expanduser(file_path))

        print(f"[NukeRead] Loading path: {file_path}")
        print(f"[NukeRead] Load as sequence: {load_as_sequence}")

        # Parse frame pattern
        pattern, frame_spec, padding = parse_frame_pattern(file_path)
        is_sequence = load_as_sequence and (frame_spec is not None and padding > 0)

        if frame_spec and padding > 0:
            print(f"[NukeRead] Detected pattern: {pattern}, padding: {padding}")
        else:
            print(f"[NukeRead] No sequence pattern detected, treating as single file")

        # Determine frames to load
        if frame_mode == "single":
            frames_to_load = [frame]
        elif frame_mode == "range":
            frames_to_load = list(range(first_frame, last_frame + 1))
        elif frame_mode == "all" and is_sequence:
            _, available_frames, _ = detect_sequence(file_path)
            frames_to_load = available_frames if available_frames else [frame]
        else:
            frames_to_load = [frame]

        # Get available frames for nearest/hold modes
        if is_sequence and missing_frames in ["hold", "nearest"]:
            _, available_frames, _ = detect_sequence(file_path)
        else:
            available_frames = []

        # Load images
        images = []
        reference_shape = None

        for f in frames_to_load:
            if is_sequence:
                actual_path = expand_frame_pattern(pattern, f, padding)
            else:
                actual_path = file_path

            img = None

            if os.path.exists(actual_path):
                img = read_image(actual_path)
            else:
                # Handle missing frames
                if missing_frames == "error":
                    print(f"[NukeRead] Frame not found: {actual_path}")
                elif missing_frames == "hold" and images:
                    # Use previous frame
                    img = images[-1].copy() if images else None
                elif missing_frames == "nearest" and available_frames:
                    # Find nearest available frame
                    nearest = min(available_frames, key=lambda x: abs(x - f))
                    nearest_path = expand_frame_pattern(pattern, nearest, padding)
                    img = read_image(nearest_path)
                # "black" - will create black image below

            if img is None:
                # Create black image with reference size
                if reference_shape:
                    img = np.zeros(reference_shape, dtype=np.float32)
                else:
                    img = np.zeros((512, 512, 3), dtype=np.float32)
            else:
                if reference_shape is None:
                    reference_shape = img.shape

            # Ensure consistent channel count (minimum 3 channels for ComfyUI)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 1:
                img = np.concatenate([img, img, img], axis=-1)

            # Apply colorspace conversion (basic)
            if colorspace == "sRGB" and img.max() <= 1.0:
                # Linear to sRGB
                img = np.where(img <= 0.0031308,
                              img * 12.92,
                              1.055 * np.power(np.clip(img, 0.0031308, None), 1/2.4) - 0.055)
            elif colorspace == "linear":
                # sRGB to linear (assume input is sRGB)
                img = np.where(img <= 0.04045,
                              img / 12.92,
                              np.power((img + 0.055) / 1.055, 2.4))

            images.append(img)

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
            "none", "rle", "zip", "zips", "piz",
            "pxr24", "b44", "b44a", "dwaa", "dwab"
        ]

        # Common bit depths
        bit_depths = ["8", "16", "16f", "32f"]

        return {
            "required": {
                "image": ("IMAGE",),
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Output path (e.g., output.%04d.exr or output.#### or output)",
                }),
                "frame_start": ("INT", {
                    "default": 1,
                    "min": -999999,
                    "max": 999999,
                    "step": 1,
                }),
            },
            "optional": {
                "file_type": ([
                    "exr", "tiff", "png", "jpg", "dpx",
                    "hdr", "tga", "bmp"
                ], {"default": "exr"}),
                "bit_depth": (bit_depths, {"default": "16f"}),
                "compression": (exr_compressions, {"default": "dwaa"}),
                "frame_padding": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of digits for frame numbers (e.g., 4 = 0001, 0002...)"
                }),
                "auto_sequence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-increment filename (image1.png, image2.png...). If disabled, overwrites existing file."
                }),
                "create_directories": ("BOOLEAN", {"default": True}),
                "colorspace": (["raw", "sRGB", "linear", "ACEScg"], {"default": "raw"}),
                "show_preview": ("BOOLEAN", {"default": True, "tooltip": "Show thumbnail preview in node"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file_paths")
    FUNCTION = "write_image"
    CATEGORY = "Nuke/IO"
    OUTPUT_NODE = True

    def write_image(self, image, file_path, frame_start=1,
                    file_type="exr", bit_depth="16f", compression="dwaa",
                    frame_padding=4, auto_sequence=True, create_directories=True, colorspace="raw",
                    show_preview=True):
        """Write image(s) to disk."""

        if not file_path:
            print("[NukeWrite] No file path specified")
            return {"ui": {"images": []}, "result": (image, "")}

        # Ensure batch dimension
        img = ensure_batch_dim(image)
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

        # Parse frame pattern
        pattern, frame_spec, padding = parse_frame_pattern(file_path)
        is_sequence = frame_spec is not None and padding > 0

        # Use custom padding if pattern detected, otherwise use frame_padding parameter
        if is_sequence:
            # Pattern already exists, use its padding unless we need to update it
            pass
        else:
            # No pattern in path, use frame_padding parameter
            padding = frame_padding

        # Ensure correct extension
        base, ext = os.path.splitext(file_path)
        if ext.lower().lstrip('.') != file_type:
            file_path = f"{base}.{file_type}"
            pattern, frame_spec, padding_from_pattern = parse_frame_pattern(file_path)
            # If pattern was detected after adding extension, use it; otherwise keep our padding
            if frame_spec is not None and padding_from_pattern > 0:
                is_sequence = True
                padding = padding_from_pattern
            else:
                pattern = file_path

        # Create output directory if needed
        if create_directories:
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        written_paths = []

        for i in range(batch_size):
            # Get frame number
            frame_num = frame_start + i

            # Determine output path - always add frame number like Nuke does
            if is_sequence:
                # Explicit frame pattern in path (e.g., %04d or ####)
                output_path = expand_frame_pattern(pattern, frame_num, padding)
            else:
                # No explicit pattern - add frame number with underscore separator
                # e.g., test/image -> test/image_0001.exr (based on frame_start and frame_padding)
                base, ext = os.path.splitext(file_path)
                frame_str = str(frame_num).zfill(padding)
                output_path = f"{base}_{frame_str}{ext}"

            # Apply auto_sequence to avoid overwrites if enabled
            if auto_sequence:
                output_path = get_unique_filepath(output_path)

            # Get pixel data
            pixels = img[i].cpu().numpy()

            # Apply colorspace conversion (basic)
            if colorspace == "sRGB":
                # Linear to sRGB
                pixels = np.where(pixels <= 0.0031308,
                                 pixels * 12.92,
                                 1.055 * np.power(np.clip(pixels, 0.0031308, None), 1/2.4) - 0.055)
            elif colorspace == "linear":
                # sRGB to linear
                pixels = np.where(pixels <= 0.04045,
                                 pixels / 12.92,
                                 np.power((pixels + 0.055) / 1.055, 2.4))

            # Prepare metadata
            metadata = {
                "Software": "ComfyUI Nuke Nodes",
                "oiio:ColorSpace": colorspace if colorspace != "raw" else "linear",
            }

            # Write the image
            success = write_image(output_path, pixels, bit_depth, compression, metadata)

            if success:
                written_paths.append(output_path)
                print(f"[NukeWrite] Written: {output_path}")
            else:
                print(f"[NukeWrite] Failed to write: {output_path}")

        # Return paths as string
        paths_str = "\n".join(written_paths)

        # Create preview if enabled
        ui_images = []
        if show_preview and image.shape[0] > 0:
            ui_images = create_preview_images(image)

        return {"ui": {"images": ui_images}, "result": (image, paths_str)}


class NukeReadInfo(NukeNodeBase):
    """
    Read Info node - displays information about an image file or sequence.

    Shows: resolution, channels, bit depth, frame range, file size, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
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
                except Exception as e:
                    info += f"Error reading metadata: {e}\n"
            elif CV2_AVAILABLE:
                try:
                    img = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        info += f"Resolution: {img.shape[1]} x {img.shape[0]}\n"
                        info += f"Channels: {img.shape[2] if len(img.shape) > 2 else 1}\n"
                        info += f"Bit Depth: {img.dtype}\n"
                except Exception as e:
                    info += f"Error reading metadata: {e}\n"
        else:
            info += f"\nFile not found: {sample_path}\n"

        # Show available libraries
        info += f"\n=== I/O LIBRARIES ===\n"
        info += f"OpenImageIO: {'Available' if OIIO_AVAILABLE else 'Not installed'}\n"
        info += f"OpenCV: {'Available' if CV2_AVAILABLE else 'Not installed'}\n"
        info += f"PIL: {'Available' if PIL_AVAILABLE else 'Not installed'}\n"

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
