"""
Thumbnail previews for the Nuke I/O nodes (OpenImageIO-backed).

Writes small PNGs into ComfyUI's temp directory and returns the ``ui``
preview dicts the front end expects.
"""

import logging
import os
import uuid

import folder_paths
import numpy as np
import torch

from .image_io import OIIO_AVAILABLE, oiio

logger = logging.getLogger(__name__)


def resize_image_oiio(img_np: np.ndarray, max_size: int = 256) -> np.ndarray:
    """
    Resize image using OpenImageIO's ImageBufAlgo.

    Args:
        img_np: Image as numpy array (H, W, C)
        max_size: Maximum dimension

    Returns:
        Resized image as numpy array
    """
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

    if not OIIO_AVAILABLE:
        # Previews are best-effort: nearest-neighbour subsample with numpy
        y_indices = np.linspace(0, height - 1, new_height).astype(int)
        x_indices = np.linspace(0, width - 1, new_width).astype(int)
        return img_np[np.ix_(y_indices, x_indices)]

    # Create ImageBuf from numpy array
    spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    src_buf = oiio.ImageBuf(spec)
    # Ensure array is contiguous in memory for OIIO
    pixels_contiguous = np.ascontiguousarray(img_np.astype(np.float32))
    src_buf.set_pixels(
        oiio.ROI(0, width, 0, height, 0, 1, 0, channels), pixels_contiguous
    )

    # Resize using OIIO
    dst_buf = oiio.ImageBufAlgo.resize(
        src_buf, roi=oiio.ROI(0, new_width, 0, new_height, 0, 1, 0, channels)
    )

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


def create_preview_images(
    images: torch.Tensor, max_size: int = 256, max_frames: int = 1000
) -> list:
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

    # One random id per call: id(tensor) can be reused by CPython once the
    # tensor is freed, which made two batches share (and cache-collide on)
    # the same thumbnail filenames.
    batch_id = uuid.uuid4().hex

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
        preview_filename = f"nuke_preview_{batch_id}_{i}.png"
        preview_path = os.path.join(temp_dir, preview_filename)

        if save_preview_oiio(img_np, preview_path):
            previews.append(
                {
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp",
                    "frame": i + 1,
                }
            )

    return previews


def save_preview_to_temp(img_np: np.ndarray, suffix: str = "") -> dict:
    """
    Save a single numpy image to temp directory for preview.
    Uses OpenImageIO for image processing and saving.

    Args:
        img_np: Image as numpy array (H, W, C) in 0-1 range
        suffix: Optional suffix for filename

    Returns:
        Preview dictionary for ComfyUI UI (or None if saving failed)
    """
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
        return {"filename": preview_filename, "subfolder": "", "type": "temp"}

    return None
