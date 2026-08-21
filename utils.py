"""
Utility functions and base classes for Nuke nodes
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the host environment
    cv2 = None


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array for processing"""
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert NumPy array back to PyTorch tensor"""
    return torch.from_numpy(array).to(device)


def ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has batch dimension"""
    if len(tensor.shape) == 3:  # H, W, C
        tensor = tensor.unsqueeze(0)  # Add batch dim: B, H, W, C
    return tensor


def remove_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Remove batch dimension if batch size is 1"""
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor values to 0-1 range"""
    return torch.clamp(tensor, 0.0, 1.0)


def get_tensor_info(tensor: torch.Tensor) -> str:
    """Get debug info about tensor"""
    return f"Shape: {tensor.shape}, Device: {tensor.device}, Dtype: {tensor.dtype}, Range: [{tensor.min():.4f}, {tensor.max():.4f}]"


# ---------------------------------------------------------------------------
# Mask handling (ComfyUI MASK tensors are [H,W] or [B,H,W], values 0..1)
# ---------------------------------------------------------------------------

def mask_to_bhw1(mask: torch.Tensor, height: int, width: int, device=None) -> torch.Tensor:
    """Normalise a MASK (or a single-channel/RGB IMAGE used as a mask) to
    [B,H,W,1] at the requested resolution.

    Accepts [H,W], [B,H,W], [B,H,W,1] and [B,H,W,C] (first channel is used).
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(-1)
    if mask.shape[-1] > 1:
        mask = mask[..., :1]
    if device is not None:
        mask = mask.to(device)
    mask = mask.to(torch.float32)
    if mask.shape[1] != height or mask.shape[2] != width:
        mask = F.interpolate(
            mask.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
    return mask


def apply_mask_mix(
    original: torch.Tensor,
    processed: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mix: float = 1.0,
) -> torch.Tensor:
    """Blend ``processed`` over ``original`` (both [B,H,W,C]) by ``mix`` and
    an optional mask:  result = original + (processed - original) * mix * mask.

    A mask with batch 1 broadcasts over the image batch.
    """
    if mix == 1.0 and mask is None:
        return processed
    weight: Union[float, torch.Tensor] = float(mix)
    if mask is not None:
        m = mask_to_bhw1(mask, original.shape[1], original.shape[2], original.device)
        m = m.to(original.dtype)
        weight = m * weight
    return original + (processed - original) * weight


# ---------------------------------------------------------------------------
# Resampling (shared by Transform, CornerPin, Reformat)
# ---------------------------------------------------------------------------

FILTER_NAMES = ["impulse", "cubic", "lanczos", "area"]


def _filter_flag(filter_name: str) -> int:
    if cv2 is None:
        raise RuntimeError("OpenCV (opencv-python-headless) is required for resampling")
    table = {
        "impulse": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    return table.get(filter_name, cv2.INTER_CUBIC)


def remap_image(
    img_hwc: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    filter_name: str = "cubic",
    black_outside: bool = True,
) -> np.ndarray:
    """Resample one (H, W, C) float32 image through source-coordinate maps.

    ``map_x`` / ``map_y`` are float32 [H_out, W_out] arrays giving, for every
    output pixel, the SOURCE coordinate to sample, in pixel units with pixel
    centres at +0.5 (Nuke convention). An identity map therefore reproduces
    the input bit-exactly for any filter. Outside the source, pixels are black
    (``black_outside=True``) or replicate the nearest edge pixel.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (opencv-python-headless) is required for resampling")
    flag = _filter_flag(filter_name)
    if flag == cv2.INTER_AREA:
        # cv2.remap has no area filter; area is only meaningful for pure
        # downscales via cv2.resize (Reformat handles that case itself).
        flag = cv2.INTER_CUBIC
    border = cv2.BORDER_CONSTANT if black_outside else cv2.BORDER_REPLICATE
    src = np.ascontiguousarray(img_hwc, dtype=np.float32)
    mx = np.ascontiguousarray(map_x, dtype=np.float32) - 0.5
    my = np.ascontiguousarray(map_y, dtype=np.float32) - 0.5
    out = cv2.remap(src, mx, my, flag, borderMode=border, borderValue=0)
    if out.ndim == 2:
        out = out[:, :, np.newaxis]
    return out


def identity_maps(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Source-coordinate maps (centres at +0.5) that reproduce the input."""
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32) + 0.5,
        np.arange(width, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    return xs, ys


class NukeNodeBase:
    """Base class for all Nuke-style nodes"""

    CATEGORY = "Nuke"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        """Define input types - to be overridden by subclasses"""
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def process(self, **kwargs):
        """Main processing function - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement process method")
