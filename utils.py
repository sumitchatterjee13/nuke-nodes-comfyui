"""
Utility functions and base classes for Nuke nodes
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch


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
