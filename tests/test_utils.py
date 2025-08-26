"""
Test utilities and fixtures for testing Nuke nodes
"""

import numpy as np
import torch


def create_test_image(width=64, height=64, channels=3, batch_size=1):
    """Create a test image tensor in ComfyUI format (B, H, W, C)"""
    # Create a gradient pattern for testing
    y = torch.linspace(0, 1, height).view(-1, 1).expand(height, width)
    x = torch.linspace(0, 1, width).view(1, -1).expand(height, width)

    if channels == 1:
        image = (x + y) / 2
        image = image.unsqueeze(-1)
    elif channels == 3:
        r = x
        g = y
        b = (x + y) / 2
        image = torch.stack([r, g, b], dim=-1)
    elif channels == 4:
        r = x
        g = y
        b = (x + y) / 2
        a = torch.ones_like(x)
        image = torch.stack([r, g, b, a], dim=-1)
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    # Add batch dimension
    image = image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return image.float()


def create_test_mask(width=64, height=64, batch_size=1):
    """Create a test mask (circular)"""
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4

    y_coords = torch.arange(height).float().view(-1, 1).expand(height, width)
    x_coords = torch.arange(width).float().view(1, -1).expand(height, width)

    distance = torch.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    mask = (distance <= radius).float().unsqueeze(-1)

    # Add batch dimension
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return mask


def assert_image_valid(image, expected_shape=None):
    """Assert that an image tensor is valid"""
    assert isinstance(image, torch.Tensor), "Image must be a torch.Tensor"
    assert (
        len(image.shape) == 4
    ), f"Image must have 4 dimensions, got {len(image.shape)}"
    assert image.dtype == torch.float32, f"Image must be float32, got {image.dtype}"
    assert torch.all(image >= 0), "Image values must be non-negative"
    assert torch.all(image <= 1), "Image values must be <= 1"

    if expected_shape is not None:
        assert (
            image.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {image.shape}"


def assert_images_similar(img1, img2, tolerance=1e-6):
    """Assert that two images are similar within tolerance"""
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    diff = torch.abs(img1 - img2)
    max_diff = torch.max(diff)
    assert (
        max_diff <= tolerance
    ), f"Images differ by {max_diff}, tolerance is {tolerance}"
