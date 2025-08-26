"""
Merge and compositing nodes that replicate Nuke's merge functionality
"""

import torch
import torch.nn.functional as F
import numpy as np
from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor

class NukeMerge(NukeNodeBase):
    """
    Advanced merge node with multiple blend modes, similar to Nuke's Merge node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "operation": (["over", "add", "multiply", "screen", "overlay", "soft_light", 
                             "hard_light", "color_dodge", "color_burn", "darken", "lighten", 
                             "difference", "exclusion", "subtract", "divide"], {"default": "over"}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Nuke/Merge"
    
    def merge(self, image_a, image_b, operation, mix, mask=None):
        """
        Merge two images using specified blend mode
        """
        # Ensure both images have batch dimension
        a = ensure_batch_dim(image_a)
        b = ensure_batch_dim(image_b)
        
        # Get dimensions
        batch_size = max(a.shape[0], b.shape[0])
        
        # Resize images to match if needed
        if a.shape[1:3] != b.shape[1:3]:
            target_h, target_w = max(a.shape[1], b.shape[1]), max(a.shape[2], b.shape[2])
            a = F.interpolate(a.permute(0, 3, 1, 2), size=(target_h, target_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            b = F.interpolate(b.permute(0, 3, 1, 2), size=(target_h, target_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        # Extract alpha channels if they exist
        if a.shape[3] >= 4:
            a_alpha = a[:, :, :, 3:4]
            a_rgb = a[:, :, :, :3]
        else:
            a_alpha = torch.ones_like(a[:, :, :, :1])
            a_rgb = a
        
        if b.shape[3] >= 4:
            b_alpha = b[:, :, :, 3:4]
            b_rgb = b[:, :, :, :3]
        else:
            b_alpha = torch.ones_like(b[:, :, :, :1])
            b_rgb = b
        
        # Apply blend mode
        result_rgb = self._apply_blend_mode(a_rgb, b_rgb, operation)
        
        # Composite with alpha
        if operation == "over":
            # Standard over operation: A over B
            result_alpha = a_alpha + b_alpha * (1 - a_alpha)
            result_rgb = (a_rgb * a_alpha + b_rgb * b_alpha * (1 - a_alpha)) / torch.clamp(result_alpha, min=1e-7)
        else:
            # For other operations, use simple alpha blending
            result_alpha = torch.max(a_alpha, b_alpha)
        
        # Apply mask if provided
        if mask is not None:
            mask = ensure_batch_dim(mask)
            if mask.shape[1:3] != result_rgb.shape[1:3]:
                mask = F.interpolate(mask.permute(0, 3, 1, 2), size=result_rgb.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            
            # Use first channel of mask as alpha
            mask_alpha = mask[:, :, :, :1]
            result_rgb = a_rgb + (result_rgb - a_rgb) * mask_alpha * mix
            result_alpha = a_alpha + (result_alpha - a_alpha) * mask_alpha * mix
        else:
            # Apply mix factor
            result_rgb = a_rgb + (result_rgb - a_rgb) * mix
            result_alpha = a_alpha + (result_alpha - a_alpha) * mix
        
        # Combine RGB and alpha
        if a.shape[3] >= 4 or b.shape[3] >= 4:
            result = torch.cat([result_rgb, result_alpha], dim=3)
        else:
            result = result_rgb
        
        return (normalize_tensor(result),)
    
    def _apply_blend_mode(self, a, b, mode):
        """Apply specified blend mode to RGB channels"""
        if mode == "over":
            return b
        elif mode == "add":
            return a + b
        elif mode == "multiply":
            return a * b
        elif mode == "screen":
            return 1 - (1 - a) * (1 - b)
        elif mode == "overlay":
            return torch.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
        elif mode == "soft_light":
            return torch.where(b < 0.5, 
                             2 * a * b + a * a * (1 - 2 * b),
                             2 * a * (1 - b) + torch.sqrt(a) * (2 * b - 1))
        elif mode == "hard_light":
            return torch.where(b < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
        elif mode == "color_dodge":
            return torch.where(b >= 1, torch.ones_like(a), torch.clamp(a / (1 - b + 1e-7), 0, 1))
        elif mode == "color_burn":
            return torch.where(b <= 0, torch.zeros_like(a), 1 - torch.clamp((1 - a) / (b + 1e-7), 0, 1))
        elif mode == "darken":
            return torch.min(a, b)
        elif mode == "lighten":
            return torch.max(a, b)
        elif mode == "difference":
            return torch.abs(a - b)
        elif mode == "exclusion":
            return a + b - 2 * a * b
        elif mode == "subtract":
            return torch.clamp(a - b, 0, 1)
        elif mode == "divide":
            return torch.clamp(a / (b + 1e-7), 0, 1)
        else:
            return b

class NukeMix(NukeNodeBase):
    """
    Simple mix node for blending two images with a factor
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix"
    CATEGORY = "Nuke/Merge"
    
    def mix(self, image_a, image_b, mix):
        """
        Linear mix between two images
        """
        a = ensure_batch_dim(image_a)
        b = ensure_batch_dim(image_b)
        
        # Resize if needed
        if a.shape[1:3] != b.shape[1:3]:
            target_h, target_w = max(a.shape[1], b.shape[1]), max(a.shape[2], b.shape[2])
            a = F.interpolate(a.permute(0, 3, 1, 2), size=(target_h, target_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            b = F.interpolate(b.permute(0, 3, 1, 2), size=(target_h, target_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        result = a * (1 - mix) + b * mix
        return (normalize_tensor(result),)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeMerge": NukeMerge,
    "NukeMix": NukeMix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMerge": "Nuke Merge",
    "NukeMix": "Nuke Mix",
}
