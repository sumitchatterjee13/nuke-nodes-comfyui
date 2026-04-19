"""
Multi-pass EXR loading and channel shuffle nodes (Nuke-style).

Provides:
  - NukeReadMultiPass: loads ALL channels from an EXR and groups them by
    layer/pass name (e.g. diffuse, N, Z, crypto00) into a dict-like custom
    type. Also outputs the beauty layer as a standard IMAGE for preview and
    a human-readable pass list.
  - NukeShufflePass: picks a specific pass by name from the multi-pass
    bundle and returns it as a standard ComfyUI IMAGE.
"""

import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

from .io_nodes import (
    OIIO_AVAILABLE,
    expand_frame_pattern,
    parse_frame_pattern,
)
from .utils import NukeNodeBase

try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None


# ---------------------------------------------------------------------------
# EXR channel grouping helpers
# ---------------------------------------------------------------------------

# Canonical ordering for channel suffixes so RGB(A) and XYZ(W) come out sorted
_SUFFIX_ORDER = {
    "R": 0, "G": 1, "B": 2, "A": 3,
    "X": 0, "Y": 1, "Z": 2, "W": 3,
    "r": 0, "g": 1, "b": 2, "a": 3,
    "x": 0, "y": 1, "z": 2, "w": 3,
}

# Top-level channel names that belong to the main beauty layer
_BEAUTY_CHANNELS = {"R", "G", "B", "A"}


def group_channels(channel_names: List[str]) -> "OrderedDict[str, List[Tuple[int, str]]]":
    """
    Group EXR channel names by layer/pass.

    Examples:
      ['R', 'G', 'B', 'A']                -> {'RGBA': [(0,'R'),(1,'G'),(2,'B'),(3,'A')]}
      ['diffuse.R', 'diffuse.G', ...]     -> {'diffuse': [(i,'R'), ...]}
      ['N.X', 'N.Y', 'N.Z']               -> {'N': [...]}
      ['Z']                               -> {'Z': [(i, '')]}

    Returns:
        OrderedDict[pass_name -> list of (channel_index, suffix)],
        with channels sorted by canonical suffix order.
    """
    groups: "OrderedDict[str, List[Tuple[int, str]]]" = OrderedDict()

    for i, name in enumerate(channel_names):
        if "." in name:
            layer, suffix = name.rsplit(".", 1)
        else:
            if name in _BEAUTY_CHANNELS:
                layer, suffix = "RGBA", name
            else:
                # Standalone single-channel pass (e.g. "Z", "depth")
                layer, suffix = name, ""

        groups.setdefault(layer, []).append((i, suffix))

    # Sort each pass's channels by canonical suffix order
    for layer, chans in groups.items():
        chans.sort(key=lambda c: _SUFFIX_ORDER.get(c[1], 99))

    return groups


def read_all_passes(filepath: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Read every channel of an EXR (or other multi-channel image) and group by
    layer.

    Returns:
        (passes_dict, raw_channel_names)
        passes_dict: {pass_name: np.ndarray[H, W, n_channels_in_pass]}
        raw_channel_names: original OIIO channel names (for debug)
    """
    if not OIIO_AVAILABLE or oiio is None:
        raise RuntimeError(
            "OpenImageIO is required for multi-pass EXR loading. "
            "Install with: pip install OpenImageIO"
        )

    inp = oiio.ImageInput.open(filepath)
    if inp is None:
        raise RuntimeError(f"OIIO could not open: {filepath} ({oiio.geterror()})")

    spec = inp.spec()
    pixels = inp.read_image("float")
    inp.close()

    if pixels is None:
        raise RuntimeError(f"OIIO returned no pixel data for: {filepath}")

    all_pixels = np.array(pixels, dtype=np.float32).reshape(
        spec.height, spec.width, spec.nchannels
    )

    channel_names = list(spec.channelnames)
    groups = group_channels(channel_names)

    passes: Dict[str, np.ndarray] = {}
    for layer, chans in groups.items():
        indices = [c[0] for c in chans]
        passes[layer] = all_pixels[:, :, indices]

    return passes, channel_names


def _passes_to_torch(passes: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert numpy pass dict to torch tensor dict (H,W,C float32)."""
    return {name: torch.from_numpy(np.ascontiguousarray(arr)).float()
            for name, arr in passes.items()}


def _pass_to_image(arr: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    """
    Convert a pass tensor [H, W, C] to ComfyUI IMAGE tensor [1, H, W, 3 or 4].

    mode:
      - auto: 1ch -> grayscale RGB, 2ch -> pad B with 0, 3/4ch -> as-is, 5+ -> first 4
      - rgb: force 3 channels (truncate or pad)
      - rgba: force 4 channels (truncate or pad with 1.0 alpha)
      - single_to_rgb: treat as single channel grayscale regardless
    """
    if arr.dim() == 2:
        arr = arr.unsqueeze(-1)
    H, W, C = arr.shape

    if mode == "single_to_rgb":
        gray = arr[..., 0:1]
        out = gray.repeat(1, 1, 3)
    elif mode == "rgb":
        if C == 1:
            out = arr.repeat(1, 1, 3)
        elif C == 2:
            zero = torch.zeros(H, W, 1, dtype=arr.dtype)
            out = torch.cat([arr, zero], dim=-1)
        elif C >= 3:
            out = arr[..., :3]
    elif mode == "rgba":
        if C == 1:
            rgb = arr.repeat(1, 1, 3)
            alpha = torch.ones(H, W, 1, dtype=arr.dtype)
            out = torch.cat([rgb, alpha], dim=-1)
        elif C == 2:
            zero = torch.zeros(H, W, 1, dtype=arr.dtype)
            alpha = torch.ones(H, W, 1, dtype=arr.dtype)
            out = torch.cat([arr, zero, alpha], dim=-1)
        elif C == 3:
            alpha = torch.ones(H, W, 1, dtype=arr.dtype)
            out = torch.cat([arr, alpha], dim=-1)
        else:
            out = arr[..., :4]
    else:  # auto
        if C == 1:
            out = arr.repeat(1, 1, 3)
        elif C == 2:
            zero = torch.zeros(H, W, 1, dtype=arr.dtype)
            out = torch.cat([arr, zero], dim=-1)
        elif C in (3, 4):
            out = arr
        else:
            out = arr[..., :4]

    return out.unsqueeze(0)  # add batch dim


def _format_pass_list(passes: Dict[str, torch.Tensor],
                     raw_channels: List[str]) -> str:
    """Build a human-readable listing of passes."""
    lines = [f"Total channels: {len(raw_channels)}",
             f"Passes: {len(passes)}", ""]
    for name, tensor in passes.items():
        H, W, C = tensor.shape
        lines.append(f"  {name:<20} ({C}ch)  {W}x{H}")
    lines.append("")
    lines.append("Raw channel names:")
    lines.append("  " + ", ".join(raw_channels))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node: NukeReadMultiPass
# ---------------------------------------------------------------------------

class NukeReadMultiPass(NukeNodeBase):
    """
    Load all passes/layers from a multi-channel EXR.

    Groups channels by layer name (e.g. diffuse, N, Z, crypto00) and outputs
    them as a custom NUKE_PASSES bundle. Also outputs the beauty layer as a
    standard IMAGE for preview and a pass list STRING for inspection.
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
                        "placeholder": "Path to EXR (supports %04d / #### patterns)",
                    },
                ),
                "frame": (
                    "INT",
                    {"default": 1, "min": -999999, "max": 999999, "step": 1},
                ),
            },
            "optional": {
                "load_as_sequence": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Expand %04d / #### patterns using the frame input"},
                ),
                "print_pass_list": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Print the pass list to the console when executed"},
                ),
            },
        }

    RETURN_TYPES = ("NUKE_PASSES", "IMAGE", "STRING")
    RETURN_NAMES = ("passes", "beauty", "pass_list")
    FUNCTION = "read_multipass"
    CATEGORY = "Nuke/IO"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def read_multipass(self, file_path, frame,
                       load_as_sequence=True, print_pass_list=True):
        if not file_path:
            print("[NukeReadMultiPass] No file path specified")
            empty = torch.zeros((1, 512, 512, 3))
            return ({}, empty, "No file loaded")

        file_path = os.path.expandvars(os.path.expanduser(file_path))

        pattern, frame_spec, padding = parse_frame_pattern(file_path)
        is_sequence = load_as_sequence and frame_spec is not None and padding > 0

        if is_sequence:
            actual_path = expand_frame_pattern(pattern, frame, padding)
        else:
            actual_path = file_path

        if not os.path.exists(actual_path):
            print(f"[NukeReadMultiPass] File not found: {actual_path}")
            empty = torch.zeros((1, 512, 512, 3))
            return ({}, empty, f"File not found: {actual_path}")

        print(f"[NukeReadMultiPass] Loading: {actual_path}")

        try:
            passes_np, channel_names = read_all_passes(actual_path)
        except Exception as e:
            print(f"[NukeReadMultiPass] Error: {e}")
            empty = torch.zeros((1, 512, 512, 3))
            return ({}, empty, f"Load error: {e}")

        passes = _passes_to_torch(passes_np)

        # Build human-readable pass list
        pass_list_str = _format_pass_list(passes, channel_names)
        if print_pass_list:
            print(f"[NukeReadMultiPass] ===== Passes in {os.path.basename(actual_path)} =====")
            print(pass_list_str)
            print(f"[NukeReadMultiPass] =============================================")

        # Pick a default "beauty" preview: prefer RGBA, else first pass
        if "RGBA" in passes:
            beauty = _pass_to_image(passes["RGBA"], mode="auto")
        elif passes:
            beauty = _pass_to_image(next(iter(passes.values())), mode="auto")
        else:
            beauty = torch.zeros((1, 512, 512, 3))

        return (passes, beauty, pass_list_str)


# ---------------------------------------------------------------------------
# Node: NukeShufflePass
# ---------------------------------------------------------------------------

class NukeShufflePass(NukeNodeBase):
    """
    Pick a specific pass by name from a NUKE_PASSES bundle and output it as
    a standard ComfyUI IMAGE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passes": ("NUKE_PASSES",),
                "pass_name": (
                    "STRING",
                    {
                        "default": "RGBA",
                        "multiline": False,
                        "placeholder": "e.g. RGBA, diffuse, N, Z, crypto00",
                    },
                ),
            },
            "optional": {
                "channel_mode": (
                    ["auto", "rgb", "rgba", "single_to_rgb"],
                    {"default": "auto",
                     "tooltip": (
                         "How to convert the pass to a 3/4-channel IMAGE:\n"
                         "  auto: 1ch->gray, 2ch->pad, 3/4ch->as-is, 5+->first 4\n"
                         "  rgb: force 3 channels\n"
                         "  rgba: force 4 channels (add alpha=1)\n"
                         "  single_to_rgb: always treat as grayscale"
                     )},
                ),
                "on_missing": (
                    ["black", "error"],
                    {"default": "black",
                     "tooltip": "If the named pass is not found: return black or raise"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "shuffle"
    CATEGORY = "Nuke/IO"

    def shuffle(self, passes, pass_name,
                channel_mode="auto", on_missing="black"):
        if not passes:
            msg = "Empty passes bundle"
            print(f"[NukeShufflePass] {msg}")
            return (torch.zeros((1, 512, 512, 3)), msg)

        name = pass_name.strip()

        if name not in passes:
            available = ", ".join(passes.keys())
            msg = f"Pass '{name}' not found. Available: {available}"
            print(f"[NukeShufflePass] {msg}")
            if on_missing == "error":
                raise ValueError(msg)
            # Fallback to black using dimensions of first pass
            first = next(iter(passes.values()))
            H, W = first.shape[:2]
            return (torch.zeros((1, H, W, 3)), msg)

        arr = passes[name]  # [H, W, C]
        img = _pass_to_image(arr, mode=channel_mode)

        C = arr.shape[-1]
        H, W = arr.shape[:2]
        info = (f"Pass '{name}': {C}ch, {W}x{H}, "
                f"range [{arr.min():.3f}, {arr.max():.3f}]")
        print(f"[NukeShufflePass] {info}")

        return (img, info)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NukeReadMultiPass": NukeReadMultiPass,
    "NukeShufflePass": NukeShufflePass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeReadMultiPass": "Nuke Read MultiPass",
    "NukeShufflePass": "Nuke Shuffle Pass",
}
