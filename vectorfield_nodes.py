"""
VectorField nodes for loading and applying LUTs (Look-Up Tables).

Similar to Nuke's Vectorfield node, this module provides LUT loading and application
capabilities for color grading and technical color transformations.

Supported LUT formats:
- .cube (Resolve, DaVinci, Adobe)
- .3dl (Autodesk, Lustre)
- .csp (Rising Sun Research Cinespace)
- .spi1d / .spi3d (Sony Pictures Imageworks)

LUT files should be placed in the ./luts folder relative to this package.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import torch

from .utils import NukeNodeBase, ensure_batch_dim, normalize_tensor


# Get the directory where this module is located
MODULE_DIR = Path(__file__).parent
LUTS_DIR = MODULE_DIR / "luts"

# Supported LUT file extensions
LUT_EXTENSIONS = {".cube", ".3dl", ".csp", ".spi1d", ".spi3d", ".lut"}


def get_available_luts() -> List[str]:
    """
    Scan the luts directory and return a list of available LUT files.

    Returns:
        List of LUT filenames (without path)
    """
    if not LUTS_DIR.exists():
        LUTS_DIR.mkdir(parents=True, exist_ok=True)
        return ["No LUTs found"]

    lut_files = []
    for file in LUTS_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in LUT_EXTENSIONS:
            lut_files.append(file.name)

    if not lut_files:
        return ["No LUTs found"]

    return sorted(lut_files)


def parse_cube_lut(filepath: Path) -> Dict:
    """
    Parse a .cube LUT file.

    Returns:
        Dictionary with 'size', 'data', 'domain_min', 'domain_max', 'type' (1D or 3D)
    """
    lut_data = {
        "title": "",
        "domain_min": [0.0, 0.0, 0.0],
        "domain_max": [1.0, 1.0, 1.0],
        "size": 0,
        "data": [],
        "type": "3D"
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    data_started = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if parts[0].upper() == "TITLE":
            lut_data["title"] = " ".join(parts[1:]).strip('"')
        elif parts[0].upper() == "DOMAIN_MIN":
            lut_data["domain_min"] = [float(x) for x in parts[1:4]]
        elif parts[0].upper() == "DOMAIN_MAX":
            lut_data["domain_max"] = [float(x) for x in parts[1:4]]
        elif parts[0].upper() == "LUT_1D_SIZE":
            lut_data["size"] = int(parts[1])
            lut_data["type"] = "1D"
        elif parts[0].upper() == "LUT_3D_SIZE":
            lut_data["size"] = int(parts[1])
            lut_data["type"] = "3D"
        elif len(parts) >= 3:
            # Data line - 3 floats
            try:
                rgb = [float(x) for x in parts[:3]]
                lut_data["data"].append(rgb)
                data_started = True
            except ValueError:
                if data_started:
                    # If we've started reading data and hit a non-float, stop
                    break

    lut_data["data"] = np.array(lut_data["data"], dtype=np.float32)
    return lut_data


def parse_3dl_lut(filepath: Path) -> Dict:
    """
    Parse a .3dl LUT file (Autodesk/Lustre format).

    Returns:
        Dictionary with LUT data
    """
    lut_data = {
        "title": filepath.stem,
        "domain_min": [0.0, 0.0, 0.0],
        "domain_max": [1.0, 1.0, 1.0],
        "size": 0,
        "data": [],
        "type": "3D",
        "input_range": None,
        "output_bit_depth": 12  # Common default
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    data_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        parts = line.split()

        # First line with numbers might be the shaper/input range
        if lut_data["input_range"] is None and len(parts) >= 2:
            try:
                # Check if this is the input bit depth line
                values = [int(x) for x in parts]
                if len(values) >= 2 and values[0] < values[-1]:
                    lut_data["input_range"] = values
                    continue
            except ValueError:
                pass

        # Data lines - 3 integers
        if len(parts) >= 3:
            try:
                rgb = [int(x) for x in parts[:3]]
                data_lines.append(rgb)
            except ValueError:
                continue

    # Determine LUT size from data count
    data_count = len(data_lines)
    for size in [17, 33, 65, 32, 64, 16]:
        if size ** 3 == data_count:
            lut_data["size"] = size
            break

    if lut_data["size"] == 0 and data_count > 0:
        # Estimate size
        lut_data["size"] = int(round(data_count ** (1/3)))

    # Normalize integer values to 0-1 range
    if data_lines:
        data_array = np.array(data_lines, dtype=np.float32)
        max_val = data_array.max()
        if max_val > 1.0:
            # Determine bit depth from max value
            if max_val <= 255:
                data_array /= 255.0
            elif max_val <= 1023:
                data_array /= 1023.0
            elif max_val <= 4095:
                data_array /= 4095.0
            elif max_val <= 65535:
                data_array /= 65535.0
        lut_data["data"] = data_array

    return lut_data


def parse_spi3d_lut(filepath: Path) -> Dict:
    """
    Parse a .spi3d LUT file (Sony Pictures Imageworks format).

    Returns:
        Dictionary with LUT data
    """
    lut_data = {
        "title": filepath.stem,
        "domain_min": [0.0, 0.0, 0.0],
        "domain_max": [1.0, 1.0, 1.0],
        "size": 0,
        "data": [],
        "type": "3D"
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    data_lines = []
    header_done = False

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # First line is usually "SPILUT 1.0"
        if parts[0].upper() == "SPILUT":
            continue

        # Second line should be "3 3" for 3D LUT with 3 input/output channels
        if not header_done and len(parts) == 2:
            try:
                int(parts[0])
                int(parts[1])
                header_done = True
                continue
            except ValueError:
                pass

        # Size line - single integer
        if header_done and lut_data["size"] == 0 and len(parts) == 1:
            try:
                lut_data["size"] = int(parts[0])
                continue
            except ValueError:
                pass

        # Data lines - index r g b output_r output_g output_b
        if len(parts) >= 6:
            try:
                # Skip input indices, take output RGB
                rgb = [float(parts[3]), float(parts[4]), float(parts[5])]
                data_lines.append(rgb)
            except ValueError:
                continue

    if data_lines:
        lut_data["data"] = np.array(data_lines, dtype=np.float32)

    return lut_data


def parse_spi1d_lut(filepath: Path) -> Dict:
    """
    Parse a .spi1d LUT file (Sony Pictures Imageworks 1D format).

    Returns:
        Dictionary with LUT data
    """
    lut_data = {
        "title": filepath.stem,
        "domain_min": [0.0, 0.0, 0.0],
        "domain_max": [1.0, 1.0, 1.0],
        "size": 0,
        "data": [],
        "type": "1D"
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    data_lines = []

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        # Look for "Version", "From", "To", "Length", "Components"
        if parts[0].lower() == "from":
            lut_data["domain_min"] = [float(parts[1])] * 3
        elif parts[0].lower() == "to":
            lut_data["domain_max"] = [float(parts[1])] * 3
        elif parts[0].lower() == "length":
            lut_data["size"] = int(parts[1])
        elif parts[0].lower() in ("version", "components"):
            continue
        elif len(parts) >= 1:
            # Data line
            try:
                if len(parts) == 1:
                    val = float(parts[0])
                    data_lines.append([val, val, val])
                elif len(parts) >= 3:
                    rgb = [float(x) for x in parts[:3]]
                    data_lines.append(rgb)
            except ValueError:
                continue

    if data_lines:
        lut_data["data"] = np.array(data_lines, dtype=np.float32)
        if lut_data["size"] == 0:
            lut_data["size"] = len(data_lines)

    return lut_data


def load_lut(filepath: Union[str, Path]) -> Optional[Dict]:
    """
    Load a LUT file and return parsed data.

    Args:
        filepath: Path to the LUT file

    Returns:
        Dictionary with LUT data or None if loading failed
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"[NukeVectorField] LUT file not found: {filepath}")
        return None

    ext = filepath.suffix.lower()

    try:
        if ext == ".cube":
            return parse_cube_lut(filepath)
        elif ext == ".3dl":
            return parse_3dl_lut(filepath)
        elif ext == ".spi3d":
            return parse_spi3d_lut(filepath)
        elif ext == ".spi1d":
            return parse_spi1d_lut(filepath)
        else:
            print(f"[NukeVectorField] Unsupported LUT format: {ext}")
            return None
    except Exception as e:
        print(f"[NukeVectorField] Error loading LUT: {e}")
        return None


def apply_1d_lut(image: np.ndarray, lut_data: Dict) -> np.ndarray:
    """
    Apply a 1D LUT to an image.

    Args:
        image: Input image as numpy array (H, W, C) in 0-1 range
        lut_data: Parsed LUT dictionary

    Returns:
        Transformed image
    """
    lut = lut_data["data"]
    size = lut_data["size"]
    domain_min = np.array(lut_data["domain_min"], dtype=np.float32)
    domain_max = np.array(lut_data["domain_max"], dtype=np.float32)

    # Normalize input to LUT domain
    normalized = (image - domain_min) / (domain_max - domain_min + 1e-10)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Scale to LUT indices
    indices = normalized * (size - 1)

    # Interpolate
    idx_low = np.floor(indices).astype(np.int32)
    idx_high = np.ceil(indices).astype(np.int32)
    idx_low = np.clip(idx_low, 0, size - 1)
    idx_high = np.clip(idx_high, 0, size - 1)

    frac = indices - idx_low

    result = np.zeros_like(image)
    for c in range(min(3, image.shape[-1])):
        lut_channel = lut[:, c] if lut.ndim > 1 else lut.flatten()
        low_vals = lut_channel[idx_low[..., c]]
        high_vals = lut_channel[idx_high[..., c]]
        result[..., c] = low_vals + frac[..., c] * (high_vals - low_vals)

    # Preserve alpha if present
    if image.shape[-1] == 4:
        result[..., 3] = image[..., 3]

    return result


def apply_3d_lut(image: np.ndarray, lut_data: Dict) -> np.ndarray:
    """
    Apply a 3D LUT to an image using trilinear interpolation.

    Args:
        image: Input image as numpy array (H, W, C) in 0-1 range
        lut_data: Parsed LUT dictionary

    Returns:
        Transformed image

    Note on .cube format:
        In .cube files, data is stored in row-major order where red varies fastest,
        then green, then blue. The LUT should be indexed as [b][g][r] to get the
        correct output for input RGB values.
    """
    lut = lut_data["data"]
    size = lut_data["size"]
    domain_min = np.array(lut_data["domain_min"], dtype=np.float32)
    domain_max = np.array(lut_data["domain_max"], dtype=np.float32)

    # Reshape LUT to 3D grid
    # .cube format: red varies fastest, so shape is (B, G, R, 3) for indexing
    try:
        lut_3d = lut.reshape((size, size, size, 3))
    except ValueError:
        print(f"[NukeVectorField] LUT data size mismatch. Expected {size**3} entries, got {len(lut)}")
        return image

    # Normalize input to LUT domain
    normalized = (image[..., :3] - domain_min) / (domain_max - domain_min + 1e-10)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Scale to LUT indices
    indices = normalized * (size - 1)

    # Get integer indices for trilinear interpolation
    idx_low = np.floor(indices).astype(np.int32)
    idx_high = np.clip(idx_low + 1, 0, size - 1)
    idx_low = np.clip(idx_low, 0, size - 1)

    # Fractional parts
    frac = indices - idx_low

    # Extract coordinates (R, G, B from input image)
    r_low, g_low, b_low = idx_low[..., 0], idx_low[..., 1], idx_low[..., 2]
    r_high, g_high, b_high = idx_high[..., 0], idx_high[..., 1], idx_high[..., 2]
    r_frac, g_frac, b_frac = frac[..., 0], frac[..., 1], frac[..., 2]

    # Trilinear interpolation
    # .cube format indexes as [B][G][R] since red varies fastest
    # 8 corners of the cube
    c000 = lut_3d[b_low, g_low, r_low]
    c001 = lut_3d[b_low, g_low, r_high]
    c010 = lut_3d[b_low, g_high, r_low]
    c011 = lut_3d[b_low, g_high, r_high]
    c100 = lut_3d[b_high, g_low, r_low]
    c101 = lut_3d[b_high, g_low, r_high]
    c110 = lut_3d[b_high, g_high, r_low]
    c111 = lut_3d[b_high, g_high, r_high]

    # Interpolate along R axis (fastest varying in .cube)
    r_frac = r_frac[..., np.newaxis]
    c00 = c000 * (1 - r_frac) + c001 * r_frac
    c01 = c010 * (1 - r_frac) + c011 * r_frac
    c10 = c100 * (1 - r_frac) + c101 * r_frac
    c11 = c110 * (1 - r_frac) + c111 * r_frac

    # Interpolate along G axis
    g_frac = g_frac[..., np.newaxis]
    c0 = c00 * (1 - g_frac) + c01 * g_frac
    c1 = c10 * (1 - g_frac) + c11 * g_frac

    # Interpolate along B axis (slowest varying in .cube)
    b_frac = b_frac[..., np.newaxis]
    result_rgb = c0 * (1 - b_frac) + c1 * b_frac

    # Combine with alpha if present
    if image.shape[-1] == 4:
        result = np.concatenate([result_rgb, image[..., 3:4]], axis=-1)
    else:
        result = result_rgb

    return result.astype(np.float32)


class NukeVectorfield(NukeNodeBase):
    """
    VectorField LUT node - loads and applies Look-Up Tables to images.

    Similar to Nuke's Vectorfield node, this applies color transformations
    defined by LUT files for color grading, technical transforms, and looks.

    Supported formats:
    - .cube (Resolve, Adobe, general purpose)
    - .3dl (Autodesk Flame, Lustre)
    - .spi1d / .spi3d (Sony Pictures Imageworks)

    Place LUT files in the 'luts' folder within the nuke-nodes package.

    Note: .cube files use red-fastest ordering, indexed as [B][G][R].
    """

    # Cache for loaded LUTs - cleared on reload to pick up any fixes
    _lut_cache: Dict[str, Dict] = {}

    @classmethod
    def INPUT_TYPES(cls):
        lut_files = get_available_luts()

        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": (lut_files, {"default": lut_files[0] if lut_files else "No LUTs found"}),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "custom_lut_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional: path to custom LUT file",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lut"
    CATEGORY = "Nuke/Color"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Check if LUT files have changed
        return float("nan")

    def apply_lut(self, image, lut_file, intensity, custom_lut_path=""):
        """
        Apply LUT transformation to the input image.

        Args:
            image: Input image tensor
            lut_file: Selected LUT file from dropdown
            intensity: Mix factor (0 = original, 1 = full LUT effect)
            custom_lut_path: Optional path to a custom LUT file

        Returns:
            Transformed image
        """
        # Determine which LUT file to use
        if custom_lut_path and os.path.exists(custom_lut_path):
            lut_path = Path(custom_lut_path)
        elif lut_file and lut_file != "No LUTs found":
            lut_path = LUTS_DIR / lut_file
        else:
            print("[NukeVectorField] No LUT file specified")
            return (image,)

        # Load LUT (with caching)
        cache_key = str(lut_path)
        if cache_key not in self._lut_cache:
            lut_data = load_lut(lut_path)
            if lut_data is None:
                return (image,)
            self._lut_cache[cache_key] = lut_data
        else:
            lut_data = self._lut_cache[cache_key]

        # Ensure batch dimension
        img = ensure_batch_dim(image)
        batch_size = img.shape[0]

        results = []
        for i in range(batch_size):
            img_np = img[i].cpu().numpy().astype(np.float32)

            # Apply LUT based on type
            if lut_data["type"] == "1D":
                transformed = apply_1d_lut(img_np, lut_data)
            else:  # 3D
                transformed = apply_3d_lut(img_np, lut_data)

            # Apply intensity (mix with original)
            if intensity < 1.0:
                transformed = img_np + intensity * (transformed - img_np)
            elif intensity > 1.0:
                # Extrapolate beyond 1.0
                diff = transformed - img_np
                transformed = img_np + intensity * diff

            results.append(transformed)

        result_np = np.stack(results, axis=0)
        result = torch.from_numpy(result_np).to(image.device)

        return (result,)


class NukeVectorfieldInfo(NukeNodeBase):
    """
    VectorField Info node - displays information about a LUT file.

    Useful for debugging and understanding LUT properties.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lut_files = get_available_luts()

        return {
            "required": {
                "lut_file": (lut_files, {"default": lut_files[0] if lut_files else "No LUTs found"}),
            },
            "optional": {
                "custom_lut_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "Nuke/Color"
    OUTPUT_NODE = True

    def get_info(self, lut_file, custom_lut_path=""):
        """Get LUT file information."""

        # Determine which LUT file to use
        if custom_lut_path and os.path.exists(custom_lut_path):
            lut_path = Path(custom_lut_path)
        elif lut_file and lut_file != "No LUTs found":
            lut_path = LUTS_DIR / lut_file
        else:
            # List available LUTs
            info = "Available LUTs:\n\n"
            info += f"LUT Directory: {LUTS_DIR}\n\n"

            luts = get_available_luts()
            if luts[0] == "No LUTs found":
                info += "No LUT files found.\n\n"
                info += "Supported formats:\n"
                info += "  - .cube (Resolve, Adobe)\n"
                info += "  - .3dl (Autodesk Flame, Lustre)\n"
                info += "  - .spi1d / .spi3d (Sony Pictures)\n"
            else:
                info += f"Found {len(luts)} LUT file(s):\n"
                for lut in luts:
                    info += f"  - {lut}\n"

            return (info,)

        # Load and display LUT info
        lut_data = load_lut(lut_path)

        if lut_data is None:
            return (f"Error loading LUT: {lut_path}",)

        info = f"LUT File: {lut_path.name}\n"
        info += f"Full Path: {lut_path}\n\n"

        if lut_data.get("title"):
            info += f"Title: {lut_data['title']}\n"

        info += f"Type: {lut_data['type']} LUT\n"
        info += f"Size: {lut_data['size']}"

        if lut_data['type'] == '3D':
            total_entries = lut_data['size'] ** 3
            info += f" ({lut_data['size']}x{lut_data['size']}x{lut_data['size']} = {total_entries} entries)"

        info += "\n"
        info += f"Domain Min: {lut_data['domain_min']}\n"
        info += f"Domain Max: {lut_data['domain_max']}\n"

        if len(lut_data['data']) > 0:
            data = lut_data['data']
            info += f"\nData Range:\n"
            info += f"  Min: [{data.min(axis=0)[0]:.4f}, {data.min(axis=0)[1]:.4f}, {data.min(axis=0)[2]:.4f}]\n"
            info += f"  Max: [{data.max(axis=0)[0]:.4f}, {data.max(axis=0)[1]:.4f}, {data.max(axis=0)[2]:.4f}]\n"

        return (info,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeVectorfield": NukeVectorfield,
    "NukeVectorfieldInfo": NukeVectorfieldInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeVectorfield": "Nuke Vectorfield (LUT)",
    "NukeVectorfieldInfo": "Nuke Vectorfield Info",
}
