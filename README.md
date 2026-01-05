# Nuke Nodes for ComfyUI

A comprehensive collection of ComfyUI custom nodes that replicate the functionality of popular Nuke compositing nodes. This package brings professional compositing workflows to ComfyUI with nodes for merging, color grading, transformations, and blur effects.

## Features

- **Read/Write Nodes**: Load and save images with full sequence support via OpenImageIO
- **Merge Nodes**: Advanced blending operations with Porter-Duff and blend modes matching Nuke's Merge node
- **Color Management**: ACES 2.0 and OCIO color space transformations with built-in configs
- **LUT Support**: Load and apply 1D/3D LUTs for color grading (.cube, .3dl, .spi formats)
- **Generate Nodes**: Create solid colors, ramps, and test patterns
- **Grade Nodes**: Professional color correction and grading tools
- **Transform Nodes**: Precise geometric transformations with filtering options
- **Blur Nodes**: Various blur algorithms including Gaussian, motion, and directional blur

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sumitchatterjee13/nuke-nodes-comfyui.git nuke-nodes
```

2. Install the required dependencies:
```bash
cd nuke-nodes
pip install -r requirements.txt
```

3. Restart ComfyUI

## Node Categories

### Read/Write Nodes (IO)

**OpenImageIO provides professional-grade image I/O with wide format support.**

- **NukeRead**: Load images or image sequences from disk
  - **file_path**: Path to image or sequence (supports `%04d` and `####` patterns)
  - **frame**: Current frame number to load
  - **frame_mode**: single, range, or all frames
  - **first_frame/last_frame**: Frame range specification
  - **missing_frames**: How to handle missing frames
    - `error`: Print error message
    - `black`: Return black frame
    - `hold`: Use previous frame
    - `nearest`: Use nearest available frame
  - **colorspace**: Basic colorspace conversion (raw, sRGB, linear, ACEScg)
  - Supports: EXR, TIFF, PNG, JPEG, DPX, HDR, TGA, BMP, PSD, and many more

- **NukeWrite**: Save images or image sequences to disk
  - **file_path**: Output path (supports `%04d` and `####` patterns)
    - Relative paths: Saved to ComfyUI's output directory (e.g., `test1` → `output/test1.exr`)
    - Subdirectories: Created automatically (e.g., `Test1/test1` → `output/Test1/test1.exr`)
    - Absolute paths: Used as-is (e.g., `C:/renders/test1.exr`)
  - **frame_start**: Starting frame number for sequences
  - **file_type**: Output format (exr, tiff, png, jpg, dpx, hdr, tga, bmp)
  - **bit_depth**: Output bit depth
    - `8`: 8-bit unsigned integer
    - `16`: 16-bit unsigned integer
    - `16f`: 16-bit float (half)
    - `32f`: 32-bit float
  - **compression**: EXR compression type
    - `none`, `rle`, `zip`, `zips`, `piz`, `pxr24`, `b44`, `b44a`, `dwaa`, `dwab`
  - **frame_padding**: Number of digits for frame numbers (default: 4)
    - `4` = 0001, 0002, 0003...
    - `5` = 00001, 00002, 00003...
  - **auto_sequence**: Auto-increment filename on each run (default: false)
    - `true`: Saves as image1.png, image2.png, image3.png... (never overwrites)
    - `false`: Always saves to same filename (overwrites previous)
    - Only applies when no frame pattern (%04d or ####) is used
  - **create_directories**: Auto-create output directories
  - **colorspace**: Apply colorspace conversion on write

- **NukeReadInfo**: Display information about image files
  - Shows resolution, channels, bit depth, compression
  - Detects sequences and shows frame range
  - Reports missing frames in sequences
  - Lists available I/O libraries

**Sequence Patterns:**

- `image.%04d.exr` - Printf style with padding (0001, 0002, ...)
- `image.####.exr` - Hash padding (0001, 0002, ...)
- `image.0001.exr` - Auto-detects sequence from single frame

**I/O Library Priority:** OpenImageIO > OpenCV > PIL

- Install OpenImageIO for best format support: `pip install OpenImageIO`

### Merge Nodes
- **NukeMerge**: Multi-input merge node matching Nuke's Merge behavior
  - **A** = Foreground (top layer)
  - **B** = Background (bottom layer)
  - **Porter-Duff operations**: over, under, in, out, atop, xor
  - **Blend modes**: plus, minus, multiply, screen, overlay, soft_light, hard_light, color_dodge, color_burn, darken, lighten, difference, exclusion, average, divide, min, max, hypot
  - **Matte operations**: mask, stencil, matte, copy
  - **mix**: Controls the blend amount between B and the result
  - **mask**: Optional mask input to limit the merge effect
- **NukeMix**: Simple linear blend between two images with customizable mix factor

### Generate Nodes
- **NukeConstant**: Generate solid color images with configurable RGBA values, width, and height

### Color Nodes (OCIO)

**OpenColorIO 2.5+ with hardcoded ACES 2.0 Studio Config - reliable colorspaces every time!**

#### 🎯 Using: ACES 2.0 Studio Config (Hardcoded)
- **Config**: `studio-config-v4.0.0_aces-v2.0_ocio-v2.5`
- **Total Colorspaces**: 55 (all hardcoded for reliability)
- **No config path needed** - works out of the box!

**Why Hardcoded?**

ComfyUI's architecture doesn't support dynamic population of dropdown menus based on runtime configurations. To provide the best user experience with maximum colorspace coverage, we've chosen ACES 2.0 Studio Config as the standard and hardcoded all 55 colorspace names directly into the node definitions. This approach offers several benefits:

- ✅ **Immediate availability** - All colorspaces show up in dropdowns without configuration
- ✅ **Consistent behavior** - Same colorspaces available across all installations
- ✅ **Maximum coverage** - ACES 2.0 Studio includes comprehensive camera IDTs from all major manufacturers
- ✅ **Professional workflow** - Industry-standard ACES 2.0 color pipeline
- ✅ **No external files** - No need to manage OCIO config files or paths

- **NukeOCIOColorSpace**: Transform between color spaces using OpenColorIO
  - **config**: Shows "ACES 2.0 Studio Config" (locked to this config)
  - **in_colorspace/out_colorspace**: Choose from 55 hardcoded colorspaces
  - Common workflow: ACEScg → sRGB Encoded Rec.709 (sRGB) for rendering
  - Supports all ACES working spaces and camera IDTs
  - Requires: `pip install opencolorio` (version 2.2+)

- **NukeOCIODisplay**: Apply display/view transforms (like Nuke's viewer process)
  - Convert scene-referred to display-referred images
  - **config**: Shows "ACES 2.0 Studio Config" (locked to this config)
  - **display**: sRGB - Display, Rec.1886 Rec.709 - Display, P3-D65 - Display, Rec.2100-PQ - Display, etc.
  - **view**: ACES 2.0 - SDR Video, Raw, and more
  - **input_colorspace**: Source color space (e.g., ACEScg)

- **NukeOCIOInfo**: Display information about the current OCIO configuration
  - Shows OCIO version and ACES 2.0 Studio Config details
  - Lists all 55 hardcoded colorspaces
  - Lists camera colorspaces with full manufacturer support
  - Lists displays and views
  - Useful for debugging OCIO setup

**Available Camera Colorspaces** (hardcoded from ACES 2.0 Studio Config):

- **ARRI**: LogC3 (EI800), LogC4, Linear ARRI Wide Gamut 3/4
- **Sony**: S-Log3 S-Gamut3, S-Log3 S-Gamut3.Cine, S-Log3 Venice S-Gamut3/Cine
- **RED**: Log3G10 REDWideGamutRGB, Linear REDWideGamutRGB
- **Canon**: CanonLog2/3 CinemaGamut D55, Linear CinemaGamut D55
- **Panasonic**: V-Log V-Gamut, Linear V-Gamut
- **Blackmagic**: BMDFilm WideGamut Gen5, DaVinci Intermediate WideGamut, Linear BMD WideGamut Gen5, Linear DaVinci WideGamut
- **Apple**: Apple Log
- **DJI**: D-Log D-Gamut, Linear D-Gamut

**All 55 Hardcoded Colorspaces:**
ACES2065-1, ACEScc, ACEScct, ACEScg, ADX10, ADX16, ARRI LogC3 (EI800), ARRI LogC4, Apple Log, BMDFilm WideGamut Gen5, Camera Rec.709, CanonLog2 CinemaGamut D55, CanonLog3 CinemaGamut D55, D-Log D-Gamut, DaVinci Intermediate WideGamut, Display P3 - Display, Display P3 HDR - Display, Gamma 1.8 Encoded Rec.709, Gamma 2.2 Encoded AP1, Gamma 2.2 Encoded AdobeRGB, Gamma 2.2 Encoded Rec.709, Gamma 2.2 Rec.709 - Display, Gamma 2.4 Encoded Rec.709, Linear ARRI Wide Gamut 3, Linear ARRI Wide Gamut 4, Linear AdobeRGB, Linear BMD WideGamut Gen5, Linear CinemaGamut D55, Linear D-Gamut, Linear DaVinci WideGamut, Linear P3-D65, Linear REDWideGamutRGB, Linear Rec.2020, Linear Rec.709 (sRGB), Linear S-Gamut3, Linear S-Gamut3.Cine, Linear V-Gamut, Linear Venice S-Gamut3, Linear Venice S-Gamut3.Cine, Log3G10 REDWideGamutRGB, P3-D65 - Display, Raw, Rec.1886 Rec.709 - Display, Rec.2100-HLG - Display, Rec.2100-PQ - Display, S-Log3 S-Gamut3, S-Log3 S-Gamut3.Cine, S-Log3 Venice S-Gamut3, S-Log3 Venice S-Gamut3.Cine, ST2084-P3-D65 - Display, V-Log V-Gamut, sRGB - Display, sRGB Encoded AP1, sRGB Encoded P3-D65, sRGB Encoded Rec.709 (sRGB)

### LUT Nodes (Vectorfield)

- **NukeVectorfield**: Load and apply Look-Up Tables (LUTs) for color grading
  - **Supported formats**: .cube, .3dl, .spi1d, .spi3d, .lut
  - **lut_file**: Select from available LUTs in `./luts` folder
  - **intensity**: Mix amount (0-2.0 range)
    - 0.0 = original image
    - 1.0 = full LUT effect
    - >1.0 = extrapolated LUT effect
  - **custom_lut_path**: Optional path to LUT file outside luts folder
  - Automatic LUT caching for performance
  - Supports both 1D and 3D LUTs with trilinear interpolation
  - HDR compatible (values > 1.0)

- **NukeVectorfieldInfo**: Display LUT file information
  - Shows title, type (1D/3D), size, domain
  - Data range and entry count
  - Lists available LUTs in luts folder

**Using LUTs:**

1. Place your LUT files (.cube, .3dl, etc.) in the `luts` folder
2. Add Vectorfield node to your workflow
3. Select LUT from dropdown or provide custom path
4. Adjust intensity to taste

### Grade Nodes
- **NukeGrade**: Professional color grading with lift, gamma, gain controls
- **NukeColorCorrect**: HSV-based color correction
- **NukeLevels**: Input/output levels adjustment

### Transform Nodes
- **NukeTransform**: 2D transformation matching Nuke's Transform node
  - **translate_x/y**: Move image in pixels
  - **rotate**: Counter-clockwise rotation in degrees
  - **scale**: Uniform scale multiplier
  - **scale_x/y**: Non-uniform scale
  - **skew_x/y**: Skew transformation in degrees
  - **skew_order**: XY or YX application order
  - **center_x/y**: Pivot point in pixels (-1 = image center)
  - **filter**: impulse, cubic (default), keys, simon, rifman, mitchell, parzen, notch, lanczos4, lanczos6, sinc4
  - **invert**: Invert the transformation matrix
- **NukeCornerPin**: Four-corner distortion for perspective correction
- **NukeCrop**: Precise cropping with soft edges

### Blur Nodes
- **NukeBlur**: Gaussian blur with separate X/Y controls
- **NukeMotionBlur**: Directional motion blur
- **NukeDefocus**: Depth-of-field style blur

### Viewer Nodes
- **NukeViewer**: Channel viewer with R/G/B/A shortcuts and gamma/gain controls
- **NukeChannelShuffle**: Rearrange and swap RGBA channels
- **NukeRamp**: Generate test ramps and gradients
- **NukeColorBars**: Generate standard color bar patterns

## Merge Operations Reference

| Operation | Description |
|-----------|-------------|
| **over** | A composited over B (A on top of B) |
| **under** | A under B (equivalent to B over A) |
| **plus** | Additive blend: A + B |
| **minus** | Subtractive blend: B - A |
| **multiply** | Darkening blend: A × B |
| **screen** | Lightening blend: 1 - (1-A)(1-B) |
| **overlay** | Contrast blend combining multiply and screen |
| **soft_light** | Gentle contrast adjustment |
| **hard_light** | Strong contrast adjustment |
| **color_dodge** | Brightens B based on A |
| **color_burn** | Darkens B based on A |
| **darken** | min(A, B) |
| **lighten** | max(A, B) |
| **difference** | \|A - B\| |
| **exclusion** | A + B - 2×A×B |
| **average** | (A + B) / 2 |
| **divide** | B / A |
| **min** | Minimum of A and B |
| **max** | Maximum of A and B |
| **hypot** | sqrt(A² + B²) |
| **in** | A masked by B's alpha |
| **out** | A where B is transparent |
| **atop** | A where B exists, B elsewhere |
| **xor** | A and B where they don't overlap |
| **mask** | A with alpha = Aα × Bα |
| **stencil** | A where B is transparent |
| **matte** | B with A's alpha as matte |
| **copy** | Just A |

## Usage

All nodes appear in the ComfyUI node menu under the "Nuke" category. Each node is designed to match the behavior and parameters of its Nuke counterpart as closely as possible.

## Requirements

- **ComfyUI** (provides PyTorch automatically)
- **OpenCV** (may need separate installation: `pip install opencv-python`)
- **NumPy** (usually included with PyTorch)
- **OpenImageIO** (optional, for Read/Write nodes: `pip install OpenImageIO`)
  - Required for NukeRead, NukeWrite, and NukeReadInfo nodes
  - Provides professional-grade image I/O with wide format support (EXR, DPX, etc.)
  - Falls back to OpenCV or PIL if not available
- **OpenColorIO** (optional, for OCIO color management: `pip install opencolorio`)
  - Required for NukeOCIOColorSpace, NukeOCIODisplay, and NukeOCIOInfo nodes
  - Version 2.5+ recommended for built-in ACES 2.0 support


## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by The Foundry's Nuke compositing software
- Built for the ComfyUI community
