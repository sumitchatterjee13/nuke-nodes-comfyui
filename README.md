# Nuke Nodes for ComfyUI

A comprehensive collection of ComfyUI custom nodes that replicate the functionality of popular Nuke compositing nodes. This package brings professional compositing workflows to ComfyUI with nodes for merging, color grading, transformations, and blur effects.

## Features

- **Merge Nodes**: Advanced blending operations with Porter-Duff and blend modes matching Nuke's Merge node
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


## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by The Foundry's Nuke compositing software
- Built for the ComfyUI community
