# Nuke Nodes for ComfyUI

A comprehensive collection of ComfyUI custom nodes that replicate the functionality of popular Nuke compositing nodes. This package brings professional compositing workflows to ComfyUI with nodes for merging, color grading, transformations, and blur effects.

## Features

- **Merge Nodes**: Advanced blending operations with multiple blend modes
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
- **NukeMerge**: Multi-input merge node with blend modes (Over, Add, Multiply, Screen, etc.)
- **NukeMix**: Blend two images with customizable mix factor

### Grade Nodes
- **NukeGrade**: Professional color grading with lift, gamma, gain controls
- **NukeColorCorrect**: HSV-based color correction
- **NukeLevels**: Input/output levels adjustment

### Transform Nodes
- **NukeTransform**: 2D transformation with translate, rotate, scale, skew
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
