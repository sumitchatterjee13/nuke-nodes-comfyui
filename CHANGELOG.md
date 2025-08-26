# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-08-26

### Added
- **Viewer Nodes**:
  - NukeViewer: Channel viewer with R/G/B/A shortcuts, gamma/gain controls, and mask overlay
  - NukeChannelShuffle: Rearrange and swap RGBA channels with full control
  - NukeRamp: Generate test ramps and gradients (horizontal, vertical, radial, diagonal, checkerboard)
  - NukeColorBars: Generate standard color bar patterns (SMPTE, RGB, primary colors, grayscale)

### Features
- Channel viewing shortcuts for debugging (R, G, B, A, RGB, RGBA, Luminance)
- Gamma and gain controls for proper viewing
- Test pattern generation for pipeline validation
- Channel manipulation and shuffling capabilities

## [1.0.0] - 2025-08-26

### Added
- Initial release of Nuke Nodes for ComfyUI
- **Merge Nodes**:
  - NukeMerge: Advanced merge node with 14 blend modes (over, add, multiply, screen, overlay, etc.)
  - NukeMix: Simple linear mixing between two images
- **Grade Nodes**:
  - NukeGrade: Professional color grading with lift/gamma/gain controls
  - NukeColorCorrect: HSV-based color correction
  - NukeLevels: Input/output levels adjustment
- **Transform Nodes**:
  - NukeTransform: 2D transformation with translate, rotate, scale, skew, and motion blur
  - NukeCornerPin: Four-corner perspective transformation
  - NukeCrop: Precise cropping with soft edges
- **Blur Nodes**:
  - NukeBlur: Gaussian blur with separate X/Y controls and multiple filter types
  - NukeMotionBlur: Directional motion blur with customizable samples and shutter
  - NukeDefocus: Depth-of-field style blur with optional depth map input

### Features
- All nodes support alpha channels
- Mask inputs for selective application
- Mix controls for blending with original images
- Professional-grade algorithms matching Nuke's behavior
- Optimized PyTorch implementations for GPU acceleration
- Comprehensive parameter controls matching Nuke's interface

### Technical Details
- Written in Python with PyTorch backend
- Modular architecture with separate files for each node category
- Comprehensive error handling and input validation
- Memory-efficient implementations
- Support for batch processing
