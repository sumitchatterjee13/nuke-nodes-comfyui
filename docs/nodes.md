# Node Documentation

## Merge Nodes

### NukeMerge

The NukeMerge node provides advanced compositing operations with multiple blend modes, similar to Nuke's Merge node.

**Inputs:**
- `image_a` (IMAGE): Background image
- `image_b` (IMAGE): Foreground image  
- `operation` (STRING): Blend mode selection
- `mix` (FLOAT): Blend amount (0.0 to 1.0)
- `mask` (IMAGE, optional): Mask for selective application

**Blend Modes:**
- `over`: Standard over operation (A over B)
- `add`: Additive blending
- `multiply`: Multiplicative blending
- `screen`: Screen blending (inverse multiply)
- `overlay`: Overlay blending
- `soft_light`: Soft light blending
- `hard_light`: Hard light blending
- `color_dodge`: Color dodge effect
- `color_burn`: Color burn effect
- `darken`: Keep darker pixels
- `lighten`: Keep lighter pixels
- `difference`: Absolute difference
- `exclusion`: Exclusion blending
- `subtract`: Subtract B from A
- `divide`: Divide A by B

**Usage:**
Use this node to composite two images with various blending modes. The mask input allows for selective application of the effect.

### NukeMix

Simple linear interpolation between two images.

**Inputs:**
- `image_a` (IMAGE): First image
- `image_b` (IMAGE): Second image
- `mix` (FLOAT): Mix factor (0.0 = image_a, 1.0 = image_b)

**Usage:**
Use this node for simple crossfades between images.

## Grade Nodes

### NukeGrade

Professional color grading with lift/gamma/gain controls.

**Inputs:**
- `image` (IMAGE): Input image
- `lift_r/g/b` (FLOAT): Lift (shadows) adjustment per channel
- `gamma_r/g/b` (FLOAT): Gamma (midtones) adjustment per channel  
- `gain_r/g/b` (FLOAT): Gain (highlights) adjustment per channel
- `multiply` (FLOAT): Overall brightness multiplier
- `offset` (FLOAT): Overall brightness offset
- `mix` (FLOAT): Effect strength
- `mask` (IMAGE, optional): Mask for selective application

**Usage:**
Primary color correction tool. Adjust lift for shadows, gamma for midtones, and gain for highlights.

### NukeColorCorrect

HSV-based color correction.

**Inputs:**
- `image` (IMAGE): Input image
- `hue` (FLOAT): Hue shift in degrees (-180 to 180)
- `saturation` (FLOAT): Saturation multiplier
- `value` (FLOAT): Value/brightness multiplier
- `contrast` (FLOAT): Contrast adjustment
- `mix` (FLOAT): Effect strength

**Usage:**
Quick color adjustments using HSV color space. Useful for overall color balance.

### NukeLevels

Input/output levels adjustment.

**Inputs:**
- `image` (IMAGE): Input image
- `input_black` (FLOAT): Input black point
- `input_white` (FLOAT): Input white point
- `gamma` (FLOAT): Gamma correction
- `output_black` (FLOAT): Output black point
- `output_white` (FLOAT): Output white point
- `mix` (FLOAT): Effect strength

**Usage:**
Adjust the tonal range and gamma of an image. Similar to Photoshop's Levels.

## Transform Nodes

### NukeTransform

2D transformation with translate, rotate, scale, and skew.

**Inputs:**
- `image` (IMAGE): Input image
- `translate_x/y` (FLOAT): Translation in pixels
- `rotate` (FLOAT): Rotation in degrees
- `scale_x/y` (FLOAT): Scale factors
- `skew_x/y` (FLOAT): Skew in degrees
- `center_x/y` (FLOAT): Transform center (0.0 to 1.0)
- `filter` (STRING): Interpolation method
- `motionblur` (FLOAT): Motion blur amount
- `shutter` (FLOAT): Motion blur shutter angle

**Usage:**
Apply geometric transformations to images with optional motion blur.

### NukeCornerPin

Four-corner perspective transformation.

**Inputs:**
- `image` (IMAGE): Input image
- `to1_x/y` through `to4_x/y` (FLOAT): Corner positions in normalized coordinates
- `filter` (STRING): Interpolation method

**Usage:**
Distort an image by repositioning its four corners. Useful for perspective correction.

### NukeCrop

Precise cropping with soft edges.

**Inputs:**
- `image` (IMAGE): Input image
- `left/right/top/bottom` (FLOAT): Crop boundaries (0.0 to 1.0)
- `softness` (FLOAT): Edge softness
- `resize` (STRING): Crop or format mode

**Usage:**
Crop images with optional soft edges. Format mode maintains original dimensions.

## Blur Nodes

### NukeBlur

Gaussian blur with separate X/Y controls.

**Inputs:**
- `image` (IMAGE): Input image
- `size_x/y` (FLOAT): Blur amount in X/Y directions
- `filter` (STRING): Blur type (gaussian, box, triangle, quadratic)
- `quality` (STRING): Quality setting (low, medium, high)
- `crop` (BOOLEAN): Crop to original bounds
- `mix` (FLOAT): Effect strength
- `mask` (IMAGE, optional): Mask for selective application

**Usage:**
Apply blur effects with precise control over direction and quality.

### NukeMotionBlur

Directional motion blur.

**Inputs:**
- `image` (IMAGE): Input image
- `distance` (FLOAT): Blur distance in pixels
- `angle` (FLOAT): Blur direction in degrees
- `samples` (INT): Number of samples for quality
- `shutter` (FLOAT): Shutter angle factor
- `center_bias` (FLOAT): Bias towards center samples
- `mix` (FLOAT): Effect strength

**Usage:**
Simulate motion blur in a specific direction.

### NukeDefocus

Depth-of-field style defocus blur.

**Inputs:**
- `image` (IMAGE): Input image
- `defocus` (FLOAT): Defocus amount
- `aspect_ratio` (FLOAT): Blur shape aspect ratio
- `quality` (STRING): Quality setting
- `method` (STRING): Blur method (gaussian, disk, hexagon)
- `mix` (FLOAT): Effect strength
- `depth_map` (IMAGE, optional): Depth map for variable blur
- `focus_distance` (FLOAT): Focus distance when using depth map

**Usage:**
Create depth-of-field effects with optional depth map control.

## Viewer Nodes

### NukeViewer

Nuke-style viewer with channel display options and viewing controls.

**Inputs:**
- `image` (IMAGE): Input image
- `channel` (STRING): Channel to display (rgba, rgb, red, green, blue, alpha, luminance)
- `gamma` (FLOAT): Gamma correction for viewing
- `gain` (FLOAT): Gain/brightness adjustment
- `show_overlay` (BOOLEAN): Show mask overlay
- `overlay_text` (STRING): Optional overlay text
- `mask` (IMAGE, optional): Mask for overlay visualization

**Channel Options:**
- `rgba`: Full color with alpha
- `rgb`: RGB channels only
- `red`: Red channel as grayscale
- `green`: Green channel as grayscale
- `blue`: Blue channel as grayscale
- `alpha`: Alpha channel as grayscale
- `luminance`: Calculated luminance

**Usage:**
Essential for inspecting images and channels during compositing. Use the channel shortcuts to quickly isolate R, G, B, or A channels. Gamma and gain controls help visualize over/under-exposed areas.

### NukeChannelShuffle

Rearrange and swap RGBA channels.

**Inputs:**
- `image` (IMAGE): Input image
- `red_from` (STRING): Source for red channel
- `green_from` (STRING): Source for green channel
- `blue_from` (STRING): Source for blue channel
- `alpha_from` (STRING): Source for alpha channel

**Channel Sources:**
- `red`, `green`, `blue`, `alpha`: Copy from existing channels
- `zero`: Set to black (0.0)
- `one`: Set to white (1.0)

**Usage:**
Useful for channel manipulation, creating custom alpha channels, or fixing channel order issues.

### NukeRamp

Generate test ramps and gradients.

**Inputs:**
- `width` (INT): Image width
- `height` (INT): Image height
- `ramp_type` (STRING): Type of ramp
- `color_start` (STRING): Start color (R,G,B format)
- `color_end` (STRING): End color (R,G,B format)
- `invert` (BOOLEAN): Invert the ramp
- `batch_size` (INT): Number of images to generate

**Ramp Types:**
- `horizontal`: Left to right gradient
- `vertical`: Top to bottom gradient
- `radial`: Circular gradient from center
- `diagonal`: Diagonal gradient
- `checkerboard`: Checkerboard pattern

**Usage:**
Generate test patterns for checking compositing operations, color spaces, and transformations.

### NukeColorBars

Generate standard color bar patterns.

**Inputs:**
- `width` (INT): Image width
- `height` (INT): Image height
- `pattern` (STRING): Color bar pattern type
- `batch_size` (INT): Number of images to generate

**Pattern Types:**
- `smpte`: Standard SMPTE color bars (75% intensity)
- `rgb_bars`: RGB primary color bars
- `primary_colors`: Basic primary colors
- `grayscale`: Grayscale steps

**Usage:**
Generate standard test patterns for monitor calibration, color pipeline testing, and reference comparisons.
