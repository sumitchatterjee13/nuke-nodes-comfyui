# Nuke Nodes for ComfyUI

A collection of ComfyUI custom nodes that replicate Nuke compositing nodes: Read/Write with EXR image sequences, multi-pass EXR, Merge, Grade, OCIO colour management, Transform, Reformat, Blur, FrameHold and more. Version 2.x follows your `$OCIO` config like Nuke does.

> **Upgrading from 1.x?** 2.0.0 is a major release with breaking changes (OCIO-driven colourspaces, `MASK` inputs, renamed/merged nodes). See [CHANGELOG.md](CHANGELOG.md) for the full list. 1.3.0 remains available on the Comfy Registry and as git tag `v1.3.0`.

## Features

- **Read/Write**: image sequences via OpenImageIO (EXR, DPX, TIFF, PNG, JPEG, HDR, TGA, BMP, WebP). Bare-path sequence detection, Nuke-style frame numbering, atomic overwrite or whole-sequence versioning, multi-channel EXR in/out.
- **Colour management**: OpenColorIO with your show config (`$OCIO`) or the built-in ACES 2.0 Studio config; colourspace, display/view and LUT (FileTransform) nodes; Read/Write convert to and from the config's working space.
- **Merge**: Porter-Duff and blend modes matching Nuke's Merge, plus Dissolve, Keymix and Constant.
- **Grade / ColorCorrect / Levels / Exposure**, all mask-aware.
- **Transform / CornerPin / Crop / Reformat** with a shared, bit-exact resampler (impulse, cubic, lanczos, area).
- **Blur / MotionBlur / Defocus**, **Viewer / ChannelShuffle / Ramp / ColorBars**, **FrameHold**.

## Installation

Via ComfyUI Manager (search "Nuke Nodes"), or:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sumitchatterjee13/nuke-nodes-comfyui.git nuke-nodes
cd nuke-nodes
pip install -r requirements.txt
```

Restart ComfyUI. All nodes appear under the **Nuke** menu.

### Requirements

- **ComfyUI** (provides PyTorch / NumPy)
- **OpenImageIO** (`>=2.5`) - required for all image I/O
- **OpenColorIO** (`>=2.2`, 2.5+ recommended) - colour management; nodes load without it but pass images through
- **OpenCV** (`opencv-python-headless`) - resampling for Transform / CornerPin / Reformat

All three install from `requirements.txt` as prebuilt wheels.

## Colour management

At startup the pack resolves its OCIO config in this order:

1. **`$OCIO`** - if the environment variable points at a `.ocio` file (a show or studio config), it is used and every colour dropdown (colourspaces, roles, displays, views) is built from it.
2. **Built-in ACES 2.0 Studio config** (`studio-config-v4.0.0_aces-v2.0_ocio-v2.5`) otherwise - works out of the box with 55 colourspaces including camera IDTs (ARRI, Sony, RED, Canon, Panasonic, Blackmagic, Apple, DJI).

The working space is the config's `scene_linear` role (ACEScg in the built-in config). Because dropdowns are built when ComfyUI starts, **changing `$OCIO` requires a ComfyUI restart** - the same model as launching Nuke from a show environment. `NukeOCIOInfo` reports which config is active.

## Nodes

### Read/Write (`Nuke/IO`)

- **NukeRead** - load an image or sequence
  - `file_path`: `%04d` / `####` patterns, a single frame (`shot.0001.exr`), or just the bare name (`shot.exr`) - with *load_as_sequence* on, siblings like `shot.0001.exr` / `shot_0001.exr` / `shot0001.exr` are detected automatically
  - `frame`, `frame_mode` (single / range / all), `first_frame` / `last_frame`, `missing_frames` (error / black / hold / nearest)
  - `colorspace`: `raw` (no conversion) or any colourspace of the active OCIO config - the file is converted *from* that space *into* the working space
  - Thumbnail preview in the node; re-executes only when the files on disk change
- **NukeWrite** - save an image or sequence (output node)
  - Relative paths go under ComfyUI's `output/`; absolute paths are used as-is; directories are created
  - Frames are numbered Nuke-style from `frame_start`: `name.0001.exr` (`frame_padding` digits); `%04d` / `####` in the path are honoured
  - `overwrite` **off** (default): if any target frame exists the whole batch is written under a versioned base name (`name_1.0001.exr`, ...) - nothing is ever clobbered; **on**: frames replace existing files atomically (temp file + rename)
  - `file_type`, `bit_depth` (8 / 16 / 16f / 32f), EXR `compression`, `colorspace` (working space -> chosen space), `channels` (rgb / rgba / all_channels for multi-pass EXR from a `NUKE_PASSES` input)
- **NukeReadInfo** - resolution, channels, bit depth, sequence range, missing frames, I/O and colour library status
- **NukeReadMultiPass** - load a multi-channel EXR into a `NUKE_PASSES` bundle + beauty image + pass list
- **NukeShufflePass** - pick one pass from a `NUKE_PASSES` bundle as an image

### Merge (`Nuke/Merge`)

- **NukeMerge** - `A` (foreground) over `B` (background), 28 operations (see table below), `mix`, optional `mask`
- **NukeDissolve** - `A` / `B` / `which` (0 = A, 1 = B)
- **NukeKeymix** - A where `mask` is 1, B where 0, with invert and mix
- **NukeConstant** (`Nuke/Generate`) - solid RGBA colour at any size

### Colour (`Nuke/Color`)

- **NukeOCIOColorSpace** - `in_colorspace` -> `out_colorspace` from the active config (roles offered as `role:<name>`)
- **NukeOCIODisplay** - display / view transform (Nuke's viewer process), forward or inverse
- **NukeOCIOFileTransform** - apply a LUT file (`.cube`, `.3dl`, `.spi1d`, `.spi3d`, `.csp`, `.clf`, `.ctf`, `.cdl`) from the `luts/` folder or a custom path, with direction, interpolation and mix
- **NukeOCIOInfo** - active config source, working space, colourspaces, roles, displays/views and LUT folder contents
- **NukeGrade** - lift / gamma / gain with per-channel offsets, multiply, offset, optional `mask`
- **NukeColorCorrect** - HSV hue / saturation / value + contrast, mix
- **NukeLevels** - input / output black & white points, gamma, mix
- **NukeExposure** - stops (with per-channel offsets), exposure type (stops / printer lights / film density), multiply, offset, preserve highlights, clamp, mix, optional `mask`

### Transform (`Nuke/Transform`)

- **NukeTransform** - translate, rotate (degrees, counter-clockwise), scale (uniform and per-axis), skew with XY/YX order, pivot (`-1` = centre), invert; filter `impulse` / `cubic` / `lanczos` / `area`. Identity parameters return the input bit-exactly. Output is RGBA with a coverage alpha.
- **NukeCornerPin** - perspective warp mapping the four image corners to `to1..to4` (Nuke's bottom-left origin)
- **NukeCrop** - crop with optional soft edge / reformat
- **NukeReformat** - Nuke's Reformat: `to format` / `to box` / `scale`, resize type `none` / `width` / `height` / `fit` / `fill` / `distort` (pixel-aspect aware), 16 format presets plus your own saved formats (`custom` + *save_format_as*), center / flip / flop / turn, black outside, filter

### Filter (`Nuke/Filter`)

- **NukeBlur** - gaussian / box / triangle / quadratic, separate X/Y size, quality, crop, optional `mask`, mix
- **NukeMotionBlur** - directional blur with shutter and centre bias
- **NukeDefocus** - disk / hexagon / gaussian bokeh, optional `depth_map` (MASK) + focus distance

### Viewer (`Nuke/Viewer`)

- **NukeViewer** - channel isolation (rgba / rgb / r / g / b / a / luminance), gamma, gain, optional red `mask` overlay
- **NukeChannelShuffle** - route any of red / green / blue / alpha / zero / one into each output channel
- **NukeRamp**, **NukeColorBars** - test pattern generators

### Time (`Nuke/Time`)

- **NukeFrameHold** - the batch is the timeline (`frame_start` = frame number of item 0). `increment` = 0 holds `first_frame` everywhere; `increment` = N steps through `first`, `first+N`, ... exactly like Nuke.

## Merge operations

| Operation | Description |
|-----------|-------------|
| **over** | A composited over B |
| **under** | A under B (B over A) |
| **plus** | A + B |
| **minus** | B - A |
| **multiply** | A × B |
| **screen** | 1 - (1-A)(1-B) |
| **overlay** | contrast blend of multiply and screen |
| **soft_light** / **hard_light** | gentle / strong contrast |
| **color_dodge** / **color_burn** | brighten / darken B by A |
| **darken** / **lighten** | min(A, B) / max(A, B) |
| **difference** | \|A - B\| |
| **exclusion** | A + B - 2AB |
| **average** | (A + B) / 2 |
| **divide** | B / A |
| **min** / **max** | minimum / maximum |
| **hypot** | sqrt(A² + B²) |
| **in** / **out** | A masked by B's alpha / A where B is transparent |
| **atop** | A where B exists, B elsewhere |
| **xor** | A and B where they don't overlap |
| **mask** / **stencil** | A with alpha Aα×Bα / A where B is transparent |
| **matte** | B with A's alpha as matte |
| **copy** | A |

## Conventions

- Images are ComfyUI `IMAGE` tensors (`[B,H,W,C]`, 0..1). Nodes accept RGB or RGBA and preserve alpha; Transform/CornerPin output RGBA (coverage alpha).
- Mask-type inputs take ComfyUI `MASK` tensors.
- File-reading nodes cache on the files' modification time and size, so they re-run only when the files change.

## Development

```bash
python -m pytest tests          # functional + interface tests (no running ComfyUI needed)
python tools/smoke_test.py      # quick standalone gate
python tools/verify_interfaces.py   # node interfaces vs tools/interface_manifest.json
python tools/dump_manifest.py   # regenerate the manifest after an INTENDED interface change
```

The interface manifest guards saved-workflow compatibility: any change to node ids, input names/order/types/defaults or outputs fails `verify_interfaces.py` until the manifest is regenerated deliberately.

## License

MIT - see [LICENSE](LICENSE).

## Acknowledgments

Inspired by The Foundry's Nuke. Built for the ComfyUI community.
