# Changelog

All notable changes to this project will be documented in this file.

## [2.0.1] - 2026-08-21

Correctness release found while writing the behavioural specification. No interface changes - saved 2.0.0 workflows load unchanged.

### Fixed

- **Frame grammar was too eager**: a literal frame number is now recognised only when preceded by `.` or `_` and followed by the extension (`name.0001.exr`, `render_0001.exr`); version-like names (`shot_v002.exr`, `out_v2.exr`) and names ending in digits (`img2024.jpg`) are single files. `#` tokens are recognised in the file name only, not in directory names. `%d` means padding 1.
- **NukeWrite collapsed sequences when the path had a token but no extension** (`out.####` wrote every frame to `out.exr`); the token is kept and `.exr` appended (`out.0001.exr`, ...). Returned paths use the native separator consistently.
- **NukeRead crashed on a missing leading frame** for footage that is not 512x512; placeholder frames now take the resolution of the first loaded frame. `missing_frames="error"` now raises (Nuke semantics) instead of logging and filling black.
- `detect_sequence` no longer mixes paddings or returns duplicate frame numbers.
- Preview thumbnails use unique (UUID) file names - previously names derived from tensor ids could collide.
- **NukeMotionBlur** sampled half a pixel off (even `samples=1` blurred and darkened edges); the grid is pixel-centre aligned and `samples=1` is an exact identity.
- **NukeDefocus**: `aspect_ratio` now widens the bokeh horizontally (it acted on the vertical axis); `hexagon` is a real hexagonal kernel (it was identical to low-quality gaussian); an unknown `method` falls back to `disk` with a warning instead of crashing; the blur never degenerates to a no-op at higher quality.
- Blur/Defocus no longer crash on images smaller than the kernel (replicate padding fallback).
- **NukeBlur `crop` now does what Nuke's does**: on (default) treats outside the image as black so blurs fade at the format edge; off treats outside as the edge colour. Note: default-parameter renders change within one kernel radius of the image border compared with 2.0.0, which always used reflect padding; set `crop` off for the previous look.
- NukeReformat returns its result on the input's device and dtype (it always returned CPU float32).
- NukeOCIOFileTransform logs a warning when `custom_lut_path` is set but missing instead of silently applying the dropdown LUT; `~` and environment variables in the path are expanded.

## [2.0.0] - 2026-08-20

Major release. Saved workflows from 1.x need re-wiring where noted below; 1.3.0 stays available on the Comfy Registry and as git tag `v1.3.0`.

### Breaking

- **OpenImageIO is now required** for all image I/O (it is installed automatically from `requirements.txt`). The OpenCV/PIL read/write fallbacks are gone.
- **Colour management follows `$OCIO`.** When the `OCIO` environment variable points at a config (show/studio config), every colour dropdown is built from it at ComfyUI startup; otherwise the built-in ACES 2.0 Studio config is used. The `config` input was removed from NukeOCIOColorSpace and NukeOCIODisplay.
- **NukeRead / NukeWrite `colorspace`** now lists `raw` plus every colourspace of the active OCIO config (was `raw/sRGB/linear/ACEScg`); conversion is done by OCIO against the config's `scene_linear` role. Workflows using `sRGB` / `linear` must pick the equivalent config name (e.g. `sRGB Encoded Rec.709 (sRGB)`, `Linear Rec.709 (sRGB)`).
- **Mask inputs are `MASK` typed** (ComfyUI convention) on NukeBlur, NukeDefocus (`depth_map`), NukeGrade, NukeExposure and NukeViewer - existing IMAGE links into those sockets are dropped on load.
- **NukeMix -> NukeDissolve** (inputs `A`, `B`, `which`; 0 = A, 1 = B - Nuke's Dissolve knob).
- **NukeExposureAdvanced merged into NukeExposure** (per-channel stops, exposure type, preserve highlights, multiply/offset, clamp, mix, mask in one node).
- **NukeVectorfield and NukeVectorfieldInfo removed** - NukeOCIOFileTransform is the single LUT engine (OCIO reads .cube/.3dl/.spi1d/.spi3d/.csp/.clf/.ctf/.cdl); the LUT folder listing moved into NukeOCIOInfo.
- **NukeTransform / NukeCornerPin filters** are now `impulse`, `cubic`, `lanczos`, `area` (resampled with OpenCV, shared with NukeReformat) instead of the 11 Nuke names that only ever produced bilinear.
- **NukeCornerPin is a true perspective (homography) warp** with Nuke's bottom-left origin for the `to*` corners; 1.x used a bilinear corner interpolation with a top-left origin, so non-identity results differ.

### Fixed

- Half-pixel resampling offset in NukeTransform / NukeCornerPin: identity parameters now return the input bit-exactly.
- NukeTransform `rotate` is now counter-clockwise for positive degrees, as documented (1.x rotated clockwise).
- NukeViewer crashed when showing a mask overlay on an RGBA image.
- Multi-pass EXR writes colour-converted bare `N` / `Z` / `P` data passes; they are now recognised as data and left untouched.
- `missing_frames="hold"` in NukeRead converted the held frame twice.

### Added

- `ocio_config.py`: shared `$OCIO` -> built-in ACES resolver, config-driven colourspace/role/display/view lists, tensor transforms.
- Roles are offered as `role:<name>` entries in the OCIO colourspace dropdowns.
- A working pytest suite (`python -m pytest tests`) alongside `tools/smoke_test.py`; the `io_nodes.py` monolith is split into `sequence.py`, `image_io.py`, `preview.py` and `io_nodes.py`; shared mask/mix and resampling helpers in `utils.py`.

### Removed

- Stale `package.json` (node list duplicated pyproject.toml); the old, uncollectable `tests/` files.

## [1.3.0] - 2026-08-20

### Fixed

- **NukeMotionBlur, NukeTransform, NukeCornerPin** now work on image batches larger than 1 frame (the sampling grid was built for a single frame and `grid_sample` raised a batch-size mismatch).

### Added

- `tools/verify_interfaces.py` + `tools/interface_manifest.json`: checks every node's ids, inputs (names, order, types, defaults, options), outputs and flags against a reference manifest - guards saved-workflow compatibility. `tools/smoke_test.py`: standalone functional regression test (no running ComfyUI needed).

## [1.2.0] - 2026-08-20

### Fixed

- **NukeLevels** no longer crashes on execution (`torch.clamp()` was called on plain float widget values, so the node raised `TypeError` on every run in all previous releases).

### Known issues (pre-existing)

- **NukeMotionBlur** fails on batches larger than 1 frame (sampling grid is not batch-expanded).
- **NukeTransform / NukeCornerPin** have a half-pixel resampling offset at identity parameters.

## [1.1.0] - 2026-07-26

### Added

- **NukeReformat** (`Nuke/Transform`): full port of Nuke's Reformat node — `to format` / `to box` / `scale` types, all six resize types (`none`, `width`, `height`, `fit`, `fill`, `distort`) with pixel-aspect-aware math, 16 standard Nuke format presets, persisted custom formats (save via *save_format_as*, stored in the ComfyUI user directory), center/flip/flop/turn, black-outside vs edge-replicate, and filter selection.
- **NukeFrameHold** (`Nuke/Time`, new category): batch-as-timeline frame holding with Nuke's exact `first frame` / `increment` semantics.
- **Sequence auto-detection**: NukeRead and NukeReadMultiPass now detect image sequences from a bare path (e.g. `render.exr` finds `render.0001.exr`, `render_0001.exr`, or `render0001.exr` siblings) when *load_as_sequence* is enabled — no `####`/`%04d` token required.
- **NukeWrite `overwrite` option**: off (default) = if any target frame exists, the entire batch is written under one versioned base name (`shot_1.0001.exr`, ...), never clobbering and never mixing names; on = frames atomically replace existing files at their exact paths (temp file + rename, crash-safe). Stale frames from a previous longer render are warned about, never deleted. Applies to multi-pass EXR writing too.

### Changed

- **Auto frame naming is now Nuke-style**: paths without a pattern token get dot-separated frame numbers (`name.0001.exr`; previously `name_0001.exr`).
- **`auto_sequence` input removed from NukeWrite**, replaced by `overwrite` (existing workflows load fine; the new default keeps no-clobber behavior).
- All node logging now uses Python's `logging` module instead of `print()`.
- `WEB_DIRECTORY` uses the standard relative form; minimum Python aligned to 3.9; `requires-comfyui` declared for the Comfy Registry.

### Fixed

- **Caching**: nodes no longer force re-execution on every queue. Pure color/OCIO nodes cache normally on their inputs; file-reading nodes (NukeRead, NukeReadMultiPass, NukeOCIOFileTransform, NukeVectorfield) re-run only when the files on disk actually change (stat-based fingerprints).
- NukeRead is no longer marked as an output node, so it doesn't execute when nothing downstream needs it.
- Package metadata placeholders (author email, repository URL, stale category list) corrected.

## [1.0.0] - 2024-09-21

Initial release: Read/Write with EXR sequence support, multi-pass EXR loader/shuffle, Merge/Keymix, OCIO ACES 2.0 color management, LUT (Vectorfield), Grade/ColorCorrect/Levels/Exposure, Transform/CornerPin/Crop, Blur/MotionBlur/Defocus, Viewer/ChannelShuffle/Ramp/ColorBars, Constant.
