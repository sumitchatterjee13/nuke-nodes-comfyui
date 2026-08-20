# Changelog

All notable changes to this project will be documented in this file.

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
