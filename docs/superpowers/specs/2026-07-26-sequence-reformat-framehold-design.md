# Design: Sequence UX fix, Write overwrite, NukeReformat, NukeFrameHold

Date: 2026-07-26. Approved by Sumit (overwrite-off = version whole sequence; auto frame naming = Nuke dot style; custom formats = persisted).

## 1. Sequence auto-detection (Read side)

Problem: `NukeRead` / `NukeReadMultiPass` require an explicit `####` / `%04d` token (or a literal frame number) in the path. A bare `render.exr` with `load_as_sequence=True` silently loads nothing/one file.

Design:
- New module-level helper in `io_nodes.py`:
  `auto_detect_sequence(filepath) -> Optional[Tuple[pattern, frames, padding]]`
  - Called only when `parse_frame_pattern` finds no frame spec (`frame_spec is None`).
  - Splits `filepath` into `dir`, `stem`, `ext`. Lists files in `dir` whose names match
    `^{stem}{sep}{digits}{ext}$` where `sep` is `.`, `_`, or empty (regex on basenames, case-insensitive extension on Windows).
  - Groups matches by `(sep, padding)`; picks the group with the most files (ties: prefer `.` sep, larger padding).
  - Returns `(pattern_with_%0Nd, sorted_frame_numbers, padding)` or `None` if no matches.
- Wire into `NukeRead.read_image`, `NukeRead.IS_CHANGED`, `NukeReadMultiPass.read_multipass`, `NukeReadMultiPass.IS_CHANGED`: when `load_as_sequence=True` and `parse_frame_pattern` yields no spec, try `auto_detect_sequence` before falling back to single-file. IS_CHANGED must mirror execute's resolution exactly (fingerprints stay stat-only).
- Existing behaviors (`####`, `%04d`, literal `name.0001.exr`) unchanged.

## 2. NukeWrite: overwrite semantics + Nuke-style frame naming

Problems: (a) no pattern in path appends `_0001` (non-Nuke); (b) `auto_sequence` uniquifies per-frame → re-renders produce mixed names (`img_0001_1.exr` next to `img_0002.exr`); (c) no explicit overwrite control.

Design:
- **Naming**: when path has no pattern token, auto-append Nuke-style dot frame number: `{base}.{frame:0{frame_padding}d}{ext}` (was `_`). Explicit `####`/`%04d` still honored. Frame numbers are always appended (existing behavior, unchanged) using `frame_start + batch_index`.
- **Replace `auto_sequence` input with `overwrite` BOOLEAN (default False)**. Old workflows: removed widget value is ignored on load; new input takes its default → safe no-clobber behavior preserved.
- **overwrite=True**: write each frame to its exact target path, atomically: write to a temp file in the same directory that keeps the real extension last... no — extension must remain the final suffix for OIIO/cv2 format inference, so temp name is `{base}.{frame}.__tmp{os.getpid()}{ext}`, then `os.replace(tmp, target)`. On write failure, remove the temp file. Never delete other existing files; if the target directory contains sequence frames beyond the written range (stale from a longer previous render), log a warning listing the count.
- **overwrite=False**: per-SEQUENCE collision handling. Before writing anything, compute the full list of target paths for the batch. If any exists (case-aware check, reuse `_file_exists_case_aware`), retry with base-name suffix `_1`, `_2`, ... applied to the whole batch (`render_1.0001.exr`, `render_1.0002.exr`, ...) until the entire set is collision-free. All frames of one execution always share one consistent base name. Remove the now-unused per-frame `get_unique_filepath` call from this path (keep the function; other callers may exist).
- `_write_multipass` gets the identical overwrite/naming semantics (it currently shares `auto_sequence`).

## 3. NukeReformat (new file `reformat_nodes.py`, category `Nuke/Transform`)

Faithful port of Nuke's Reformat (Foundry reference, Transform Nodes > Reformat).

Inputs:
- `image` (IMAGE)
- `type`: `["to format", "to box", "scale"]` default `to format`
- `format`: dropdown = built-in Nuke formats + user-saved formats + `"custom"`. Built-ins (name, w, h, pixel_aspect):
  PAL 720x576 1.09; NTSC 720x486 0.91; PAL_16:9 720x576 1.46; NTSC_16:9 720x486 1.21;
  HD_720 1280x720 1.0; HD_1080 1920x1080 1.0; UHD_4K 3840x2160 1.0;
  2K_DCP 2048x1080 1.0; 4K_DCP 4096x2160 1.0;
  1K_Super_35(full-ap) 1024x778 1.0; 2K_Super_35(full-ap) 2048x1556 1.0; 4K_Super_35(full-ap) 4096x3112 1.0;
  square_256 256x256 1.0; square_512 512x512 1.0; square_1K 1024x1024 1.0; square_2K 2048x2048 1.0
- `custom_width`, `custom_height` (INT), `custom_pixel_aspect` (FLOAT, default 1.0) — used when `format="custom"`
- `save_format_as` (STRING, default "") — when non-empty and `format="custom"`, persist `{name: [w, h, pa]}` to `folder_paths.get_user_directory()/nuke_nodes/user_formats.json` (create dirs; merge; log). Saved names appear in the dropdown (list built at INPUT_TYPES time by reading the JSON). `VALIDATE_INPUTS` returns True so values saved mid-session or from another machine don't fail combo validation.
- `box_width`, `box_height` (INT), `box_pixel_aspect` (FLOAT) — for `to box`
- `scale` (FLOAT, default 1.0, min 0.01) — for `scale`; output dims = round(input * scale) per axis
- `resize_type`: `["none", "width", "height", "fit", "fill", "distort"]` default `width`
- `center` (BOOLEAN, True), `flip` (BOOLEAN, False = vertical flip), `flop` (BOOLEAN, False = horizontal), `turn` (BOOLEAN, False = rotate 90° CCW)
- `filter`: `["impulse", "cubic", "lanczos", "area"]` default `cubic` → cv2 INTER_NEAREST / INTER_CUBIC / INTER_LANCZOS4 / INTER_AREA (documented approximation of Nuke's filter set)
- `black_outside` (BOOLEAN, True): True pads black; False replicates edge pixels (cv2 BORDER_CONSTANT vs BORDER_REPLICATE)

Semantics (pixel-aspect aware; display width = w × pa):
- `turn` applies first (like Nuke's transform order for reformat), then flip/flop.
- Output canvas (W_out, H_out, pa_out) from type: to format → format; to box → box fields; scale → scaled input dims, pa preserved.
- Scale factors: sx_raw = (W_out·pa_out)/(W_in·pa_in), sy_raw = H_out/H_in.
  - none → s=1 (no resample; image placed on canvas, crop/pad per `center`)
  - width → s = sx_raw both axes; height → s = sy_raw both axes
  - fit → min(sx_raw, sy_raw) [Foundry: smallest side fills]; fill → max(sx_raw, sy_raw)
  - distort → sx=sx_raw, sy=sy_raw independently
  (aspect-preserving modes resample uniformly in display space: pixel dims scale by s·(pa_in/pa_out) horizontally, s vertically)
- Resampled image is placed on the output canvas: centered when `center=True`, else lower-left aligned (image coordinates: bottom-left origin like Nuke → in array terms align to bottom row). Overflow is cropped; underflow padded per `black_outside`.
- Whole batch processed; output IMAGE only. Alpha (4-channel) passes through the same transform.

## 4. NukeFrameHold (new file `time_nodes.py`, category `Nuke/Time`)

Batch-as-timeline semantics. Inputs: `image` (IMAGE batch, B frames), `first_frame` (INT, default 1), `increment` (INT, default 0, min 0), `frame_start` (INT, default 1 — frame number of batch item 0).

For output item i (timeline frame t = frame_start + i):
- increment == 0 → held = first_frame
- increment > 0 → held = first_frame + increment · floor((t − first_frame)/increment)
- clamp held to [frame_start, frame_start + B − 1]; out[i] = in[held − frame_start]

Output: IMAGE, same batch length. Pure function → no IS_CHANGED.

## Integration (done by coordinator, not agents)

- Register both new files in `__init__.py` (imports + mapping updates, same pattern as existing files).
- Add `"Nuke/Time"` to `NODE_CATEGORIES` in `version.py`.
- Node IDs: `NukeReformat`, `NukeFrameHold`; display names "Nuke Reformat", "Nuke FrameHold".

## Testing

Pytest collection from repo root is broken (pre-existing relative-import issue) — verify via `python -m py_compile`, plus self-contained scratchpad harnesses that stub `folder_paths` and exercise: auto-detect (dot/underscore/no-sep, mixed padding, no matches), overwrite on/off (atomicity, whole-sequence versioning), reformat math (fit/fill/width/height/none/distort with pixel aspect, center on/off, turn/flip/flop, format persistence), framehold formula (increment 0/1/5, clamping).
