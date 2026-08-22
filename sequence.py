"""
Frame-sequence path utilities shared by the Nuke I/O nodes.

Pure path / os.stat helpers - nothing here reads pixel data, so every
function is cheap enough to call from IS_CHANGED on every queue.

Sequence patterns supported:
- %04d style (printf format): image.%04d.exr
- #### style (hash padding): image.####.exr
- Literal trailing frame number: image.0001.exr
- Bare paths (image.exr) via on-disk auto-detection
- Frame ranges: 1-100, 1-100x2 (every 2nd frame)
"""

import glob
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Sequence Pattern Utilities
# ============================================================================


def parse_frame_pattern(filepath: str) -> Tuple[str, Optional[str], int]:
    """
    Parse a filepath to detect frame pattern and padding.

    Supports:
    - %04d style: image.%04d.exr
    - #### style: image.####.exr
    - Literal frame number: image.0001.exr

    Grammar (strict on purpose so version tags and dates are not mistaken
    for frame numbers):
    - ``%0Nd`` anywhere in the path; bare ``%d`` is padding 1 (it expands
      unpadded, so it reports the padding it actually produces).
    - a run of ``#`` in the BASENAME only (a ``#`` in a directory name is
      not a frame token).
    - a literal frame number only when the digits are immediately preceded
      by ``.`` or ``_`` and immediately followed by the extension:
      ``name.0001.exr`` / ``render_0001.exr`` are sequences, while
      ``shot_v002.exr``, ``out_v2.exr``, ``frame1.png`` and ``img2024.jpg``
      are single files.

    Returns:
        (base_pattern, frame_spec, padding)
        - base_pattern: pattern with %0Nd placeholder
        - frame_spec: original frame specifier or None
        - padding: number of digits for padding
    """
    # Normalize path separators for consistent handling
    filepath = filepath.replace("\\", "/")
    slash = filepath.rfind("/")
    dir_prefix, basename = filepath[: slash + 1], filepath[slash + 1 :]

    # Check for %0Nd pattern
    match = re.search(r"%(\d*)d", filepath)
    if match:
        padding = int(match.group(1)) if match.group(1) else 1
        return filepath, match.group(0), padding

    # Check for #### pattern (basename only)
    match = re.search(r"(#+)", basename)
    if match:
        hashes = match.group(1)
        padding = len(hashes)
        pattern = dir_prefix + basename.replace(hashes, f"%0{padding}d")
        return pattern, hashes, padding

    # Check for a literal frame number in the basename: digits preceded by
    # "." or "_" and immediately followed by the extension.
    match = re.match(r"^(.*[._])(\d+)(\.[^.]+)$", basename)
    if match:
        frame_str = match.group(2)
        padding = len(frame_str)
        pattern = f"{dir_prefix}{match.group(1)}%0{padding}d{match.group(3)}"
        return pattern, frame_str, padding

    return filepath, None, 0


def expand_frame_pattern(pattern: str, frame: int, padding: int = 4) -> str:
    """
    Expand a frame pattern to an actual filename.

    Args:
        pattern: Pattern with %0Nd or #### placeholder
        frame: Frame number
        padding: Digit padding

    Returns:
        Expanded filename
    """
    # Handle %0Nd pattern
    if "%" in pattern:
        return pattern % frame

    # Handle #### pattern
    if "#" in pattern:
        hashes = re.search(r"#+", pattern).group(0)
        return pattern.replace(hashes, str(frame).zfill(len(hashes)))

    return pattern


def detect_sequence(filepath: str) -> Tuple[str, List[int], int]:
    """
    Detect an image sequence from a single file path.

    Args:
        filepath: Path to one file in the sequence

    Returns:
        (pattern, frames, padding)
        - pattern: Frame pattern string
        - frames: List of available frame numbers
        - padding: Digit padding
    """
    # Normalize path separators
    filepath = filepath.replace("\\", "/")

    pattern, frame_spec, padding = parse_frame_pattern(filepath)

    if frame_spec is None or padding == 0:
        # Not a sequence, single file
        if os.path.exists(filepath):
            return filepath, [0], 0
        return filepath, [], 0

    # Find all matching files
    # Convert pattern to glob pattern
    glob_pattern = re.sub(r"%\d*d", "*", pattern)
    glob_pattern = re.sub(r"#+", "*", glob_pattern)

    logger.info(f"[NukeRead] Searching for sequence with pattern: {glob_pattern}")

    matching_files = glob.glob(glob_pattern)

    if not matching_files:
        logger.warning(f"[NukeRead] No files found matching pattern: {glob_pattern}")
        # Check if directory exists
        directory = os.path.dirname(glob_pattern)
        if os.path.exists(directory):
            logger.info(f"[NukeRead] Directory exists: {directory}")
            # List files in directory for debugging
            try:
                files = os.listdir(directory)
                logger.info(
                    f"[NukeRead] Files in directory: {files[:10]}..."
                )  # Show first 10
            except Exception as e:
                logger.error(f"[NukeRead] Error listing directory: {e}")
        else:
            logger.warning(f"[NukeRead] Directory does not exist: {directory}")
        return pattern, [], padding

    logger.info(f"[NukeRead] Found {len(matching_files)} files in sequence")

    # Extract frame numbers. The "*" glob also matches files with a
    # different padding (beauty.1.exr next to beauty.0001.exr) and arbitrary
    # text in the frame field, so validate each hit against the pattern:
    # the frame field must be all digits and, for padding > 1, exactly
    # `padding` digits wide. Padding 1 (%d) accepts any digit count.
    token = re.search(r"%\d*d", pattern)
    digits_rx = r"(\d+)" if padding <= 1 else r"(\d{%d})" % padding
    frame_rx = re.compile(
        "^"
        + re.escape(pattern[: token.start()])
        + digits_rx
        + re.escape(pattern[token.end():])
        + "$",
        re.IGNORECASE if _is_windows() else 0,
    )

    frames = set()
    for f in matching_files:
        match = frame_rx.match(f.replace("\\", "/"))
        if match:
            frames.add(int(match.group(1)))

    return pattern, sorted(frames), padding


def auto_detect_sequence(filepath: str) -> Optional[Tuple[str, List[int], int]]:
    """
    Auto-detect an on-disk image sequence from a bare file path that has no
    frame token (no ####, %0Nd, or literal trailing frame number).

    Given e.g. "renders/beauty.exr", scans the directory for files whose
    basenames match "beauty<sep><digits>.exr" where <sep> is ".", "_" or
    empty ("beauty.0001.exr", "beauty_0001.exr", "beauty0001.exr").
    Matches are grouped by (separator, padding); the group with the most
    files wins (ties prefer the "." separator, then larger padding).
    Extension matching is case-insensitive on Windows.

    Stat-only (a single os.listdir, no file contents are read), so it is
    safe to call from IS_CHANGED on every queue.

    Returns:
        (pattern, frames, padding) where pattern contains a %0Nd token and
        frames is the sorted list of detected frame numbers, or None when
        no matching files are found.
    """
    filepath = filepath.replace("\\", "/")
    directory = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    stem, ext = os.path.splitext(basename)
    if not stem:
        return None

    try:
        entries = os.listdir(directory or ".")
    except OSError:
        return None

    ext_rx = f"(?i:{re.escape(ext)})" if _is_windows() else re.escape(ext)
    # The no-separator form (beauty0001.exr) is only allowed when the stem
    # ends in a non-digit; otherwise "img2024.jpg" would be split into
    # "img2024" + frame and a date/version-like name mistaken for a sequence.
    sep_rx = r"([._])" if stem[-1].isdigit() else r"([._]?)"
    name_rx = re.compile("^" + re.escape(stem) + sep_rx + r"(\d+)" + ext_rx + "$")

    # Group matching frame numbers by (separator, padding)
    groups: Dict[Tuple[str, int], List[int]] = {}
    for entry in entries:
        match = name_rx.match(entry)
        if not match:
            continue
        sep, digits = match.group(1), match.group(2)
        groups.setdefault((sep, len(digits)), []).append(int(digits))

    if not groups:
        return None

    # Most files wins; ties prefer "." separator, then larger padding
    sep_rank = {".": 2, "_": 1, "": 0}
    (sep, padding), frame_list = max(
        groups.items(),
        key=lambda item: (len(item[1]), sep_rank.get(item[0][0], 0), item[0][1]),
    )
    frames = sorted(set(frame_list))

    prefix = f"{directory}/" if directory else ""
    pattern = f"{prefix}{stem}{sep}%0{padding}d{ext}"
    return pattern, frames, padding


def parse_frame_range(range_str: str) -> List[int]:
    """
    Parse a frame range string like "1-100" or "1-100x2" (every 2nd frame).

    Args:
        range_str: Frame range string

    Returns:
        List of frame numbers
    """
    if not range_str or range_str.strip() == "":
        return []

    frames = []

    for part in range_str.split(","):
        part = part.strip()

        # Check for step (x2)
        step = 1
        if "x" in part:
            part, step_str = part.split("x")
            step = int(step_str)

        # Check for range (-)
        if "-" in part:
            start, end = part.split("-")
            frames.extend(range(int(start), int(end) + 1, step))
        else:
            frames.append(int(part))

    return sorted(set(frames))


def file_change_token(filepath: str) -> str:
    """
    Build a stable change-detection token for a single file path.

    Uses only os.stat (never reads file contents), so it is cheap enough to
    run from IS_CHANGED on every queue.

    Returns:
        "<path>|<mtime_ns>|<size>" when the file exists, otherwise a
        deterministic "missing:<path>" token so the fingerprint changes
        (and the node re-runs) when the file appears later.
    """
    try:
        st = os.stat(filepath)
        return f"{filepath}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return f"missing:{filepath}"


# ============================================================================
# File Counter Utilities
# ============================================================================


def _is_windows() -> bool:
    """Check if running on Windows."""
    import sys

    return (
        sys.platform.startswith("win")
        or sys.platform == "cygwin"
        or sys.platform == "msys"
    )


def _normalize_for_comparison(filename: str) -> str:
    """
    Normalize filename for comparison based on platform.
    Windows filesystem is case-insensitive, Linux/Mac are case-sensitive.
    """
    if _is_windows():
        return filename.lower()
    return filename


def _file_exists_case_aware(filepath: str) -> bool:
    """
    Check if file exists, handling case sensitivity properly for each platform.
    On Windows (case-insensitive), os.path.exists() already handles this.
    On Linux/Mac (case-sensitive), os.path.exists() is already correct.
    """
    # os.path.exists handles platform-specific case sensitivity correctly
    return os.path.exists(filepath)


def get_unique_filepath(filepath: str) -> str:
    """
    Get a unique filepath that doesn't overwrite any existing file.

    Handles various naming patterns and preserves zero-padding:
    - image_0001.exr -> image_0002.exr -> image_0003.exr (preserves padding)
    - image_1.exr -> image_2.exr -> image_3.exr
    - image-1.exr -> image-2.exr -> image-3.exr
    - image.png -> image1.png -> image2.png

    Works correctly on both Windows (case-insensitive) and Linux/Mac (case-sensitive).

    Args:
        filepath: The desired output filepath

    Returns:
        A filepath that is guaranteed not to exist
    """
    # If file doesn't exist, use it as-is
    if not _file_exists_case_aware(filepath):
        return filepath

    directory = os.path.dirname(filepath) or "."
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)

    # Build a set of existing filenames for efficient lookup
    # Normalize for case-insensitive comparison on Windows
    try:
        existing_files = set()
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                existing_files.add(_normalize_for_comparison(f))
    except (OSError, PermissionError):
        existing_files = set()

    def _exists(name: str) -> bool:
        """Check if a filename exists in the directory (case-aware)."""
        normalized = _normalize_for_comparison(name)
        if normalized in existing_files:
            return True
        # Double-check with filesystem (handles race conditions)
        return _file_exists_case_aware(os.path.join(directory, name))

    # Pattern 1: Check if base already ends with a separator and number (e.g., image_0001, image-3)
    # Match patterns like: name_0001, name-123, name.123
    separator_pattern = re.match(r"^(.+?)([_\-\.])(\d+)$", base)

    if separator_pattern:
        # File already has a separator+number pattern, increment it
        prefix = separator_pattern.group(1)
        separator = separator_pattern.group(2)
        num_str = separator_pattern.group(3)
        current_num = int(num_str)
        # Preserve the original padding (e.g., 0001 has padding of 4)
        padding = len(num_str)

        # Find next available number
        num = current_num + 1
        max_attempts = 100000
        while num < current_num + max_attempts:
            # Use zfill to preserve padding
            new_filename = f"{prefix}{separator}{str(num).zfill(padding)}{ext}"
            if not _exists(new_filename):
                return os.path.join(directory, new_filename)
            num += 1

        # Fallback: use timestamp
        import time

        timestamp = int(time.time() * 1000)
        return os.path.join(directory, f"{prefix}{separator}{timestamp}{ext}")

    # Pattern 2: Check if base ends with a number directly (e.g., image2, render001)
    direct_number_pattern = re.match(r"^(.+?)(\d+)$", base)

    if direct_number_pattern:
        prefix = direct_number_pattern.group(1)
        num_str = direct_number_pattern.group(2)
        current_num = int(num_str)
        # Preserve the original padding
        padding = len(num_str)

        # Find next available number
        num = current_num + 1
        max_attempts = 100000
        while num < current_num + max_attempts:
            # Use zfill to preserve padding
            new_filename = f"{prefix}{str(num).zfill(padding)}{ext}"
            if not _exists(new_filename):
                return os.path.join(directory, new_filename)
            num += 1

        # Fallback: use timestamp
        import time

        timestamp = int(time.time() * 1000)
        return os.path.join(directory, f"{prefix}{timestamp}{ext}")

    # Pattern 3: No number in filename, start with 1
    # Try appending number directly: image.png -> image1.png
    num = 1
    max_attempts = 100000
    while num < max_attempts:
        new_filename = f"{base}{num}{ext}"
        if not _exists(new_filename):
            return os.path.join(directory, new_filename)
        num += 1

    # Ultimate fallback: use timestamp
    import time

    timestamp = int(time.time() * 1000)
    return os.path.join(directory, f"{base}_{timestamp}{ext}")


def _versioned_sequence_pattern(pattern: str, version: int) -> str:
    """
    Insert a _<version> base-name suffix into a %0Nd sequence pattern.

    "render.%04d.exr" -> "render_1.%04d.exr"
    "render_%04d.exr" -> "render_1_%04d.exr"

    The suffix goes on the base name, before the separator that precedes the
    frame token, so every frame of a versioned sequence shares one
    consistent base name.
    """
    match = re.search(r"%\d*d", pattern)
    if match is None:
        base, ext = os.path.splitext(pattern)
        return f"{base}_{version}{ext}"
    insert_at = match.start()
    if insert_at > 0 and pattern[insert_at - 1] in "._-":
        insert_at -= 1
    return f"{pattern[:insert_at]}_{version}{pattern[insert_at:]}"


def _warn_stale_frames(
    sequence_pattern: str, first_frame: int, last_frame: int
) -> None:
    """
    Warn (never delete) about on-disk frames matching a %0Nd sequence
    pattern whose frame number lies beyond the just-written range — stale
    leftovers from a longer previous render.
    """
    sequence_pattern = sequence_pattern.replace("\\", "/")
    directory = os.path.dirname(sequence_pattern) or "."
    basename = os.path.basename(sequence_pattern)
    match = re.search(r"%\d*d", basename)
    if match is None:
        return

    flags = re.IGNORECASE if _is_windows() else 0
    frame_rx = re.compile(
        "^"
        + re.escape(basename[: match.start()])
        + r"(\d+)"
        + re.escape(basename[match.end():])
        + "$",
        flags,
    )

    try:
        entries = os.listdir(directory)
    except OSError:
        return

    stale = sorted(
        int(m.group(1))
        for f in entries
        if (m := frame_rx.match(f)) and int(m.group(1)) > last_frame
    )
    if stale:
        logger.warning(
            f"[NukeWrite] {len(stale)} stale frame(s) beyond written range "
            f"{first_frame}-{last_frame} left in {directory} "
            f"(e.g. frame {stale[0]}); not deleted"
        )

