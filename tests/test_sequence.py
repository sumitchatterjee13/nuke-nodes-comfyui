"""Frame-sequence grammar and on-disk detection (sequence.py), real temp files."""
import os

import pytest


@pytest.fixture(scope="module")
def seq(pack):
    from nuke_nodes import sequence
    return sequence


def _touch(directory, *names):
    os.makedirs(directory, exist_ok=True)
    for name in names:
        with open(os.path.join(directory, name), "wb"):
            pass


# ------------------------------------------------------------ parse grammar

@pytest.mark.parametrize("path, expected", [
    ("name.0001.exr", ("name.%04d.exr", "0001", 4)),
    ("render_0001.exr", ("render_%04d.exr", "0001", 4)),
    ("img.1.exr", ("img.%01d.exr", "1", 1)),
    ("a/b/img.%04d.exr", ("a/b/img.%04d.exr", "%04d", 4)),
    ("img.%06d.png", ("img.%06d.png", "%06d", 6)),
    ("img.%d.exr", ("img.%d.exr", "%d", 1)),
    (r"C:\x\img.####.exr", ("C:/x/img.%04d.exr", "####", 4)),
    ("img.##.exr", ("img.%02d.exr", "##", 2)),
    ("out.####", ("out.%04d", "####", 4)),
    ("a/##/img.####.exr", ("a/##/img.%04d.exr", "####", 4)),
])
def test_parse_frame_pattern_sequences(seq, path, expected):
    assert seq.parse_frame_pattern(path) == expected


@pytest.mark.parametrize("path", [
    "shot_v002.exr", "out_v2.exr", "frame1.png", "img2024.jpg",
    "img.exr", "v01/img.exr", "img.0001", "a/##/img.exr", "a/2024/img.exr",
])
def test_parse_frame_pattern_single_files(seq, path):
    pattern, frame_spec, padding = seq.parse_frame_pattern(path)
    assert frame_spec is None and padding == 0
    assert pattern == path.replace("\\", "/")


def test_percent_d_padding_matches_expansion(seq):
    pattern, _, padding = seq.parse_frame_pattern("img.%d.exr")
    assert padding == 1
    assert seq.expand_frame_pattern(pattern, 7, padding) == "img.7.exr"
    assert seq.expand_frame_pattern("img.%04d.exr", 7) == "img.0007.exr"
    assert seq.expand_frame_pattern("img.####.exr", 7) == "img.0007.exr"


# ------------------------------------------------------------ detect_sequence

def test_detect_sequence_mixed_padding_dedup(seq, tmp_root):
    d = os.path.join(tmp_root, "seq_mixed")
    _touch(d, "beauty.0001.exr", "beauty.0002.exr", "beauty.1.exr")
    for probe in ("beauty.####.exr", "beauty.0001.exr", "beauty.%04d.exr"):
        pattern, frames, padding = seq.detect_sequence(os.path.join(d, probe))
        assert frames == [1, 2], probe
        assert padding == 4 and pattern.endswith("beauty.%04d.exr")
    # padding 1 accepts any digit count, still unique and sorted
    _, frames, padding = seq.detect_sequence(os.path.join(d, "beauty.%d.exr"))
    assert (frames, padding) == ([1, 2], 1)
    _, frames, padding = seq.detect_sequence(os.path.join(d, "beauty.1.exr"))
    assert (frames, padding) == ([1, 2], 1)


def test_detect_sequence_rejects_non_digit_frame_field(seq, tmp_root):
    d = os.path.join(tmp_root, "seq_junk")
    _touch(d, "sh.0003.exr", "sh.final.exr", "sh.0004_v2.exr", "sh.00005.exr")
    _, frames, _ = seq.detect_sequence(os.path.join(d, "sh.####.exr"))
    assert frames == [3]


def test_detect_sequence_single_file_and_missing(seq, tmp_root):
    d = os.path.join(tmp_root, "seq_single")
    _touch(d, "shot_v002.exr")
    path = os.path.join(d, "shot_v002.exr").replace("\\", "/")
    assert seq.detect_sequence(path) == (path, [0], 0)
    assert seq.detect_sequence(os.path.join(d, "zz.%04d.exr"))[1] == []


# ------------------------------------------------------------ auto_detect

def test_auto_detect_no_separator_needs_non_digit_stem(seq, tmp_root):
    d = os.path.join(tmp_root, "seq_auto")
    _touch(d, "img2024.jpg", "img2025.jpg", "img2026.jpg", "img20245.jpg")
    # stem ends in a digit: "img2024" + "5" is not a frame
    assert seq.auto_detect_sequence(os.path.join(d, "img2024.jpg")) is None
    # stem ends in a letter: the no-separator siblings are still accepted
    result = seq.auto_detect_sequence(os.path.join(d, "img.jpg"))
    assert result is not None
    _, frames, padding = result
    assert frames == [2024, 2025, 2026] and padding == 4
    # a digit-ending stem still detects "." and "_" separated siblings
    _touch(d, "img2024.0001.jpg", "img2024.0002.jpg")
    pattern, frames, padding = seq.auto_detect_sequence(os.path.join(d, "img2024.jpg"))
    assert pattern.endswith("img2024.%04d.jpg") and frames == [1, 2]


def test_auto_detect_separator_siblings(seq, tmp_root):
    d = os.path.join(tmp_root, "seq_auto2")
    _touch(d, "b_0001.exr", "b_0002.exr", "b_0003.exr", "b.0001.exr")
    pattern, frames, padding = seq.auto_detect_sequence(os.path.join(d, "b.exr"))
    assert pattern.endswith("b_%04d.exr") and frames == [1, 2, 3] and padding == 4
