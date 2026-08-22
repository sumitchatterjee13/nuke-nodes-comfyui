"""NukeWrite token/extension handling, NukeRead missing frames, preview naming."""
import os
import re

import pytest
import torch

from conftest import result_of

COMMON = dict(channels="rgb", file_type="exr", bit_depth="16f",
              compression="zip", show_preview=False)


@pytest.fixture(scope="module")
def oiio_required(pack):
    from nuke_nodes import image_io
    if not image_io.OIIO_AVAILABLE:
        pytest.skip("OpenImageIO not installed")


@pytest.fixture
def footage():
    torch.manual_seed(1)
    return torch.rand(3, 48, 64, 3)


def _write(run, path, image, **overrides):
    kw = dict(COMMON, **overrides)
    return result_of(run("NukeWrite", file_path=path, image=image, **kw))[1].splitlines()


def _read(run, path, **overrides):
    kw = dict(frame=1, frame_mode="range", first_frame=1, last_frame=3,
              show_preview=False)
    kw.update(overrides)
    return result_of(run("NukeRead", file_path=path, **kw))[0]


# ------------------------------------------------------ _apply_file_type

@pytest.mark.parametrize("path, file_type, expected", [
    ("out.####", "exr", "out.####.exr"),
    ("out.%04d", "exr", "out.%04d.exr"),
    ("out_####", "exr", "out_####.exr"),
    ("out.####.png", "exr", "out.####.exr"),
    ("out.####.exr", "exr", "out.####.exr"),
    ("out.%04d_v2.exr", "exr", "out.%04d_v2.exr"),
    ("out.####_final", "png", "out.####_final.png"),
    ("out", "exr", "out.exr"),
    ("x.tif", "tiff", "x.tiff"),
    ("shot.EXR", "exr", "shot.EXR"),
    ("out_v2.exr", "exr", "out_v2.exr"),
])
def test_apply_file_type(pack, path, file_type, expected):
    from nuke_nodes.io_nodes import _apply_file_type
    assert _apply_file_type(path, file_type) == expected
    joined = os.path.join("some", "dir", path)
    assert _apply_file_type(joined, file_type) == os.path.join("some", "dir", expected)


# ------------------------------------------------------ NukeWrite tokens

@pytest.mark.parametrize("token_path, names", [
    ("out.####", ["out.0001.exr", "out.0002.exr", "out.0003.exr"]),
    ("out.%04d", ["out.0001.exr", "out.0002.exr", "out.0003.exr"]),
    ("out_####", ["out_0001.exr", "out_0002.exr", "out_0003.exr"]),
    ("out.####.png", ["out.0001.exr", "out.0002.exr", "out.0003.exr"]),
    ("out.%d.exr", ["out.1.exr", "out.2.exr", "out.3.exr"]),
])
def test_write_token_without_extension_keeps_token(run, tmp_root, oiio_required,
                                                   footage, token_path, names):
    d = os.path.join(tmp_root, "wtok", token_path.replace("%", "p").replace("#", "h"))
    paths = _write(run, os.path.join(d, token_path), footage)
    assert [os.path.basename(p) for p in paths] == names
    assert len(set(paths)) == 3
    assert sorted(os.listdir(d)) == sorted(names)
    back = _read(run, os.path.join(d, names[0]), load_as_sequence=True)
    assert back.shape == (3, 48, 64, 3) and torch.allclose(back, footage, atol=2e-3)


def test_write_version_tag_is_not_a_frame(run, tmp_root, oiio_required, footage):
    d = os.path.join(tmp_root, "wver")
    paths = _write(run, os.path.join(d, "out_v2.exr"), footage)
    assert [os.path.basename(p) for p in paths] == [
        "out_v2.0001.exr", "out_v2.0002.exr", "out_v2.0003.exr"]


def test_read_version_tag_single_file_with_sequence_on(run, tmp_root, oiio_required, footage):
    d = os.path.join(tmp_root, "rver")
    written = _write(run, os.path.join(d, "plate.exr"), footage[:1])
    single = os.path.join(d, "plate_v2.exr")
    os.replace(written[0], single)
    out = _read(run, single, load_as_sequence=True, frame=7, frame_mode="single")
    assert out.shape == (1, 48, 64, 3)
    assert torch.allclose(out[0], footage[0], atol=2e-3)


def test_write_paths_use_native_separator(run, tmp_root, oiio_required, footage):
    d = os.path.join(tmp_root, "wsep")
    for name in ("tok.####.exr", "plain.exr"):
        mixed = os.path.join(d, name).replace("\\", "/")
        paths = _write(run, mixed, footage)
        assert len(paths) == 3
        for p in paths:
            assert p == os.path.normpath(p)
            assert os.path.isfile(p)
            assert p.startswith(os.path.normpath(d))


# ------------------------------------------------------ NukeRead missing frames

@pytest.fixture
def gapped(run, tmp_root, oiio_required, footage):
    """48x64 footage at out.0001-0003.exr with the LEADING frame deleted."""
    d = os.path.join(tmp_root, "rlead")
    _write(run, os.path.join(d, "out.####.exr"), footage, overwrite=True)
    os.remove(os.path.join(d, "out.0001.exr"))
    return os.path.join(d, "out.####.exr")


def test_missing_leading_frame_black(run, gapped, footage):
    out = _read(run, gapped, missing_frames="black")
    assert out.shape == (3, 48, 64, 3)
    assert torch.all(out[0] == 0)
    assert torch.allclose(out[1:], footage[1:], atol=2e-3)


def test_missing_leading_frame_hold(run, gapped, footage):
    out = _read(run, gapped, missing_frames="hold")
    assert out.shape == (3, 48, 64, 3)
    assert torch.all(out[0] == 0)          # nothing to hold yet -> black
    assert torch.allclose(out[1:], footage[1:], atol=2e-3)


def test_missing_leading_frame_nearest(run, gapped, footage):
    out = _read(run, gapped, missing_frames="nearest")
    assert out.shape == (3, 48, 64, 3)
    assert torch.allclose(out[0], footage[1], atol=2e-3)


def test_missing_frame_error_raises(run, gapped):
    with pytest.raises(RuntimeError) as excinfo:
        _read(run, gapped, missing_frames="error")
    assert "out.0001.exr" in str(excinfo.value)


def test_interior_gap_hold_and_black(run, tmp_root, oiio_required, footage):
    d = os.path.join(tmp_root, "rgap")
    _write(run, os.path.join(d, "out.####.exr"), footage)
    os.remove(os.path.join(d, "out.0002.exr"))
    pattern = os.path.join(d, "out.####.exr")
    held = _read(run, pattern, missing_frames="hold", last_frame=4)
    assert held.shape == (4, 48, 64, 3)
    assert torch.allclose(held[1], held[0]) and torch.allclose(held[3], held[2])
    black = _read(run, pattern, missing_frames="black", last_frame=4)
    assert torch.all(black[1] == 0) and torch.all(black[3] == 0)
    assert torch.allclose(black[2], footage[2], atol=2e-3)
    with pytest.raises(RuntimeError, match="out.0002.exr"):
        _read(run, pattern, missing_frames="error")


def test_nothing_loads_falls_back_to_512(run, tmp_root, oiio_required):
    pattern = os.path.join(tmp_root, "rnone", "out.####.exr")
    out = _read(run, pattern, missing_frames="black", last_frame=2)
    assert out.shape == (2, 512, 512, 3) and torch.all(out == 0)


# ------------------------------------------------------ preview naming

def test_preview_filenames_never_collide(pack, oiio_required):
    from nuke_nodes.preview import create_preview_images
    names = []
    for _ in range(2):
        t = torch.rand(2, 8, 8, 3)
        records = create_preview_images(t)
        assert len(records) == 2
        names.extend(r["filename"] for r in records)
        del t
    assert len(set(names)) == 4
    for name in names:
        assert re.fullmatch(r"nuke_preview_[0-9a-f]{32}_\d+\.png", name)
