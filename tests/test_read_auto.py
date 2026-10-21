"""NukeRead ``colorspace=auto``: detect the file's colourspace from its header."""
import os

import numpy as np
import pytest
import torch

from conftest import result_of


@pytest.fixture(scope="module")
def oiio(pack):
    from nuke_nodes import image_io
    if not image_io.OIIO_AVAILABLE:
        pytest.skip("OpenImageIO not installed")
    return image_io.oiio


@pytest.fixture(scope="module")
def aces_config(pack):
    from nuke_nodes import ocio_config
    if not ocio_config.OCIO_AVAILABLE or ocio_config.get_config() is None:
        pytest.skip("OpenColorIO not installed")
    names = ocio_config.colorspace_names()
    for needed in ("ACES2065-1", "sRGB Encoded Rec.709 (sRGB)"):
        if needed not in names:
            pytest.skip(f"active OCIO config has no {needed!r}")
    return ocio_config


@pytest.fixture
def plate():
    torch.manual_seed(11)
    img = torch.rand(1, 24, 24, 3) * 6.0 - 0.25
    return img.half().float()


def _write(pack, tmp_root, name, image, **kw):
    cls = pack.NODE_CLASS_MAPPINGS["NukeWrite"]
    args = dict(file_path=os.path.join(tmp_root, name), image=image, channels="rgb",
                file_type="exr", bit_depth="16f", compression="zips",
                colorspace="raw", show_preview=False)
    args.update(kw)
    paths = result_of(cls().write_image(**args))[1].splitlines()
    assert len(paths) == 1, paths
    return paths[0]


def _read(run, path, colorspace):
    return result_of(run("NukeRead", file_path=path, frame=1, load_as_sequence=False,
                         colorspace=colorspace, show_preview=False))[0]


# ------------------------------------------------------------ interface

def test_read_colorspace_offers_auto_after_raw(pack, aces_config):
    options = pack.NODE_CLASS_MAPPINGS["NukeRead"].INPUT_TYPES()["optional"]["colorspace"][0]
    assert options[:2] == ["raw", "auto"]
    write_options = pack.NODE_CLASS_MAPPINGS["NukeWrite"].INPUT_TYPES()["optional"]["colorspace"][0]
    assert "auto" not in write_options, "auto is a Read-only option"


# ------------------------------------------------------------ hint resolution

@pytest.mark.parametrize("hint, expected", [
    ("lin_ap0_scene", "ACES2065-1"),                        # interop id
    ("ACES2065-1", "ACES2065-1"),                           # config name
    ("srgb_rec709_scene", "sRGB Encoded Rec.709 (sRGB)"),   # OIIO's 8-bit default
    ("bogus-space", None),
    ("", None),
])
def test_resolve_colorspace_hint(aces_config, hint, expected):
    assert aces_config.resolve_colorspace_hint(hint) == expected


def test_read_colorspace_hint_from_header(pack, oiio, aces_config, tmp_root, plate):
    from nuke_nodes import image_io
    tagged = _write(pack, tmp_root, "hint_tagged.exr", plate, colorspace="ACES2065-1")
    untagged = _write(pack, tmp_root, "hint_untagged.exr", plate)
    # OIIO >= 3.1 reports the interop id, older versions the written name
    assert image_io.read_colorspace_hint(tagged) in ("lin_ap0_scene", "ACES2065-1")
    assert image_io.read_colorspace_hint(untagged) is None


# ------------------------------------------------------------ Read behaviour

def test_auto_reads_tagged_exr_like_the_named_colorspace(pack, run, oiio, aces_config,
                                                        tmp_root, plate):
    path = _write(pack, tmp_root, "auto_ap0.exr", plate, colorspace="ACES2065-1")
    named = _read(run, path, "ACES2065-1")
    auto = _read(run, path, "auto")
    assert not torch.allclose(named, _read(run, path, "raw")), "sanity: AP0 -> ACEScg changes pixels"
    assert torch.allclose(auto, named, atol=1e-6)


def test_auto_on_untagged_exr_passes_pixels_through(pack, run, oiio, aces_config, tmp_root, plate):
    path = _write(pack, tmp_root, "auto_untagged.exr", plate)
    auto = _read(run, path, "auto")
    assert np.array_equal(auto.numpy(), plate.numpy())


def test_auto_on_png_assumes_srgb(pack, run, oiio, aces_config, tmp_root):
    torch.manual_seed(5)
    img = torch.rand(1, 16, 16, 3)
    path = _write(pack, tmp_root, "auto_8bit.png", img, file_type="png", bit_depth="8")
    auto = _read(run, path, "auto")
    srgb = _read(run, path, "sRGB Encoded Rec.709 (sRGB)")
    assert not torch.allclose(srgb, _read(run, path, "raw")), "sanity: sRGB -> ACEScg changes pixels"
    assert torch.equal(auto, srgb)
