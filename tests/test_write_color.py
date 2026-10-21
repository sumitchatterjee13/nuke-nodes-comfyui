"""NukeWrite colour tagging, lossless defaults and the ACES2065-1 round trip."""
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
def aces_available(pack):
    from nuke_nodes import ocio_config
    if not ocio_config.OCIO_AVAILABLE or ocio_config.get_config() is None:
        pytest.skip("OpenColorIO not installed")
    if "ACES2065-1" not in ocio_config.colorspace_names():
        pytest.skip("active OCIO config has no ACES2065-1 colourspace")


@pytest.fixture
def plate():
    """Scene-linear plate with negatives and values above 1, pre-rounded to half."""
    torch.manual_seed(7)
    img = torch.rand(1, 32, 32, 3) * 8.0 - 0.5
    return img.half().float()


def _header(oiio, path):
    inp = oiio.ImageInput.open(path)
    spec = inp.spec()
    inp.close()
    attribs = spec.extra_attribs
    return {attribs[i].name: attribs[i].value for i in range(len(attribs))}


def _colour_tags(header):
    return {k: v for k, v in header.items()
            if k.lower() in ("oiio:colorspace", "colorinteropid")}


def _write_with_node_defaults(pack, tmp_root, name, image, **overrides):
    """Write using the node's OWN declared defaults (not the manifest's)."""
    cls = pack.NODE_CLASS_MAPPINGS["NukeWrite"]
    optional = cls.INPUT_TYPES()["optional"]
    kwargs = {k: v[1]["default"] for k, v in optional.items()
              if len(v) > 1 and "default" in v[1]}
    kwargs.update(image=image, file_path=os.path.join(tmp_root, name),
                  show_preview=False)
    kwargs.update(overrides)
    paths = result_of(cls().write_image(**kwargs))[1].splitlines()
    assert len(paths) == 1, paths
    return paths[0]


# ------------------------------------------------------------ defaults

def test_default_compression_is_lossless(pack, oiio, tmp_root, plate):
    from nuke_nodes import image_io
    path = _write_with_node_defaults(pack, tmp_root, "defaults_plate.exr", plate,
                                     channels="rgb", colorspace="raw")
    back = image_io.read_image(path)
    assert np.array_equal(back, plate[0].numpy()), \
        "the default EXR compression must reproduce the input bit-for-bit"


# ------------------------------------------------------------ colour tags

def test_raw_exr_carries_no_colour_tag(pack, oiio, tmp_root, plate):
    path = _write_with_node_defaults(pack, tmp_root, "raw_plate.exr", plate,
                                     colorspace="raw")
    tags = _colour_tags(_header(oiio, path))
    assert tags == {}, f"a raw write must not claim a colourspace, got {tags}"


def test_raw_multipass_exr_carries_no_colour_tag(pack, oiio, tmp_root):
    torch.manual_seed(3)
    passes = {"RGBA": torch.rand(16, 16, 4), "depth": torch.rand(16, 16, 1)}
    cls = pack.NODE_CLASS_MAPPINGS["NukeWrite"]
    out = cls().write_image(
        file_path=os.path.join(tmp_root, "raw_passes.exr"), image=None,
        channels="all_channels", passes=passes, file_type="exr",
        bit_depth="16f", compression="zips", colorspace="raw",
        show_preview=False)
    path = result_of(out)[1].splitlines()[0]
    tags = _colour_tags(_header(oiio, path))
    assert tags == {}, f"a raw multi-pass write must not claim a colourspace, got {tags}"


def test_named_colorspace_exr_is_tagged(pack, oiio, aces_available, tmp_root, plate):
    path = _write_with_node_defaults(pack, tmp_root, "ap0_plate.exr", plate,
                                     colorspace="ACES2065-1")
    tags = set(_colour_tags(_header(oiio, path)).values())
    # OIIO >= 3.1 stores the interop id, older versions the config name
    assert tags & {"ACES2065-1", "lin_ap0_scene"}, tags


# ------------------------------------------------------------ round trip

def test_aces2065_round_trip_is_lossless(pack, run, oiio, aces_available, tmp_root, plate):
    from nuke_nodes import image_io
    # The source plate is written losslessly so only the colour round trip is measured.
    src = _write_with_node_defaults(pack, tmp_root, "src_ap0.exr", plate,
                                    channels="rgb", colorspace="raw",
                                    compression="zips")
    working = result_of(run("NukeRead", file_path=src, frame=1, load_as_sequence=False,
                            colorspace="ACES2065-1", show_preview=False))[0]
    assert not torch.allclose(working, plate), "AP0 -> ACEScg must change the pixels"
    out = _write_with_node_defaults(pack, tmp_root, "rt_ap0.exr", working,
                                    channels="rgb", colorspace="ACES2065-1",
                                    bit_depth="16f", compression="zips")
    back = image_io.read_image(out)
    err = np.abs(back - plate[0].numpy()).max()
    assert err < 1e-5, f"ACES2065-1 -> ACEScg -> ACES2065-1 drifted by {err}"
