"""NukeReadMultiPass ``colorspace``: convert light passes, leave data passes alone."""
import os

import pytest
import torch

from conftest import result_of


@pytest.fixture(scope="module")
def aces_config(pack):
    from nuke_nodes import image_io, ocio_config
    if not image_io.OIIO_AVAILABLE:
        pytest.skip("OpenImageIO not installed")
    if not ocio_config.OCIO_AVAILABLE or ocio_config.get_config() is None:
        pytest.skip("OpenColorIO not installed")
    if "ACES2065-1" not in ocio_config.colorspace_names():
        pytest.skip("active OCIO config has no ACES2065-1 colourspace")
    return ocio_config


@pytest.fixture
def bundle():
    """Working-space light passes plus one data pass, pre-rounded to half."""
    torch.manual_seed(21)
    return {
        "RGBA": (torch.rand(16, 16, 4) * 4.0).half().float(),
        "diffuse": (torch.rand(16, 16, 3) * 2.0).half().float(),
        "depth": (torch.rand(16, 16, 1) * 100.0).half().float(),
    }


def _write_bundle(pack, tmp_root, name, passes, colorspace):
    cls = pack.NODE_CLASS_MAPPINGS["NukeWrite"]
    out = cls().write_image(
        file_path=os.path.join(tmp_root, name), image=None, channels="all_channels",
        passes=passes, file_type="exr", bit_depth="16f", compression="zips",
        colorspace=colorspace, show_preview=False)
    return result_of(out)[1].splitlines()[0]


def _read_bundle(run, path, colorspace):
    out = result_of(run("NukeReadMultiPass", file_path=path, load_as_sequence=False,
                        print_pass_list=False, colorspace=colorspace))
    return out[0], out[1]


def test_multipass_colorspace_offers_raw_auto_config(pack, aces_config):
    optional = pack.NODE_CLASS_MAPPINGS["NukeReadMultiPass"].INPUT_TYPES()["optional"]
    options, config = optional["colorspace"]
    assert options[:2] == ["raw", "auto"]
    assert config["default"] == "raw"


def test_multipass_auto_round_trips_light_passes_and_keeps_data(pack, run, aces_config,
                                                                tmp_root, bundle):
    path = _write_bundle(pack, tmp_root, "mp_ap0.exr", bundle, colorspace="ACES2065-1")

    raw, _ = _read_bundle(run, path, "raw")
    assert not torch.allclose(raw["RGBA"], bundle["RGBA"], atol=1e-3), \
        "sanity: the file holds AP0 pixels, so a raw read differs from the working-space input"

    auto, beauty = _read_bundle(run, path, "auto")
    named, _ = _read_bundle(run, path, "ACES2065-1")
    for name in ("RGBA", "diffuse"):
        assert torch.allclose(auto[name], bundle[name], atol=2e-3), name
        assert torch.equal(auto[name], named[name]), name
    assert torch.equal(auto["depth"], raw["depth"]), "data passes must not be converted"
    assert torch.allclose(beauty[0, ..., :3], bundle["RGBA"][..., :3], atol=2e-3)


def test_multipass_raw_is_untouched(pack, run, aces_config, tmp_root, bundle):
    path = _write_bundle(pack, tmp_root, "mp_raw.exr", bundle, colorspace="raw")
    raw, _ = _read_bundle(run, path, "raw")
    for name, tensor in bundle.items():
        assert torch.equal(raw[name], tensor), name
