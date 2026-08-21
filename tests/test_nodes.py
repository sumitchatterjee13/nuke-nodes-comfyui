"""Functional tests for the nuke-nodes pack (no running ComfyUI needed)."""
import os

import pytest
import torch

from conftest import result_of


@pytest.fixture
def img():
    torch.manual_seed(0)
    return torch.rand(2, 64, 64, 3)


@pytest.fixture
def half_mask():
    m = torch.zeros(64, 64)
    m[:, 32:] = 1.0
    return m


# --------------------------------------------------------------- merge family

def test_merge_over(run):
    a = torch.zeros(1, 8, 8, 4); a[..., 0] = 1.0; a[..., 3] = 0.5   # red, alpha .5
    b = torch.zeros(1, 8, 8, 4); b[..., 2] = 1.0; b[..., 3] = 1.0   # blue, opaque
    out = result_of(run("NukeMerge", A=a, B=b, operation="over", mix=1.0))[0]
    assert torch.allclose(out[0, 0, 0], torch.tensor([0.5, 0.0, 0.5, 1.0]), atol=1e-5)


def test_dissolve_endpoints(run, img):
    zeros = torch.zeros_like(img)
    assert torch.equal(result_of(run("NukeDissolve", A=img, B=zeros, which=0.0))[0], img)
    assert torch.equal(result_of(run("NukeDissolve", A=img, B=zeros, which=1.0))[0], zeros)


def test_keymix_mask_split(run, img, half_mask):
    zeros = torch.zeros_like(img)
    out = result_of(run("NukeKeymix", A=img, B=zeros, mask=half_mask, invert_mask=False, mix=1.0))[0]
    assert torch.allclose(out[:, :, :32, :3], zeros[:, :, :32])
    assert torch.allclose(out[:, :, 32:, :3], img[:, :, 32:], atol=1e-6)


def test_constant_dims(run):
    out = result_of(run("NukeConstant", width=128, height=64))[0]
    assert out.shape[1:3] == (64, 128)


# --------------------------------------------------------------- colour / grade

@pytest.mark.parametrize("node", ["NukeGrade", "NukeLevels", "NukeColorCorrect"])
def test_neutral_is_identity(run, img, node):
    out = result_of(run(node, image=img))[0]
    assert torch.allclose(out, img, atol=1e-4)


def test_grade_mask_limits_effect(run, img, half_mask):
    out = result_of(run("NukeGrade", image=img, gain=2.0, mask=half_mask))[0]
    assert torch.allclose(out[:, :, :32], img[:, :, :32], atol=1e-6)
    assert not torch.allclose(out[:, :, 32:], img[:, :, 32:])


def test_exposure(run, img):
    out = result_of(run("NukeExposure", image=img, stops=1.0, clamp_output=False))[0]
    assert out.mean() > img.mean()
    out = result_of(run("NukeExposure", image=img, stops_g_offset=1.0, clamp_output=False))[0]
    assert torch.allclose(out[..., 0], img[..., 0], atol=1e-5)      # only G changed
    assert not torch.allclose(out[..., 1], img[..., 1])


# --------------------------------------------------------------- filter

def test_blur_with_mask(run, img, half_mask):
    out = result_of(run("NukeBlur", image=img, size_x=5.0, size_y=5.0, mask=half_mask))[0]
    assert torch.equal(out[:, :, :32], img[:, :, :32])
    assert not torch.equal(out[:, :, 32:], img[:, :, 32:])


def test_motion_blur_and_defocus_batch(run, img):
    assert result_of(run("NukeMotionBlur", image=img, distance=10.0))[0].shape == img.shape
    depth = torch.rand(2, 64, 64)
    assert result_of(run("NukeDefocus", image=img, defocus=3.0, depth_map=depth))[0].shape == img.shape


# --------------------------------------------------------------- transform

def test_transform_identity_bit_exact(run, img):
    out = result_of(run("NukeTransform", image=img))[0]
    assert torch.equal(out[..., :3], img)
    assert torch.all(out[..., 3] == 1.0)


def test_transform_rotate_is_ccw(run):
    sq = torch.rand(1, 32, 32, 3)
    out = result_of(run("NukeTransform", image=sq, rotate=90.0, filter="impulse"))[0][..., :3]
    assert torch.equal(out, torch.rot90(sq, k=1, dims=(1, 2)))


def test_cornerpin_identity_bit_exact(run, img):
    out = result_of(run("NukeCornerPin", image=img))[0]
    assert torch.equal(out[..., :3], img)


def test_reformat_fit(run, img):
    out = result_of(run("NukeReformat", image=img, format="HD_1080", resize_type="fit"))[0]
    assert out.shape[1:3] == (1080, 1920)


def test_framehold_pattern(run):
    batch = torch.rand(15, 8, 8, 3)
    out = result_of(run("NukeFrameHold", image=batch, first_frame=1, increment=5, frame_start=1))[0]
    assert torch.equal(out[7], batch[5]) and torch.equal(out[12], batch[10])


# --------------------------------------------------------------- viewer

def test_viewer_channels_and_overlay(run, img, half_mask):
    out = result_of(run("NukeViewer", image=img, channel="red"))[0]
    assert torch.allclose(out[..., 0], out[..., 1]) and torch.allclose(out[..., 1], out[..., 2])
    rgba = torch.cat([img, torch.ones_like(img[..., :1])], dim=-1)
    out = result_of(run("NukeViewer", image=rgba, channel="rgba", show_overlay=True, mask=half_mask))[0]
    assert out.shape == rgba.shape and torch.all(out[..., 3] == 1.0)


def test_channel_shuffle(run, img):
    out = result_of(run("NukeChannelShuffle", image=img, red_from="green"))[0]
    assert torch.allclose(out[..., 0], img[..., 1])


# --------------------------------------------------------------- io

def test_write_read_roundtrip_and_overwrite(run, pack, tmp_root):
    from nuke_nodes import image_io
    if not image_io.OIIO_AVAILABLE:
        pytest.skip("OpenImageIO not installed")
    seq = torch.rand(3, 32, 32, 3)
    base = os.path.join(tmp_root, "shot.exr")
    common = dict(image=seq, channels="rgb", file_type="exr", bit_depth="16f",
                  compression="zip", show_preview=False)
    paths = result_of(run("NukeWrite", file_path=base, **common))[1].splitlines()
    assert len(paths) == 3 and paths[0].endswith("shot.0001.exr")
    back = result_of(run("NukeRead", file_path=base, frame=1, frame_mode="all",
                         first_frame=1, last_frame=3, show_preview=False))[0]
    assert back.shape[0] == 3 and torch.allclose(back, seq, atol=2e-3)
    versioned = result_of(run("NukeWrite", file_path=base, **common))[1].splitlines()
    assert all("shot_1." in p for p in versioned)
    again = result_of(run("NukeWrite", file_path=base, overwrite=True, **common))[1].splitlines()
    assert again == paths
    assert not [f for f in os.listdir(tmp_root) if "__tmp" in f]


def test_read_fingerprint_stable(run, pack, tmp_root):
    cls = pack.NODE_CLASS_MAPPINGS["NukeRead"]
    kw = dict(file_path=os.path.join(tmp_root, "shot.exr"), frame=1, load_as_sequence=True,
              first_frame=1, last_frame=3, frame_mode="all", missing_frames="black")
    assert cls.IS_CHANGED(**kw) == cls.IS_CHANGED(**kw)


def test_multipass_reads_exr(run, tmp_root):
    from nuke_nodes import image_io
    if not image_io.OIIO_AVAILABLE:
        pytest.skip("OpenImageIO not installed")
    path = os.path.join(tmp_root, "shot.0001.exr")
    if not os.path.exists(path):
        pytest.skip("needs the EXR written by the roundtrip test")
    out = result_of(run("NukeReadMultiPass", file_path=path, load_as_sequence=False,
                        print_pass_list=False))
    assert isinstance(out[0], dict) and out[1].shape[-1] == 3


# --------------------------------------------------------------- ocio

def test_ocio_nodes(run, pack, img):
    from nuke_nodes import ocio_config
    if not ocio_config.OCIO_AVAILABLE:
        pytest.skip("OpenColorIO not installed")
    info = result_of(run("NukeOCIOInfo"))[0]
    assert ocio_config.config_source().split("=")[0].split(" ")[0] in info
    grey = torch.full((1, 4, 4, 3), 0.18)
    out = result_of(run("NukeOCIOColorSpace", image=grey, in_colorspace="ACEScg",
                        out_colorspace="sRGB Encoded Rec.709 (sRGB)"))[0]
    assert abs(out[0, 0, 0, 0].item() - 0.4614) < 2e-3
    out = result_of(run("NukeOCIODisplay", image=img))[0]
    assert out.shape == img.shape and torch.isfinite(out).all()


def test_ocio_dropdowns_come_from_config(pack):
    from nuke_nodes import ocio_config
    it = pack.NODE_CLASS_MAPPINGS["NukeOCIOColorSpace"].INPUT_TYPES()
    options = it["required"]["in_colorspace"][0]
    assert options[: len(ocio_config.colorspace_names())] == ocio_config.colorspace_names()
    assert "config" not in it["required"]
