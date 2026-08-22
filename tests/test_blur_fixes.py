"""Regression tests for the 2.0.1 blur_nodes.py fixes (NukeBlur, NukeMotionBlur,
NukeDefocus). Run from the repo root:  python -m pytest tests
"""
import torch
import pytest

from conftest import result_of


def _impulse(h, w, r, c, channels=3):
    img = torch.zeros(1, h, w, channels)
    img[0, r, c] = 1.0
    return img


def _axis_variances(response_hw, centre):
    """Second moments of a (non-negative) 2-D response along columns (x) and rows (y)."""
    h, w = response_hw.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    weights = response_hw / response_hw.sum()
    var_x = (weights * (xs - centre) ** 2).sum().item()
    var_y = (weights * (ys - centre) ** 2).sum().item()
    return var_x, var_y


# ------------------------------------------------------------- NukeMotionBlur

def test_motion_blur_samples_1_is_bit_exact_identity(run):
    torch.manual_seed(0)
    img = torch.rand(2, 6, 8, 3)  # non-power-of-two dims on purpose
    out = result_of(run("NukeMotionBlur", image=img, samples=1))[0]
    assert torch.equal(out, img)
    out = result_of(run("NukeMotionBlur", image=img, shutter=0.0))[0]
    assert torch.equal(out, img)


def test_motion_blur_distance_10_changes_image_and_keeps_batch_shape(run):
    torch.manual_seed(1)
    img = torch.rand(2, 16, 24, 3)
    out = result_of(run("NukeMotionBlur", image=img, distance=10.0))[0]
    assert out.shape == img.shape
    assert not torch.equal(out, img)


def test_motion_blur_shifts_are_pixel_aligned(run):
    imp = _impulse(64, 64, 32, 32)
    # samples 2, shutter 1, distance 8 -> whole-pixel shifts of +/-4
    out = result_of(run("NukeMotionBlur", image=imp, distance=8.0, angle=0.0,
                        samples=2, shutter=1.0))[0]
    row = out[0, 32, :, 0]
    assert torch.allclose(row[28], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(row[36], torch.tensor(0.5), atol=1e-6)
    assert row[32].abs() < 1e-6
    assert torch.allclose(out[0, :, :, 0].sum(), torch.tensor(1.0), atol=1e-5)
    # angle 90 moves the same taps along the column
    out = result_of(run("NukeMotionBlur", image=imp, distance=8.0, angle=90.0,
                        samples=2, shutter=1.0))[0]
    col = out[0, :, 32, 0]
    assert torch.allclose(col[28], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(col[36], torch.tensor(0.5), atol=1e-6)
    # a constant image is untouched in the interior (no half-pixel darkening)
    const = torch.full((1, 64, 64, 3), 0.5)
    out = result_of(run("NukeMotionBlur", image=const, distance=8.0))[0]
    assert torch.allclose(out[0, 4:-4, 4:-4], const[0, 4:-4, 4:-4], atol=1e-6)


def test_motion_blur_center_bias_weights_samples(run):
    imp = _impulse(64, 64, 32, 32)
    base = dict(image=imp, distance=8.0, angle=0.0, samples=3, shutter=1.0)
    neutral = result_of(run("NukeMotionBlur", **base))[0][0, 32, :, 0]
    assert torch.allclose(neutral[[28, 32, 36]], torch.tensor([1 / 3, 1 / 3, 1 / 3]), atol=1e-6)
    plus = result_of(run("NukeMotionBlur", center_bias=1.0, **base))[0][0, 32, :, 0]
    assert torch.allclose(plus[[28, 32, 36]], torch.tensor([0.25, 0.5, 0.25]), atol=1e-6)
    minus = result_of(run("NukeMotionBlur", center_bias=-1.0, **base))[0][0, 32, :, 0]
    assert torch.allclose(minus[[28, 32, 36]], torch.tensor([0.5, 0.0, 0.5]), atol=1e-6)


# ----------------------------------------------------------------- NukeDefocus

@pytest.mark.parametrize("method", ["gaussian", "disk", "hexagon"])
def test_defocus_aspect_ratio_widens_horizontally(run, method):
    imp = _impulse(17, 17, 8, 8)
    out = result_of(run("NukeDefocus", image=imp, defocus=3.0, aspect_ratio=3.0,
                        method=method))[0]
    var_x, var_y = _axis_variances(out[0, :, :, 0], 8)
    assert var_x > 3.0 * var_y, (method, var_x, var_y)
    assert var_x > 0.5


def test_defocus_hexagon_is_distinct_and_hexagonal(run):
    imp = _impulse(17, 17, 8, 8)
    hexa = result_of(run("NukeDefocus", image=imp, defocus=2.0, method="hexagon"))[0]
    gaus = result_of(run("NukeDefocus", image=imp, defocus=2.0, method="gaussian"))[0]
    disk = result_of(run("NukeDefocus", image=imp, defocus=2.0, method="disk"))[0]
    assert not torch.equal(hexa, gaus)
    assert not torch.equal(hexa, disk)
    assert not torch.equal(gaus, disk)
    # energy preserved, flat-top: the centre row is wider than the rows above/below
    resp = hexa[0, :, :, 0]
    assert torch.allclose(resp.sum(), torch.tensor(1.0), atol=1e-5)
    assert (resp[8] > 0).sum() > (resp[7] > 0).sum()
    assert torch.allclose(resp, resp.flip(0), atol=1e-6)
    assert torch.allclose(resp, resp.flip(1), atol=1e-6)


def test_defocus_unknown_method_falls_back_to_disk(run):
    imp = _impulse(17, 17, 8, 8)
    out = result_of(run("NukeDefocus", image=imp, defocus=2.0, method="bogus"))[0]
    ref = result_of(run("NukeDefocus", image=imp, defocus=2.0, method="disk"))[0]
    assert torch.equal(out, ref)


@pytest.mark.parametrize("quality", ["low", "medium", "high"])
@pytest.mark.parametrize("method", ["gaussian", "disk", "hexagon"])
def test_defocus_never_noop_for_positive_defocus(run, method, quality):
    imp = _impulse(17, 17, 8, 8)
    out = result_of(run("NukeDefocus", image=imp, defocus=1.0, method=method,
                        quality=quality))[0]
    assert not torch.equal(out, imp)
    assert out[0, 8, 8, 0] < 1.0


def test_defocus_depth_map_is_uniform_max_deviation(run):
    torch.manual_seed(2)
    img = torch.rand(1, 32, 32, 3)
    depth = torch.zeros(32, 32)
    depth[:, :16] = 1.0
    a = result_of(run("NukeDefocus", image=img, defocus=2.0, depth_map=depth,
                      focus_distance=0.5))[0]
    b = result_of(run("NukeDefocus", image=img, defocus=1.0))[0]
    assert torch.equal(a, b)
    flat = result_of(run("NukeDefocus", image=img, defocus=2.0,
                         depth_map=torch.full((32, 32), 0.5)))[0]
    assert torch.equal(flat, img)


# ------------------------------------------- small images / reflect-pad limit

@pytest.mark.parametrize("crop", [True, False])
@pytest.mark.parametrize("filter", ["gaussian", "box", "triangle", "quadratic"])
def test_blur_size_8_on_4x4_does_not_raise(run, filter, crop):
    torch.manual_seed(3)
    img = torch.rand(1, 4, 4, 3)
    out = result_of(run("NukeBlur", image=img, size_x=8.0, size_y=8.0, filter=filter,
                        quality="high", crop=crop))[0]
    assert out.shape == img.shape
    assert torch.isfinite(out).all()
    if not crop:
        const = torch.full((1, 4, 4, 3), 0.4)
        out = result_of(run("NukeBlur", image=const, size_x=8.0, size_y=8.0,
                            filter=filter, quality="high", crop=False))[0]
        assert torch.allclose(out, const, atol=1e-6)


@pytest.mark.parametrize("method", ["gaussian", "disk", "hexagon"])
def test_defocus_large_on_4x4_does_not_raise(run, method):
    torch.manual_seed(4)
    img = torch.rand(1, 4, 4, 3)
    out = result_of(run("NukeDefocus", image=img, defocus=8.0, method=method,
                        quality="high"))[0]
    assert out.shape == img.shape
    assert torch.isfinite(out).all()


def test_motion_blur_large_on_4x4_does_not_raise(run):
    img = torch.rand(1, 4, 4, 3)
    out = result_of(run("NukeMotionBlur", image=img, distance=50.0, shutter=1.0))[0]
    assert out.shape == img.shape


# --------------------------------------------------------------- NukeBlur crop

def test_blur_crop_true_fades_edges_to_black(run):
    white = torch.ones(1, 16, 16, 3)
    out = result_of(run("NukeBlur", image=white, size_x=2.0, size_y=2.0))[0]  # crop default True
    assert out[0, 0, 0, 0] < 0.9          # corner darkened
    assert out[0, 0, 8, 0] < 0.95         # edge midpoint darkened
    assert torch.allclose(out[0, 8, 8], torch.ones(3), atol=1e-6)  # interior white
    # box is exact: 3/5 along an edge, 9/25 in the corner for size 1
    out = result_of(run("NukeBlur", image=white, size_x=1.0, size_y=1.0, filter="box"))[0]
    assert torch.allclose(out[0, 0, 8, 0], torch.tensor(0.6), atol=1e-6)
    assert torch.allclose(out[0, 0, 0, 0], torch.tensor(0.36), atol=1e-6)


def test_blur_crop_false_keeps_edges(run):
    white = torch.ones(1, 16, 16, 3)
    for filter in ("gaussian", "box", "triangle", "quadratic"):
        out = result_of(run("NukeBlur", image=white, size_x=2.0, size_y=2.0,
                            filter=filter, crop=False))[0]
        assert torch.allclose(out, white, atol=1e-6), filter
    # the documented row: reflect padding keeps the left edge at 1
    row = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0.0]).view(1, 1, 12, 1).repeat(1, 1, 1, 3)
    out = result_of(run("NukeBlur", image=row, size_x=1.0, size_y=0.0, filter="box", crop=False))[0]
    assert torch.allclose(out[0, 0, :, 0],
                          torch.tensor([1, 1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0.0]), atol=1e-6)
    out = result_of(run("NukeBlur", image=row, size_x=1.0, size_y=0.0, filter="box", crop=True))[0]
    assert torch.allclose(out[0, 0, :, 0],
                          torch.tensor([0.6, 0.8, 1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0.0]), atol=1e-6)


# ----------------------------------------------------------- NukeBlur mask/mix

def test_blur_mask_and_mix_unchanged(run):
    torch.manual_seed(5)
    img = torch.rand(1, 16, 24, 4)
    mask = torch.zeros(16, 24)
    mask[:, :12] = 1.0
    out = result_of(run("NukeBlur", image=img, size_x=2.0, size_y=2.0, mask=mask))[0]
    assert torch.equal(out[:, :, 12:, :3], img[:, :, 12:, :3])   # unmasked RGB untouched
    assert torch.equal(out[..., 3], img[..., 3])                  # alpha never blurred
    assert not torch.equal(out[:, :, :12, :3], img[:, :, :12, :3])
    # mix
    assert torch.equal(result_of(run("NukeBlur", image=img, size_x=3.0, size_y=3.0, mix=0.0))[0], img)
    full = result_of(run("NukeBlur", image=img, size_x=3.0, size_y=3.0))[0]
    half = result_of(run("NukeBlur", image=img, size_x=3.0, size_y=3.0, mix=0.5))[0]
    assert torch.allclose(half, img + (full - img) * 0.5, atol=1e-6)
    # size 0 and unknown filter are exact identities
    assert torch.equal(result_of(run("NukeBlur", image=img, size_x=0.0, size_y=0.0))[0], img)
    assert torch.equal(result_of(run("NukeBlur", image=img, filter="nope"))[0], img)
