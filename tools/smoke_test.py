"""Standalone functional smoke test for the nuke-nodes pack.

Usage (from the repo root):  python tools/smoke_test.py

Needs torch + numpy (ComfyUI's env). OpenImageIO, OpenColorIO and OpenCV are
the pack's declared dependencies; the test reports which are active and
skips what a missing one gates. No running ComfyUI, no internet.
Exit 0 = all checks passed.  (The pytest suite in tests/ covers the same
ground in more depth - this script is the quick, dependency-free gate.)
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="nuke_nodes_smoke_")

fp = types.ModuleType("folder_paths")
for _f in ("get_temp_directory", "get_output_directory",
           "get_input_directory", "get_user_directory"):
    setattr(fp, _f, lambda: TMP)
sys.modules["folder_paths"] = fp

spec = importlib.util.spec_from_file_location(
    "nuke_nodes", os.path.join(REPO, "__init__.py"),
    submodule_search_locations=[REPO])
mod = importlib.util.module_from_spec(spec)
sys.modules["nuke_nodes"] = mod
spec.loader.exec_module(mod)

import torch  # noqa: E402

with open(os.path.join(REPO, "tools", "interface_manifest.json"), encoding="utf-8") as f:
    REF = json.load(f)
N = mod.NODE_CLASS_MAPPINGS
PASSED = 0


def ok(cond, msg):
    global PASSED
    if not cond:
        print(f"FAIL: {msg}")
        shutil.rmtree(TMP, ignore_errors=True)
        sys.exit(1)
    PASSED += 1
    print(f"  ok: {msg}")


def run(node_id, **overrides):
    """Call a node's FUNCTION with manifest defaults, overridden by kwargs."""
    ref = REF[node_id]
    kwargs = {}
    for section in ("inputs_required", "inputs_optional"):
        for name, d in ref[section].items():
            if "default" in d["config"]:
                kwargs[name] = d["config"]["default"]
            elif isinstance(d["type"], list):
                kwargs[name] = d["type"][0]
    kwargs.update(overrides)
    return getattr(N[node_id](), ref["function"])(**kwargs)


def result_of(out):
    return out["result"] if isinstance(out, dict) else out


ok(len(N) == 29, f"29 nodes registered (got {len(N)})")
ok(len(mod.NODE_DISPLAY_NAME_MAPPINGS) == 29, "29 display names")
ok(mod.WEB_DIRECTORY == "./web", "WEB_DIRECTORY == './web'")

torch.manual_seed(0)
img = torch.rand(2, 64, 64, 3)
zeros = torch.zeros_like(img)
half = torch.zeros(64, 64)
half[:, 32:] = 1.0

out = result_of(run("NukeMerge", A=img, B=zeros, operation="over", mix=1.0))
ok(out[0].shape == img.shape, "NukeMerge 'over' keeps shape")
d0 = result_of(run("NukeDissolve", A=img, B=zeros, which=0.0))[0]
d1 = result_of(run("NukeDissolve", A=img, B=zeros, which=1.0))[0]
ok(torch.equal(d0, img) and torch.equal(d1, zeros), "NukeDissolve which=0 -> A, which=1 -> B")
out = result_of(run("NukeConstant", width=128, height=64))
ok(out[0].shape[1:3] == (64, 128), "NukeConstant 128x64")

for node in ("NukeGrade", "NukeLevels", "NukeColorCorrect"):
    out = result_of(run(node, image=img))
    ok(torch.allclose(out[0], img, atol=1e-4), f"{node} neutral ~= identity")
out = result_of(run("NukeGrade", image=img, gain=2.0, mask=half))
ok(torch.allclose(out[0][:, :, :32], img[:, :, :32], atol=1e-6), "NukeGrade MASK limits effect")
out = result_of(run("NukeExposure", image=img, stops=1.0, clamp_output=False))
ok(out[0].mean() > img.mean(), "NukeExposure +1 stop brightens")

out = result_of(run("NukeBlur", image=img, size_x=5.0, size_y=5.0, mask=half))
ok(torch.equal(out[0][:, :, :32], img[:, :, :32]) and not torch.equal(out[0][:, :, 32:], img[:, :, 32:]),
   "NukeBlur blurs only the masked half")
out = result_of(run("NukeMotionBlur", image=img, distance=10.0))
ok(out[0].shape == img.shape, "NukeMotionBlur batch=2 runs")
out = result_of(run("NukeDefocus", image=img, defocus=3.0, depth_map=torch.rand(2, 64, 64)))
ok(out[0].shape == img.shape, "NukeDefocus with MASK depth_map runs")

out = result_of(run("NukeTransform", image=img))
ok(torch.equal(out[0][..., :3], img) and torch.all(out[0][..., 3] == 1), "NukeTransform identity bit-exact")
sq = torch.rand(1, 32, 32, 3)
out = result_of(run("NukeTransform", image=sq, rotate=90.0, filter="impulse"))
ok(torch.equal(out[0][..., :3], torch.rot90(sq, k=1, dims=(1, 2))), "NukeTransform rotate=90 is CCW")
out = result_of(run("NukeCornerPin", image=img))
ok(torch.equal(out[0][..., :3], img), "NukeCornerPin identity bit-exact")
out = result_of(run("NukeReformat", image=img, format="HD_1080", resize_type="fit"))
ok(out[0].shape[1:3] == (1080, 1920), "NukeReformat fit -> 1920x1080")

out = result_of(run("NukeViewer", image=img, channel="red"))
ok(torch.allclose(out[0][..., 0], out[0][..., 1]), "NukeViewer red -> mono")
rgba = torch.cat([img, torch.ones_like(img[..., :1])], dim=-1)
out = result_of(run("NukeViewer", image=rgba, channel="rgba", show_overlay=True, mask=half))
ok(out[0].shape == rgba.shape, "NukeViewer rgba + MASK overlay")
out = result_of(run("NukeChannelShuffle", image=img, red_from="green"))
ok(torch.allclose(out[0][..., 0], img[..., 1]), "NukeChannelShuffle R<-G")
out = result_of(run("NukeRamp", width=128, height=64))
ok(out[0].shape[1:3] == (64, 128), "NukeRamp 128x64")
out = result_of(run("NukeColorBars", width=128, height=64))
ok(out[0].shape[1:3] == (64, 128), "NukeColorBars 128x64")

batch = torch.rand(15, 8, 8, 3)
out = result_of(run("NukeFrameHold", image=batch, first_frame=1, increment=5, frame_start=1))
ok(torch.equal(out[0][7], batch[5]) and torch.equal(out[0][12], batch[10]), "NukeFrameHold increment=5 pattern")

from nuke_nodes import image_io, ocio_config  # noqa: E402
print(f"  libs: OIIO={image_io.OIIO_AVAILABLE} OCIO={ocio_config.OCIO_AVAILABLE} "
      f"({ocio_config.config_source()})")

if image_io.OIIO_AVAILABLE:
    seq = torch.rand(3, 32, 32, 3)
    base = os.path.join(TMP, "shot.exr")
    common = dict(image=seq, channels="rgb", file_type="exr", bit_depth="16f",
                  compression="zip", show_preview=False)
    paths = result_of(run("NukeWrite", file_path=base, **common))[1].splitlines()
    ok(len(paths) == 3 and paths[0].endswith("shot.0001.exr"), "NukeWrite 3-frame exr sequence")
    back = result_of(run("NukeRead", file_path=base, frame=1, frame_mode="all",
                         first_frame=1, last_frame=3, show_preview=False))[0]
    ok(back.shape[0] == 3, "NukeRead bare-path auto-detect loads 3 frames")
    ok(torch.allclose(back, seq, atol=2e-3), "read-back matches written")
    out = result_of(run("NukeWrite", file_path=base, **common))[1].splitlines()
    ok(all("shot_1." in p for p in out), "overwrite=False versions whole sequence")
    out = result_of(run("NukeWrite", file_path=base, overwrite=True, **common))[1].splitlines()
    ok(out == paths and not [f for f in os.listdir(TMP) if "__tmp" in f], "overwrite=True reuses exact paths, no temp files")
    out = result_of(run("NukeReadMultiPass", file_path=paths[0], load_as_sequence=False, print_pass_list=False))
    ok(isinstance(out[0], dict) and out[1].shape[-1] == 3, "NukeReadMultiPass reads EXR")
    if ocio_config.OCIO_AVAILABLE:
        cs = "sRGB Encoded Rec.709 (sRGB)"
        base2 = os.path.join(TMP, "srgb.exr")
        result_of(run("NukeWrite", file_path=base2, colorspace=cs, **common))
        back = result_of(run("NukeRead", file_path=base2, frame=1, frame_mode="all", first_frame=1,
                             last_frame=3, colorspace=cs, show_preview=False))[0]
        ok(torch.allclose(back, seq, atol=1e-2), "Write->Read OCIO colorspace round trip")
else:
    print("  skip: I/O checks (OpenImageIO missing)")

if ocio_config.OCIO_AVAILABLE:
    out = result_of(run("NukeOCIOInfo"))
    ok(isinstance(out[0], str) and "luts" in out[0].lower(), "NukeOCIOInfo reports config + LUT folder")
    grey = torch.full((1, 4, 4, 3), 0.18)
    out = result_of(run("NukeOCIOColorSpace", image=grey, in_colorspace="ACEScg",
                        out_colorspace="sRGB Encoded Rec.709 (sRGB)"))
    ok(abs(out[0][0, 0, 0, 0].item() - 0.4614) < 2e-3, "OCIO ACEScg->sRGB = 0.4614")
    out = result_of(run("NukeOCIODisplay", image=img))
    ok(out[0].shape == img.shape and torch.isfinite(out[0]).all(), "NukeOCIODisplay default display/view")
else:
    print("  skip: OCIO checks (opencolorio missing)")

shutil.rmtree(TMP, ignore_errors=True)
print(f"ALL SMOKE CHECKS PASSED ({PASSED} checks)")
