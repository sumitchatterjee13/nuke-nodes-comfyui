"""Standalone functional smoke test for the nuke-nodes pack.

Usage (from the repo root):  python tools/smoke_test.py

Needs torch + numpy (ComfyUI's env). cv2 / OpenImageIO / PyOpenColorIO are
optional; the test reports which are active and skips what they gate.
No running ComfyUI, no internet. Exit 0 = all checks passed.
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


ok(len(N) == 32, f"32 nodes registered (got {len(N)})")
ok(len(mod.NODE_DISPLAY_NAME_MAPPINGS) == 32, "32 display names")
ok(mod.WEB_DIRECTORY == "./web", "WEB_DIRECTORY == './web'")

img = torch.rand(2, 64, 64, 3)
zeros = torch.zeros_like(img)

out = result_of(run("NukeMerge", A=img, B=zeros, operation="over", mix=1.0))
ok(out[0].shape == img.shape, "NukeMerge 'over' keeps shape")
m0 = result_of(run("NukeMix", image_a=img, image_b=zeros, mix=0.0))[0]
m1 = result_of(run("NukeMix", image_a=img, image_b=zeros, mix=1.0))[0]
ok((torch.allclose(m0, img) and torch.allclose(m1, zeros)) or
   (torch.allclose(m0, zeros) and torch.allclose(m1, img)), "NukeMix endpoints return pure A / pure B")
out = result_of(run("NukeConstant", width=128, height=64))
ok(out[0].shape[1:3] == (64, 128), "NukeConstant 128x64")

out = result_of(run("NukeGrade", image=img))
ok(torch.allclose(out[0], img, atol=1e-5), "NukeGrade neutral ~= identity")
out = result_of(run("NukeLevels", image=img))
ok(torch.allclose(out[0], img, atol=1e-5), "NukeLevels neutral ~= identity")
out = result_of(run("NukeColorCorrect", image=img))
ok(torch.allclose(out[0], img, atol=1e-4), "NukeColorCorrect neutral ~= identity")
out = result_of(run("NukeExposure", image=img, stops=1.0, clamp_output=False))
ok(out[0].mean() > img.mean(), "NukeExposure +1 stop brightens")

out = result_of(run("NukeBlur", image=img, size_x=5.0, size_y=5.0))
ok(out[0].shape == img.shape and not torch.equal(out[0], img), "NukeBlur blurs")
out = result_of(run("NukeMotionBlur", image=img, distance=10.0))
ok(out[0].shape == img.shape, "NukeMotionBlur batch=2 runs")
out = result_of(run("NukeDefocus", image=img, defocus=3.0))
ok(out[0].shape == img.shape, "NukeDefocus runs")

out = result_of(run("NukeTransform", image=img, translate_x=10.0))
ok(out[0].shape[:3] == img.shape[:3], "NukeTransform batch=2 runs (outputs RGBA)")
out = result_of(run("NukeCornerPin", image=img))
ok(out[0].shape[:3] == img.shape[:3], "NukeCornerPin batch=2 runs")
out = result_of(run("NukeViewer", image=img, channel="red"))
ok(torch.allclose(out[0][..., 0], out[0][..., 1]), "NukeViewer red -> mono")
out = result_of(run("NukeChannelShuffle", image=img, red_from="green"))
ok(torch.allclose(out[0][..., 0], img[..., 1]), "NukeChannelShuffle R<-G")
out = result_of(run("NukeRamp", width=128, height=64))
ok(out[0].shape[1:3] == (64, 128), "NukeRamp 128x64")
out = result_of(run("NukeColorBars", width=128, height=64))
ok(out[0].shape[1:3] == (64, 128), "NukeColorBars 128x64")

batch = torch.rand(15, 8, 8, 3)
out = result_of(run("NukeFrameHold", image=batch, first_frame=1, increment=5, frame_start=1))
ok(torch.equal(out[0][7], batch[5]) and torch.equal(out[0][12], batch[10]),
   "NukeFrameHold increment=5 pattern")

from nuke_nodes import io_nodes  # noqa: E402
print(f"  optional libs: OIIO={io_nodes.OIIO_AVAILABLE} "
      f"CV2={io_nodes.CV2_AVAILABLE} PIL={io_nodes.PIL_AVAILABLE}")

if io_nodes.CV2_AVAILABLE:
    out = result_of(run("NukeReformat", image=img, format="HD_1080", resize_type="fit"))
    ok(out[0].shape[1:3] == (1080, 1920), "NukeReformat fit -> 1920x1080")
else:
    print("  skip: NukeReformat (cv2 missing)")

ext = "exr" if io_nodes.OIIO_AVAILABLE else "png"
seq = torch.rand(3, 32, 32, 3)
base = os.path.join(TMP, "shot." + ext)
out = result_of(run("NukeWrite", file_path=base, image=seq, channels="rgb", file_type=ext, compression="zip",
                    bit_depth="16f" if ext == "exr" else "8", show_preview=False))
paths = out[1].splitlines()
ok(len(paths) == 3 and paths[0].endswith("shot.0001." + ext), f"NukeWrite 3-frame {ext} sequence")
out = result_of(run("NukeRead", file_path=base, frame=1, frame_mode="all",
                    first_frame=1, last_frame=3, show_preview=False))
ok(out[0].shape[0] == 3, "NukeRead bare-path auto-detect loads 3 frames")
ok(torch.allclose(out[0], seq, atol=2e-3 if ext == "exr" else 2e-2), "read-back matches written")
out = result_of(run("NukeWrite", file_path=base, image=seq, channels="rgb", file_type=ext, compression="zip",
                    bit_depth="16f" if ext == "exr" else "8", show_preview=False))
ok(all("shot_1." in p for p in out[1].splitlines()), "overwrite=False versions whole sequence")
out = result_of(run("NukeWrite", file_path=base, image=seq, channels="rgb", file_type=ext, compression="zip",
                    bit_depth="16f" if ext == "exr" else "8", overwrite=True, show_preview=False))
ok(out[1].splitlines() == paths, "overwrite=True reuses exact paths")

if io_nodes.OIIO_AVAILABLE:
    out = result_of(run("NukeReadMultiPass", file_path=paths[0], load_as_sequence=False,
                        print_pass_list=False))
    ok(isinstance(out[0], dict) and out[1].shape[-1] == 3, "NukeReadMultiPass reads EXR")
else:
    print("  skip: multipass (OpenImageIO missing)")

try:
    import PyOpenColorIO  # noqa: F401
    out = result_of(run("NukeOCIOInfo"))
    ok(isinstance(out[0], str) and len(out[0]) > 20, "NukeOCIOInfo reports config")
    out = result_of(run("NukeOCIOColorSpace", image=img, in_colorspace="ACEScg",
                        out_colorspace="sRGB Encoded Rec.709 (sRGB)"))
    ok(out[0].shape == img.shape and not torch.equal(out[0], img), "OCIO ACEScg->sRGB transforms")
except ImportError:
    print("  skip: OCIO nodes (opencolorio missing)")

shutil.rmtree(TMP, ignore_errors=True)
print(f"ALL SMOKE CHECKS PASSED ({PASSED} checks)")
