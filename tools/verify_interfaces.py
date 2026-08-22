"""Check every node's V1 interface against tools/interface_manifest.json.

Usage (from the repo root):
    python tools/verify_interfaces.py              # all modules
    python tools/verify_interfaces.py merge_nodes  # one or more modules

Needs only torch + numpy (ComfyUI's env) - no running ComfyUI, no internet.
Exit 0 = full parity with the reference manifest; exit 1 = mismatches printed.

Option lists stored as markers in the manifest (see tools/dump_manifest.py)
are resolved against the LIVE environment - the active OCIO config and the
luts/ folder - so parity holds under any $OCIO.
"""
import importlib
import importlib.util
import json
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "tools", "interface_manifest.json")

# Stub ComfyUI's runtime module so node files import outside ComfyUI
fp = types.ModuleType("folder_paths")
for _f in ("get_temp_directory", "get_output_directory",
           "get_input_directory", "get_user_directory"):
    setattr(fp, _f, lambda: ".")
sys.modules.setdefault("folder_paths", fp)

# Register a package shell so relative imports inside modules resolve,
# WITHOUT executing __init__.py (so partially rebuilt packs still check).
spec = importlib.util.spec_from_file_location(
    "nuke_nodes", os.path.join(REPO, "__init__.py"),
    submodule_search_locations=[REPO])
pkg = importlib.util.module_from_spec(spec)
sys.modules["nuke_nodes"] = pkg

with open(MANIFEST, encoding="utf-8") as f:
    REF = json.load(f)

CFG_KEYS = ["default", "min", "max", "step", "multiline",
            "placeholder", "label_on", "label_off", "round"]
modules = sys.argv[1:] or sorted({m["module"] for m in REF.values()})
failures = []
checked = 0
_markers = None


def resolve_marker(marker):
    """Turn a manifest marker into the live option list it stands for."""
    global _markers
    if _markers is None:
        ocio_config = importlib.import_module("nuke_nodes.ocio_config")
        colorspace_nodes = importlib.import_module("nuke_nodes.colorspace_nodes")
        names = ocio_config.colorspace_names()
        roles = [f"role:{r}" for r in ocio_config.role_names()]
        _markers = {
            "<<ocio:colorspaces+roles>>": names + roles,
            "<<ocio:raw+colorspaces>>": ["raw"] + names,
            "<<ocio:displays>>": ocio_config.display_names(),
            "<<ocio:views>>": ocio_config.view_names(),
            "<<luts-folder>>": colorspace_nodes._get_available_ocio_luts(),
        }
    return _markers.get(marker, marker)


def norm_inputs(section):
    out = {}
    for name, defn in section.items():
        t = defn[0] if isinstance(defn, tuple) else defn
        cfg = defn[1] if isinstance(defn, tuple) and len(defn) > 1 else {}
        out[name] = {"type": t if isinstance(t, str) else list(t), "config": cfg}
    return out


for mod_name in modules:
    module = importlib.import_module(f"nuke_nodes.{mod_name}")
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
    display = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    if mappings is None:
        failures.append(f"{mod_name}: no NODE_CLASS_MAPPINGS")
        continue
    expected = {nid for nid, m in REF.items() if m["module"] == mod_name}
    for nid in expected - set(mappings):
        failures.append(f"{mod_name}: node {nid!r} missing from NODE_CLASS_MAPPINGS")
    for nid, cls in mappings.items():
        checked += 1
        ref = REF.get(nid)
        if ref is None:
            failures.append(f"{mod_name}: unexpected node id {nid!r}")
            continue
        if display.get(nid) != ref["display_name"]:
            failures.append(f"{nid}: display name {ref['display_name']!r} != {display.get(nid)!r}")
        for attr, key in (("CATEGORY", "category"), ("FUNCTION", "function")):
            if getattr(cls, attr, None) != ref[key]:
                failures.append(f"{nid}: {attr} {ref[key]!r} != {getattr(cls, attr, None)!r}")
        if list(getattr(cls, "RETURN_TYPES", ())) != ref["return_types"]:
            failures.append(f"{nid}: RETURN_TYPES {ref['return_types']} != {list(getattr(cls, 'RETURN_TYPES', ()))}")
        got_names = list(getattr(cls, "RETURN_NAMES", ())) or None
        if got_names != ref["return_names"]:
            failures.append(f"{nid}: RETURN_NAMES {ref['return_names']} != {got_names}")
        if bool(getattr(cls, "OUTPUT_NODE", False)) != ref["output_node"]:
            failures.append(f"{nid}: OUTPUT_NODE should be {ref['output_node']}")
        if hasattr(cls, "IS_CHANGED") != ref["has_is_changed"]:
            failures.append(f"{nid}: IS_CHANGED presence should be {ref['has_is_changed']}")
        if hasattr(cls, "VALIDATE_INPUTS") != ref["has_validate"]:
            failures.append(f"{nid}: VALIDATE_INPUTS presence should be {ref['has_validate']}")
        it = cls.INPUT_TYPES()
        for section, key in (("required", "inputs_required"), ("optional", "inputs_optional")):
            got = norm_inputs(it.get(section, {}))
            want = ref[key]
            if list(got) != list(want):
                failures.append(f"{nid}: {section} input order {list(want)} != {list(got)}")
            for name, w in want.items():
                g = got.get(name)
                if g is None:
                    continue
                want_type = w["type"]
                if isinstance(want_type, str) and want_type.startswith("<<"):
                    want_type = resolve_marker(want_type)
                if g["type"] != want_type:
                    shown = w["type"] if isinstance(w["type"], str) else w["type"]
                    failures.append(f"{nid}: input {name!r} type {shown!r} != {g['type']!r}")
                for k in CFG_KEYS:
                    if k in w["config"] and g["config"].get(k) != w["config"][k]:
                        failures.append(f"{nid}: input {name!r} {k}={w['config'][k]!r} != {g['config'].get(k)!r}")
        if dict(it.get("hidden", {})) != ref["inputs_hidden"]:
            failures.append(f"{nid}: hidden inputs {ref['inputs_hidden']} != {dict(it.get('hidden', {}))}")

print(f"checked {checked} nodes in {len(modules)} module(s)")
if failures:
    print(f"{len(failures)} INTERFACE MISMATCHES:")
    for f_ in failures:
        print(" -", f_)
    sys.exit(1)
print("INTERFACE PARITY: OK")
