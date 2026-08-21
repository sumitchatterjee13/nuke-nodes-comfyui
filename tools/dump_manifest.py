"""Regenerate tools/interface_manifest.json from the current node definitions.

Run this ONLY after an intentional interface change (renamed node, new input,
changed default/options), then commit the manifest together with the code.
verify_interfaces.py compares the live pack against this file.

Usage (from the repo root):  python tools/dump_manifest.py
"""
import importlib.util
import json
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tools", "interface_manifest.json")

fp = types.ModuleType("folder_paths")
for _f in ("get_temp_directory", "get_output_directory",
           "get_input_directory", "get_user_directory"):
    setattr(fp, _f, lambda: ".")
sys.modules.setdefault("folder_paths", fp)

spec = importlib.util.spec_from_file_location(
    "nuke_nodes", os.path.join(REPO, "__init__.py"),
    submodule_search_locations=[REPO])
mod = importlib.util.module_from_spec(spec)
sys.modules["nuke_nodes"] = mod
spec.loader.exec_module(mod)


def norm(section):
    out = {}
    for name, defn in section.items():
        t = defn[0] if isinstance(defn, tuple) else defn
        cfg = defn[1] if isinstance(defn, tuple) and len(defn) > 1 else {}
        out[name] = {"type": t if isinstance(t, str) else list(t), "config": cfg}
    return out


manifest = {}
for node_id, cls in sorted(mod.NODE_CLASS_MAPPINGS.items()):
    it = cls.INPUT_TYPES()
    manifest[node_id] = {
        "class": cls.__name__,
        "module": cls.__module__.split(".")[-1],
        "display_name": mod.NODE_DISPLAY_NAME_MAPPINGS.get(node_id),
        "category": getattr(cls, "CATEGORY", None),
        "function": getattr(cls, "FUNCTION", None),
        "return_types": list(getattr(cls, "RETURN_TYPES", ())),
        "return_names": list(getattr(cls, "RETURN_NAMES", ())) or None,
        "output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
        "has_is_changed": hasattr(cls, "IS_CHANGED"),
        "has_validate": hasattr(cls, "VALIDATE_INPUTS"),
        "doc": (cls.__doc__ or "").strip(),
        "inputs_required": norm(it.get("required", {})),
        "inputs_optional": norm(it.get("optional", {})),
        "inputs_hidden": dict(it.get("hidden", {})),
    }

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1, default=str)
print(f"wrote {OUT}: {len(manifest)} nodes")
