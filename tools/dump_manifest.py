"""Regenerate tools/interface_manifest.json from the current node definitions.

Run this ONLY after an intentional interface change (renamed node, new input,
changed default/options), then commit the manifest together with the code.
verify_interfaces.py compares the live pack against this file.

Environment-derived option lists are stored as MARKERS instead of literal
values, so the manifest is valid under any $OCIO config and any luts/ folder:
  <<ocio:colorspaces+roles>>  ocio_config.colorspace_names() + "role:<name>" entries
  <<ocio:raw+colorspaces>>    ["raw"] + ocio_config.colorspace_names()
  <<ocio:displays>>           ocio_config.display_names()
  <<ocio:views>>              ocio_config.view_names()
  <<luts-folder>>             colorspace_nodes._get_available_ocio_luts()
Tooltips and docstrings are not part of the manifest (they are documentation,
not interface).

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

from nuke_nodes import colorspace_nodes, ocio_config  # noqa: E402

CONFIG_KEYS = ("default", "min", "max", "step", "multiline", "placeholder",
               "label_on", "label_off", "round")


def marker_lists():
    names = ocio_config.colorspace_names()
    roles = [f"role:{r}" for r in ocio_config.role_names()]
    return {
        "<<ocio:colorspaces+roles>>": names + roles,
        "<<ocio:raw+colorspaces>>": ["raw"] + names,
        "<<ocio:displays>>": ocio_config.display_names(),
        "<<ocio:views>>": ocio_config.view_names(),
        "<<luts-folder>>": colorspace_nodes._get_available_ocio_luts(),
    }


MARKERS = marker_lists()


def norm(section):
    out = {}
    for name, defn in section.items():
        t = defn[0] if isinstance(defn, tuple) else defn
        cfg = defn[1] if isinstance(defn, tuple) and len(defn) > 1 else {}
        if isinstance(t, list):
            t = list(t)
            for marker, live in MARKERS.items():
                if t == live:
                    t = marker
                    break
        out[name] = {"type": t, "config": {k: v for k, v in cfg.items() if k in CONFIG_KEYS}}
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
        "inputs_required": norm(it.get("required", {})),
        "inputs_optional": norm(it.get("optional", {})),
        "inputs_hidden": dict(it.get("hidden", {})),
    }


def dump_compact(manifest):
    """One line per scalar field and per input - readable AND diff-friendly."""
    lines = ["{"]
    node_ids = list(manifest)
    for ni, node_id in enumerate(node_ids):
        m = manifest[node_id]
        lines.append(f" {json.dumps(node_id)}: {{")
        scalars = ["class", "module", "display_name", "category", "function",
                   "return_types", "return_names", "output_node", "has_is_changed", "has_validate"]
        for key in scalars:
            lines.append(f"  {json.dumps(key)}: {json.dumps(m[key], ensure_ascii=False)},")
        for sec in ("inputs_required", "inputs_optional"):
            items = list(m[sec].items())
            if not items:
                lines.append(f"  {json.dumps(sec)}: {{}},")
                continue
            lines.append(f"  {json.dumps(sec)}: {{")
            for ii, (name, d) in enumerate(items):
                comma = "," if ii < len(items) - 1 else ""
                lines.append(f"   {json.dumps(name)}: {json.dumps(d, ensure_ascii=False)}{comma}")
            lines.append("  },")
        lines.append(f"  \"inputs_hidden\": {json.dumps(m['inputs_hidden'], ensure_ascii=False)}")
        lines.append(" }" + ("," if ni < len(node_ids) - 1 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


text = dump_compact(manifest)
json.loads(text)  # self-check: the compact writer must produce valid JSON
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print(f"wrote {OUT}: {len(manifest)} nodes")
