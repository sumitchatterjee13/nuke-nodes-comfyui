"""pytest fixtures for nuke-nodes.

The pack is loaded the way ComfyUI loads it (as a package via its __init__.py)
with ComfyUI's ``folder_paths`` module stubbed, so the suite runs anywhere
torch + numpy are installed - no running ComfyUI needed.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def tmp_root():
    path = tempfile.mkdtemp(prefix="nuke_nodes_tests_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def pack(tmp_root):
    fp = types.ModuleType("folder_paths")
    for name in ("get_temp_directory", "get_output_directory",
                 "get_input_directory", "get_user_directory"):
        setattr(fp, name, lambda: tmp_root)
    sys.modules["folder_paths"] = fp
    spec = importlib.util.spec_from_file_location(
        "nuke_nodes", os.path.join(REPO, "__init__.py"),
        submodule_search_locations=[REPO])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nuke_nodes"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def manifest():
    with open(os.path.join(REPO, "tools", "interface_manifest.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def run(pack, manifest):
    """Call a node's FUNCTION with manifest defaults, overridden by kwargs.

    Returns the node's raw return value (a tuple, or a ui/result dict).
    """
    def _run(node_id, **overrides):
        ref = manifest[node_id]
        kwargs = {}
        for section in ("inputs_required", "inputs_optional"):
            for name, d in ref[section].items():
                if "default" in d["config"]:
                    kwargs[name] = d["config"]["default"]
                elif isinstance(d["type"], list):
                    kwargs[name] = d["type"][0]
        kwargs.update(overrides)
        cls = pack.NODE_CLASS_MAPPINGS[node_id]
        return getattr(cls(), ref["function"])(**kwargs)
    return _run


def result_of(out):
    """Normalise a node return value to its result tuple."""
    return out["result"] if isinstance(out, dict) else out
