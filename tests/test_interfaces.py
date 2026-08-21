"""Interface parity: every node must match tools/interface_manifest.json.

Run ``python tools/dump_manifest.py`` after an INTENTIONAL interface change
and commit the manifest together with the code.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_interface_parity():
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "verify_interfaces.py")],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_registration(pack):
    assert len(pack.NODE_CLASS_MAPPINGS) == len(pack.NODE_DISPLAY_NAME_MAPPINGS)
    assert set(pack.NODE_CLASS_MAPPINGS) == set(pack.NODE_DISPLAY_NAME_MAPPINGS)
    assert pack.WEB_DIRECTORY == "./web"
    for node_id, cls in pack.NODE_CLASS_MAPPINGS.items():
        assert node_id.startswith("Nuke"), node_id
        assert cls.CATEGORY.startswith("Nuke/"), node_id
        assert callable(getattr(cls, cls.FUNCTION)), node_id
