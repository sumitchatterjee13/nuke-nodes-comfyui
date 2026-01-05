"""
Pytest configuration for nuke-nodes tests.

This file ensures pytest can run tests without importing the main package,
which has relative imports that fail when run standalone.
"""

import sys
import os

# Ensure tests directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Do NOT add parent directory to avoid importing the package __init__.py
# which has relative imports that fail outside of ComfyUI context


def pytest_ignore_collect(collection_path, config):
    """Ignore the package __init__.py during collection."""
    if collection_path.name == "__init__.py" and collection_path.parent.name == "nuke-nodes":
        return True
    return False
