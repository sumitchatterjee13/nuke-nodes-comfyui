"""
Version information for Nuke Nodes ComfyUI package
"""

# Version information
__version__ = "1.0.0"
__version_info__ = tuple(int(part) for part in __version__.split("."))

# Package metadata
__title__ = "nuke-nodes"
__description__ = "ComfyUI custom nodes that replicate Nuke compositing functionality"
__author__ = "Nuke Nodes Team" 
__author_email__ = "contact@example.com"
__license__ = "MIT"
__url__ = "https://github.com/your-username/nuke-nodes"
__copyright__ = "Copyright 2024 Nuke Nodes Team"

# ComfyUI specific information
COMFYUI_MIN_VERSION = "0.1.0"
PYTHON_MIN_VERSION = "3.8"

# Node categories and count
NODE_CATEGORIES = [
    "Nuke/Merge",
    "Nuke/Color", 
    "Nuke/Transform",
    "Nuke/Filter",
    "Nuke/Viewer"
]

# Release information
RELEASE_DATE = "2024-09-21"
IS_STABLE = True
IS_BETA = False

def get_version():
    """Get the current version string"""
    return __version__

def get_version_info():
    """Get the version as a tuple of integers"""
    return __version_info__

def get_full_version():
    """Get full version information including metadata"""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "title": __title__,
        "description": __description__,
        "author": __author__,
        "license": __license__,
        "url": __url__,
        "release_date": RELEASE_DATE,
        "is_stable": IS_STABLE,
        "is_beta": IS_BETA,
        "node_categories": NODE_CATEGORIES
    }

def check_compatibility():
    """Check compatibility with current environment"""
    import sys
    
    # Check Python version
    python_version = sys.version_info
    min_python = tuple(int(x) for x in PYTHON_MIN_VERSION.split('.'))
    
    if python_version < min_python:
        raise RuntimeError(
            f"Python {PYTHON_MIN_VERSION} or higher is required. "
            f"Current version: {sys.version}"
        )
    
    return True

if __name__ == "__main__":
    # Print version information when run directly
    import json
    print(json.dumps(get_full_version(), indent=2))
