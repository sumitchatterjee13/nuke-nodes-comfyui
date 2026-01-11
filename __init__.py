"""
Nuke Nodes for ComfyUI
A collection of custom nodes that replicate Nuke compositing functionality
"""

import os

# Web directory for JavaScript extensions (preview widgets, etc.)
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

# Import version information
from .version import (
    __version__,
    __title__,
    __description__, 
    __author__,
    __license__,
    __url__,
    get_version,
    get_version_info,
    get_full_version
)

from .blur_nodes import *
from .colorspace_nodes import *
from .grade_nodes import *
from .io_nodes import *
from .merge_nodes import *
from .transform_nodes import *
from .vectorfield_nodes import *
from .viewer_nodes import *

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all node classes and their mappings
from .blur_nodes import NODE_CLASS_MAPPINGS as blur_mappings
from .blur_nodes import NODE_DISPLAY_NAME_MAPPINGS as blur_display_mappings
from .colorspace_nodes import NODE_CLASS_MAPPINGS as colorspace_mappings
from .colorspace_nodes import NODE_DISPLAY_NAME_MAPPINGS as colorspace_display_mappings
from .grade_nodes import NODE_CLASS_MAPPINGS as grade_mappings
from .grade_nodes import NODE_DISPLAY_NAME_MAPPINGS as grade_display_mappings
from .io_nodes import NODE_CLASS_MAPPINGS as io_mappings
from .io_nodes import NODE_DISPLAY_NAME_MAPPINGS as io_display_mappings
from .merge_nodes import NODE_CLASS_MAPPINGS as merge_mappings
from .merge_nodes import NODE_DISPLAY_NAME_MAPPINGS as merge_display_mappings
from .transform_nodes import NODE_CLASS_MAPPINGS as transform_mappings
from .transform_nodes import NODE_DISPLAY_NAME_MAPPINGS as transform_display_mappings
from .vectorfield_nodes import NODE_CLASS_MAPPINGS as vectorfield_mappings
from .vectorfield_nodes import NODE_DISPLAY_NAME_MAPPINGS as vectorfield_display_mappings
from .viewer_nodes import NODE_CLASS_MAPPINGS as viewer_mappings
from .viewer_nodes import NODE_DISPLAY_NAME_MAPPINGS as viewer_display_mappings

# Combine all mappings
NODE_CLASS_MAPPINGS.update(blur_mappings)
NODE_CLASS_MAPPINGS.update(colorspace_mappings)
NODE_CLASS_MAPPINGS.update(grade_mappings)
NODE_CLASS_MAPPINGS.update(io_mappings)
NODE_CLASS_MAPPINGS.update(merge_mappings)
NODE_CLASS_MAPPINGS.update(transform_mappings)
NODE_CLASS_MAPPINGS.update(vectorfield_mappings)
NODE_CLASS_MAPPINGS.update(viewer_mappings)

NODE_DISPLAY_NAME_MAPPINGS.update(blur_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(colorspace_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(grade_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(io_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(merge_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(transform_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(vectorfield_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(viewer_display_mappings)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    "__license__",
    "__url__",
    "get_version",
    "get_version_info",
    "get_full_version"
]
