# Pytest configuration - ignore the package __init__.py during test collection
# The package uses relative imports that only work within ComfyUI context
collect_ignore = ["__init__.py", "blur_nodes.py", "colorspace_nodes.py", "grade_nodes.py",
                  "io_nodes.py", "merge_nodes.py", "transform_nodes.py", "vectorfield_nodes.py",
                  "viewer_nodes.py", "utils.py", "version.py"]
