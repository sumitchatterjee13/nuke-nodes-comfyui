# Installation Guide

## Requirements

- **ComfyUI** (latest version recommended) - provides PyTorch
- **Python 3.8 or higher** (usually satisfied by ComfyUI)
- **NumPy** (usually included with PyTorch)
- **OpenCV** (may need separate installation)

> **Note**: PyTorch should NOT be installed separately as it's provided by ComfyUI. Installing a different PyTorch version could break ComfyUI compatibility.

## Installation Methods

### Method 1: Git Clone (Recommended)

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/nuke-nodes-comfyui.git nuke-nodes
```

3. Install dependencies:
```bash
cd nuke-nodes
pip install -r requirements.txt
```

4. Restart ComfyUI

### Method 2: Manual Download

1. Download the repository as a ZIP file from GitHub
2. Extract to `ComfyUI/custom_nodes/nuke-nodes`
3. Install dependencies:
```bash
cd ComfyUI/custom_nodes/nuke-nodes
pip install -r requirements.txt
```
4. Restart ComfyUI

### Method 3: ComfyUI Manager

If you have ComfyUI Manager installed:

1. Open ComfyUI Manager
2. Search for "Nuke Nodes"
3. Click Install
4. Restart ComfyUI

## Verifying Installation

After installation and restart, you should see the new nodes in the ComfyUI node menu under the "Nuke" category:

- Nuke/Merge (NukeMerge, NukeMix)
- Nuke/Color (NukeGrade, NukeColorCorrect, NukeLevels)
- Nuke/Transform (NukeTransform, NukeCornerPin, NukeCrop)
- Nuke/Filter (NukeBlur, NukeMotionBlur, NukeDefocus)

## Troubleshooting

### Common Issues

**1. Nodes not appearing in menu**
- Ensure the directory is named correctly
- Check that `__init__.py` is present
- Restart ComfyUI completely
- Check console for error messages

**2. Import errors**
```
ImportError: No module named 'torch'
```
- Install PyTorch: `pip install torch`
- Ensure you're using the correct Python environment

**3. CUDA/GPU issues**
- Nodes should work on both CPU and GPU
- If GPU errors occur, try switching to CPU mode temporarily

**4. Memory errors**
- Reduce image sizes for testing
- Lower quality settings on blur nodes
- Close other applications to free memory

### Getting Help

If you encounter issues:

1. Check the [Issues](https://github.com/yourusername/nuke-nodes-comfyui/issues) page
2. Search for existing solutions
3. Create a new issue with:
   - ComfyUI version
   - Python version
   - Error messages
   - Steps to reproduce

## Updating

To update to the latest version:

```bash
cd ComfyUI/custom_nodes/nuke-nodes
git pull origin main
pip install -r requirements.txt --upgrade
```

Then restart ComfyUI.

## Uninstalling

To remove the nodes:

1. Delete the `nuke-nodes` directory from `ComfyUI/custom_nodes/`
2. Restart ComfyUI

The nodes will no longer appear in the menu.
