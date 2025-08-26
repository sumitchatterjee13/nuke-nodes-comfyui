#!/usr/bin/env python3
"""
Setup script for Nuke Nodes for ComfyUI
Helps verify installation and dependencies
"""

import importlib
import os
import subprocess
import sys


def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version}")
        return True


def check_dependencies():
    """Check if required dependencies are installed"""
    dependencies = {"torch": "PyTorch", "numpy": "NumPy", "cv2": "OpenCV"}

    missing = []
    pytorch_missing = False

    for module, name in dependencies.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {name}: {version}")
        except ImportError:
            if module == "torch":
                print(f"❌ {name}: Not installed (CRITICAL - ComfyUI requires PyTorch)")
                pytorch_missing = True
            else:
                print(f"❌ {name}: Not installed")
                missing.append(module)

    if pytorch_missing:
        print(
            "⚠️  PyTorch is missing. This usually means ComfyUI is not properly installed."
        )
        print("   Please ensure ComfyUI is working before installing these nodes.")

    return missing


def install_dependencies(missing):
    """Install missing dependencies (excluding PyTorch which should come from ComfyUI)"""
    if not missing:
        return True

    print(f"\n📦 Installing missing dependencies: {', '.join(missing)}")

    # Map module names to pip package names (excluding torch which should come from ComfyUI)
    package_map = {"numpy": "numpy", "cv2": "opencv-python"}

    # Filter out torch from missing dependencies as it should come from ComfyUI
    safe_missing = [dep for dep in missing if dep != "torch"]

    if not safe_missing:
        print("✅ No additional dependencies needed")
        return True

    packages = [package_map.get(dep, dep) for dep in safe_missing]

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_comfyui_structure():
    """Check if we're in the correct ComfyUI structure"""
    current_dir = os.getcwd()

    # Check if we're in a custom_nodes directory
    if "custom_nodes" in current_dir:
        print("✅ Located in ComfyUI custom_nodes directory")
        return True
    else:
        print("⚠️  Not in ComfyUI custom_nodes directory")
        print("This package should be installed in ComfyUI/custom_nodes/")
        return False


def verify_node_files():
    """Verify all required node files exist"""
    required_files = [
        "__init__.py",
        "utils.py",
        "merge_nodes.py",
        "grade_nodes.py",
        "transform_nodes.py",
        "blur_nodes.py",
    ]

    missing_files = []

    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)

    return len(missing_files) == 0


def test_imports():
    """Test importing the node modules"""
    modules = ["merge_nodes", "grade_nodes", "transform_nodes", "blur_nodes"]

    success = True

    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ Import {module}")
        except Exception as e:
            print(f"❌ Import {module}: {e}")
            success = False

    return success


def main():
    """Main setup function"""
    print("🔧 Nuke Nodes for ComfyUI - Setup & Verification")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return False

    print("\n📋 Checking dependencies...")
    missing = check_dependencies()

    if missing:
        install = input(f"\nInstall missing dependencies? (y/n): ")
        if install.lower() == "y":
            if not install_dependencies(missing):
                return False
        else:
            print("⚠️  Some dependencies are missing. Nodes may not work properly.")

    print("\n📁 Checking file structure...")
    check_comfyui_structure()

    print("\n📄 Verifying node files...")
    if not verify_node_files():
        print("❌ Some required files are missing")
        return False

    print("\n🔍 Testing imports...")
    if not test_imports():
        print("❌ Some modules failed to import")
        return False

    print("\n✅ Setup complete! Restart ComfyUI to use the nodes.")
    print("\nNodes will appear in the menu under 'Nuke' category:")
    print("- Nuke/Merge")
    print("- Nuke/Color")
    print("- Nuke/Transform")
    print("- Nuke/Filter")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
