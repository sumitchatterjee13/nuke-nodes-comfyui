#!/usr/bin/env python3
"""
Version Information Demo for Nuke Nodes ComfyUI Package
This script demonstrates how to access and use the version information.
"""

import sys
import json

def demo_version_info():
    """Demonstrate version information access"""
    print("🔍 Nuke Nodes - Version Information Demo")
    print("=" * 50)
    
    try:
        # Method 1: Import from version module directly
        print("\n📦 Method 1: Direct import from version module")
        from version import get_full_version, check_compatibility
        
        # Check compatibility first
        try:
            check_compatibility()
            print("✅ Environment compatibility check passed")
        except RuntimeError as e:
            print(f"❌ Compatibility check failed: {e}")
            return False
        
        # Get full version info
        version_info = get_full_version()
        print(f"📦 Package: {version_info['title']}")
        print(f"🏷️  Version: {version_info['version']}")
        print(f"📝 Description: {version_info['description']}")
        print(f"👨‍💻 Author: {version_info['author']}")
        print(f"📄 License: {version_info['license']}")
        
        # Method 2: Import from main package (would work in ComfyUI)
        print("\n📦 Method 2: Import from main package")
        # This would normally work: from . import __version__, get_version
        # But for demo purposes, we'll import directly
        from version import __version__, get_version
        print(f"🏷️  Version (simple): {__version__}")
        print(f"🏷️  Version (function): {get_version()}")
        
        # Method 3: Show JSON format (useful for APIs)
        print("\n📦 Method 3: JSON format")
        print(json.dumps(version_info, indent=2))
        
        # Method 4: Show available merge operations
        print("\n🔧 Available Merge Operations:")
        try:
            # Import the merge node to show available operations
            sys.path.append('.')
            from merge_nodes import NukeMerge
            operations = NukeMerge.INPUT_TYPES()['required']['operation'][0]
            for i, op in enumerate(operations, 1):
                print(f"  {i:2d}. {op}")
            print(f"\nTotal merge operations: {len(operations)}")
        except Exception as e:
            print(f"⚠️  Could not load merge operations: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing version information: {e}")
        return False


def demo_comfyui_integration():
    """Show how version info would be used in ComfyUI context"""
    print("\n🎨 ComfyUI Integration Example")
    print("=" * 40)
    
    print("In ComfyUI, you could access version info like this:")
    print("""
# In your ComfyUI workflow or custom script:
import custom_nodes.nuke_nodes as nuke_nodes

# Get version information
print(f"Nuke Nodes version: {nuke_nodes.__version__}")
print(f"Description: {nuke_nodes.__description__}")

# Check if nodes are available
if "NukeMerge" in nuke_nodes.NODE_CLASS_MAPPINGS:
    print("✅ NukeMerge node is available")
    merge_node = nuke_nodes.NODE_CLASS_MAPPINGS["NukeMerge"]
    operations = merge_node.INPUT_TYPES()['required']['operation'][0]
    print(f"Available operations: {len(operations)}")

# Get full version details for debugging/support
version_details = nuke_nodes.get_full_version()
print(f"Full version info: {version_details}")
    """)


if __name__ == "__main__":
    print("Starting version information demonstration...\n")
    
    success = demo_version_info()
    
    if success:
        demo_comfyui_integration()
        print("\n✅ Version demo completed successfully!")
    else:
        print("\n❌ Version demo failed!")
        sys.exit(1)
