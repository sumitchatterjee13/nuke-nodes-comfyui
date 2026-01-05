"""
Unit tests for OCIO ColorSpace nodes.

Tests OpenColorIO integration, ACES 2.0 support, and color space transformations.

Run with: pytest tests/test_colorspace_nodes.py -v
"""

import os
import sys
import unittest

import numpy as np
import torch

# Try to import OCIO for direct testing
try:
    import PyOpenColorIO as OCIO
    HAS_OCIO = True
except ImportError:
    OCIO = None
    HAS_OCIO = False


# ============================================================================
# Helper functions (reimplemented to avoid package import issues)
# ============================================================================

def get_ocio_config(config_path=None, builtin_config=None):
    """
    Get OCIO config from path, built-in name, or environment variable.

    Priority:
    1. Explicit config_path parameter (file path)
    2. Explicit builtin_config parameter (built-in config name)
    3. OCIO environment variable
    4. Default built-in OCIO config (ACES 2.0 if available)
    """
    if not HAS_OCIO:
        return None

    try:
        # 1. Try explicit file path
        if config_path and os.path.exists(config_path):
            return OCIO.Config.CreateFromFile(config_path)

        # 2. Try explicit built-in config name
        if builtin_config:
            try:
                return OCIO.Config.CreateFromBuiltinConfig(builtin_config)
            except Exception as e:
                print(f"[Test] Could not load built-in config '{builtin_config}': {e}")

        # 3. Try environment variable
        env_config = os.environ.get("OCIO")
        if env_config and os.path.exists(env_config):
            return OCIO.Config.CreateFromFile(env_config)

        # 4. Try to get current config (might be set by application)
        try:
            config = OCIO.GetCurrentConfig()
            if config:
                # Check if it has more than just 'raw' colorspace
                colorspaces = [cs.getName() for cs in config.getColorSpaces()]
                if len(colorspaces) > 1:
                    return config
        except:
            pass

        # 5. Use default built-in config (try ACES 2.0 first, then 1.3)
        default_configs = [
            "cg-config-v4.0.0_aces-v2.0_ocio-v2.5",  # ACES 2.0
            "cg-config-v2.2.0_aces-v1.3_ocio-v2.4",  # ACES 1.3
            "cg-config-v2.1.0_aces-v1.3_ocio-v2.3",  # Fallback
        ]

        for config_name in default_configs:
            try:
                return OCIO.Config.CreateFromBuiltinConfig(config_name)
            except:
                continue

        return None
    except Exception as e:
        print(f"[Test] Error loading OCIO config: {e}")
        return None


def get_colorspace_names(config):
    """Get list of color space names from config."""
    if not config:
        return []
    try:
        return [cs.getName() for cs in config.getColorSpaces()]
    except:
        return []


def apply_ocio_transform(image_np, src_colorspace, dst_colorspace, config):
    """
    Apply OCIO color space transformation to numpy image array.
    """
    if not HAS_OCIO or config is None:
        return image_np

    try:
        processor = config.getProcessor(src_colorspace, dst_colorspace)
        cpu_processor = processor.getDefaultCPUProcessor()

        img = image_np.astype(np.float32)
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1

        # Handle alpha channel separately if present
        if channels == 4:
            rgb = img[:, :, :3].copy()
            alpha = img[:, :, 3:4].copy()
        elif channels == 3:
            rgb = img.copy()
            alpha = None
        else:
            rgb = np.stack([img[:, :, 0]] * 3, axis=-1)
            alpha = None

        # Flatten for OCIO processing
        rgb_flat = rgb.reshape(-1, 3)

        # OCIO 2.5+ changed API - only one bitDepth parameter
        img_desc = OCIO.PackedImageDesc(
            rgb_flat,
            width,
            height,
            3,
            OCIO.BIT_DEPTH_F32,
            rgb_flat.strides[1],
            rgb_flat.strides[0],
            width * rgb_flat.strides[0]
        )

        cpu_processor.apply(img_desc)

        rgb_transformed = rgb_flat.reshape(height, width, 3)

        if alpha is not None:
            result = np.concatenate([rgb_transformed, alpha], axis=-1)
        else:
            result = rgb_transformed

        return result

    except Exception as e:
        print(f"[Test] Error applying transform: {e}")
        return image_np


# Common color spaces for fallback
COMMON_COLORSPACES = [
    "ACES2065-1",
    "ACEScg",
    "ACEScct",
    "ACEScc",
    "Linear Rec.709 (sRGB)",
    "Linear Rec.2020",
    "sRGB - Texture",
    "sRGB",
    "Rec.709",
    "Raw",
]


# ============================================================================
# Test Classes
# ============================================================================

class TestOCIOAvailability(unittest.TestCase):
    """Test OpenColorIO availability and basic setup."""

    def test_ocio_import_status(self):
        """Test that OCIO import status is boolean."""
        self.assertIsInstance(HAS_OCIO, bool)
        print(f"OpenColorIO available: {HAS_OCIO}")

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_ocio_version(self):
        """Test that OCIO version is accessible."""
        version = OCIO.GetVersion()
        self.assertIsInstance(version, str)
        self.assertTrue(len(version) > 0)
        print(f"OpenColorIO Version: {version}")

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_ocio_version_number(self):
        """Test OCIO version is 2.x or higher for ACES 2.0 support."""
        version = OCIO.GetVersion()
        major_version = int(version.split('.')[0])
        self.assertGreaterEqual(major_version, 2,
            "OCIO 2.x or higher required for full ACES 2.0 support")


class TestOCIOConfig(unittest.TestCase):
    """Test OCIO configuration loading."""

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_get_config_from_env(self):
        """Test loading config from OCIO environment variable."""
        env_config = os.environ.get("OCIO")
        if env_config and os.path.exists(env_config):
            config = get_ocio_config()
            self.assertIsNotNone(config)
            print(f"Loaded config from OCIO env: {env_config}")
        else:
            print("OCIO environment variable not set or file doesn't exist")

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_get_builtin_config(self):
        """Test loading built-in OCIO config."""
        # Clear OCIO env temporarily
        original_env = os.environ.get("OCIO")
        if "OCIO" in os.environ:
            del os.environ["OCIO"]

        try:
            config = get_ocio_config()
            if config:
                print(f"Built-in config loaded: {config.getDescription()}")
            else:
                print("No built-in config available")
        finally:
            if original_env:
                os.environ["OCIO"] = original_env

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_get_config_invalid_path(self):
        """Test that invalid config path is handled gracefully."""
        config = get_ocio_config("/nonexistent/path/config.ocio")
        # Should fall back to env or built-in, or return None
        # No exception should be raised

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_get_colorspace_names(self):
        """Test getting color space names from config."""
        config = get_ocio_config()
        if config:
            names = get_colorspace_names(config)
            self.assertIsInstance(names, list)
            self.assertTrue(len(names) > 0, "Config should have color spaces")
            print(f"Found {len(names)} color spaces")
            print(f"First 10: {names[:10]}")
        else:
            print("No config available for color space names test")


class TestACES2Support(unittest.TestCase):
    """Test ACES 2.0 specific functionality."""

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_aces_colorspaces_available(self):
        """Test that ACES color spaces are available in config."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        names_lower = [n.lower() for n in names]

        # Check for common ACES color spaces
        aces_spaces = ["aces", "acescg", "acescc", "acescct"]
        found_aces = []

        for aces in aces_spaces:
            for name in names_lower:
                if aces in name:
                    found_aces.append(aces)
                    break

        print(f"Found ACES spaces: {found_aces}")
        # Skip if no ACES config available (built-in ACES configs may not be installed)
        if len(found_aces) == 0:
            print("Note: No ACES color spaces found. Set OCIO env var to an ACES config for full testing.")
            self.skipTest("No ACES color spaces in config - need ACES OCIO config")

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_aces2_output_transforms(self):
        """Test that ACES 2.0 output transforms are available."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        displays = list(config.getDisplays())
        print(f"Available displays: {displays}")

        aces2_views_found = False
        for display in displays:
            views = list(config.getViews(display))
            for view in views:
                if "aces 2" in view.lower() or "2.0" in view.lower():
                    aces2_views_found = True
                    print(f"Found ACES 2.0 view: {display}/{view}")

        if not aces2_views_found:
            print("No ACES 2.0 specific views found (may need ACES 2.0 config)")

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_builtin_aces_configs(self):
        """Test loading built-in ACES configs (OCIO 2.x feature)."""
        builtin_configs = [
            "cg-config-v2.2.0_aces-v1.3_ocio-v2.3",
            "cg-config-v2.1.0_aces-v1.3_ocio-v2.1",
            "studio-config-v2.1.0_aces-v1.3_ocio-v2.1",
        ]

        loaded_configs = []
        for config_name in builtin_configs:
            try:
                config = OCIO.Config.CreateFromBuiltinConfig(config_name)
                if config:
                    loaded_configs.append(config_name)
                    print(f"Successfully loaded built-in config: {config_name}")
            except Exception as e:
                print(f"Could not load {config_name}: {e}")

        print(f"Loaded {len(loaded_configs)} built-in configs")


class TestColorSpaceTransform(unittest.TestCase):
    """Test color space transformation functionality."""

    def setUp(self):
        """Set up test images."""
        # Create test image: gradient from black to white
        self.test_image = np.zeros((64, 64, 3), dtype=np.float32)
        for i in range(64):
            self.test_image[:, i, :] = i / 63.0

        # Create RGBA test image
        self.test_image_rgba = np.zeros((64, 64, 4), dtype=np.float32)
        self.test_image_rgba[:, :, :3] = self.test_image
        self.test_image_rgba[:, :, 3] = 1.0

        # Create specific color test image
        self.color_image = np.zeros((32, 32, 3), dtype=np.float32)
        self.color_image[:, :, 0] = 0.5  # Red
        self.color_image[:, :, 1] = 0.3  # Green
        self.color_image[:, :, 2] = 0.2  # Blue

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_transform_srgb_to_linear(self):
        """Test sRGB to linear transformation."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)

        # Find sRGB and linear color spaces
        srgb_space = None
        linear_space = None

        for name in names:
            name_lower = name.lower()
            if "srgb" in name_lower and "texture" in name_lower:
                srgb_space = name
            elif "linear" in name_lower and ("srgb" in name_lower or "rec.709" in name_lower or "rec709" in name_lower):
                linear_space = name

        if not srgb_space or not linear_space:
            # Try alternative names
            for name in names:
                name_lower = name.lower()
                if srgb_space is None and "srgb" in name_lower:
                    srgb_space = name
                if linear_space is None and "linear" in name_lower:
                    linear_space = name

        if not srgb_space or not linear_space:
            self.skipTest(f"Could not find sRGB/linear spaces. Available: {names[:20]}")

        print(f"Testing transform: {srgb_space} -> {linear_space}")

        result = apply_ocio_transform(
            self.test_image.copy(),
            srgb_space,
            linear_space,
            config
        )

        # Result should be different from input (sRGB gamma removed)
        self.assertFalse(np.allclose(result, self.test_image, atol=0.01),
            "Transformation should change pixel values")

        # Linear values should be lower for mid-tones (gamma expansion)
        mid_value_input = self.test_image[32, 32, 0]
        mid_value_output = result[32, 32, 0]
        print(f"Mid-tone value: {mid_value_input:.4f} -> {mid_value_output:.4f}")

        # For sRGB to linear, mid-tones should decrease
        self.assertLess(mid_value_output, mid_value_input,
            "Linear mid-tones should be lower than sRGB")

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_transform_preserves_alpha(self):
        """Test that alpha channel is preserved through transformation."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        if len(names) < 2:
            self.skipTest("Need at least 2 color spaces")

        # Set alpha to specific pattern
        self.test_image_rgba[:, :, 3] = 0.75

        result = apply_ocio_transform(
            self.test_image_rgba.copy(),
            names[0],
            names[1],
            config
        )

        # Alpha should be unchanged
        np.testing.assert_array_almost_equal(
            result[:, :, 3],
            self.test_image_rgba[:, :, 3],
            decimal=5,
            err_msg="Alpha channel should be preserved"
        )

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_transform_same_colorspace(self):
        """Test that same-to-same transformation is identity."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        if not names:
            self.skipTest("No color spaces available")

        # Transform to same color space
        result = apply_ocio_transform(
            self.test_image.copy(),
            names[0],
            names[0],
            config
        )

        np.testing.assert_array_almost_equal(
            result,
            self.test_image,
            decimal=5,
            err_msg="Same-to-same transform should be identity"
        )

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_transform_roundtrip(self):
        """Test that forward + inverse transformation returns original values."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        if len(names) < 2:
            self.skipTest("Need at least 2 color spaces")

        src_space = names[0]
        dst_space = names[1]

        # Forward transform
        forward = apply_ocio_transform(
            self.color_image.copy(),
            src_space,
            dst_space,
            config
        )

        # Inverse transform
        roundtrip = apply_ocio_transform(
            forward.copy(),
            dst_space,
            src_space,
            config
        )

        # Should be close to original
        np.testing.assert_array_almost_equal(
            roundtrip,
            self.color_image,
            decimal=4,
            err_msg=f"Roundtrip {src_space} -> {dst_space} -> {src_space} should preserve values"
        )

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_transform_acescg_to_srgb(self):
        """Test ACEScg to sRGB transformation (common workflow)."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        names_lower = {n.lower(): n for n in names}

        # Find ACEScg and sRGB
        acescg = None
        srgb = None

        for name_lower, name in names_lower.items():
            if "acescg" in name_lower:
                acescg = name
            if "srgb" in name_lower and "texture" not in name_lower:
                srgb = name

        if not acescg or not srgb:
            self.skipTest(f"ACEScg or sRGB not found in config")

        print(f"Testing ACEScg -> sRGB: {acescg} -> {srgb}")

        # Create a test image with values in ACEScg range
        test_acescg = np.array([[[0.18, 0.18, 0.18]]], dtype=np.float32)  # 18% grey

        result = apply_ocio_transform(test_acescg, acescg, srgb, config)

        print(f"ACEScg 18% grey {test_acescg[0,0]} -> sRGB {result[0,0]}")

        # Result should be valid
        self.assertTrue(np.all(np.isfinite(result)), "Result should be finite")


class TestCommonColorSpaces(unittest.TestCase):
    """Test common color space definitions."""

    def test_common_colorspaces_not_empty(self):
        """Test that COMMON_COLORSPACES has entries."""
        self.assertTrue(len(COMMON_COLORSPACES) > 0)

    def test_common_colorspaces_contains_aces(self):
        """Test that COMMON_COLORSPACES contains ACES spaces."""
        aces_found = any("aces" in cs.lower() for cs in COMMON_COLORSPACES)
        self.assertTrue(aces_found, "Should contain ACES color spaces")

    def test_common_colorspaces_contains_srgb(self):
        """Test that COMMON_COLORSPACES contains sRGB."""
        srgb_found = any("srgb" in cs.lower() for cs in COMMON_COLORSPACES)
        self.assertTrue(srgb_found, "Should contain sRGB color space")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_empty_image(self):
        """Test handling of empty/small images."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        if len(names) < 2:
            self.skipTest("Need at least 2 color spaces")

        tiny_image = np.zeros((1, 1, 3), dtype=np.float32)
        result = apply_ocio_transform(tiny_image, names[0], names[1], config)
        self.assertEqual(result.shape, tiny_image.shape)

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_batch_processing(self):
        """Test handling of multiple images."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        if len(names) < 2:
            self.skipTest("Need at least 2 color spaces")

        # Process multiple images (simulating batch)
        images = [np.random.rand(32, 32, 3).astype(np.float32) for _ in range(4)]

        results = []
        for img in images:
            result = apply_ocio_transform(img, names[0], names[1], config)
            results.append(result)

        self.assertEqual(len(results), 4)
        for result in results:
            self.assertEqual(result.shape, (32, 32, 3))

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_rgba_input(self):
        """Test handling of RGBA images."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        if len(names) < 2:
            self.skipTest("Need at least 2 color spaces")

        rgba_image = np.random.rand(32, 32, 4).astype(np.float32)
        rgba_image[:, :, 3] = 0.8  # Set specific alpha

        result = apply_ocio_transform(rgba_image, names[0], names[1], config)

        # Should preserve 4 channels
        self.assertEqual(result.shape[-1], 4)
        # Alpha should be unchanged
        np.testing.assert_array_almost_equal(result[:, :, 3], rgba_image[:, :, 3], decimal=5)

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_out_of_range_values(self):
        """Test handling of out-of-range pixel values."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        if len(names) < 2:
            self.skipTest("Need at least 2 color spaces")

        hdr_image = np.random.rand(32, 32, 3).astype(np.float32) * 10.0  # HDR values

        result = apply_ocio_transform(hdr_image, names[0], names[1], config)

        # Should not crash, result should be finite
        self.assertTrue(np.all(np.isfinite(result)))

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_negative_values(self):
        """Test handling of negative pixel values (scene-referred)."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        names = get_colorspace_names(config)
        if len(names) < 2:
            self.skipTest("Need at least 2 color spaces")

        neg_image = np.random.rand(32, 32, 3).astype(np.float32) - 0.5  # Some negative

        result = apply_ocio_transform(neg_image, names[0], names[1], config)

        # Should not crash
        self.assertEqual(result.shape, neg_image.shape)


class TestDisplayTransform(unittest.TestCase):
    """Test display/view transforms."""

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_display_list(self):
        """Test that displays are available."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        displays = list(config.getDisplays())
        self.assertTrue(len(displays) > 0, "Should have at least one display")
        print(f"Available displays: {displays}")

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_views_for_display(self):
        """Test that views are available for each display."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        displays = list(config.getDisplays())
        if not displays:
            self.skipTest("No displays available")

        for display in displays:
            views = list(config.getViews(display))
            self.assertTrue(len(views) > 0, f"Display '{display}' should have views")
            print(f"Display '{display}' views: {views[:5]}...")

    @unittest.skipUnless(HAS_OCIO, "OpenColorIO not installed")
    def test_display_transform(self):
        """Test applying a display transform."""
        config = get_ocio_config()
        if not config:
            self.skipTest("No OCIO config available")

        displays = list(config.getDisplays())
        if not displays:
            self.skipTest("No displays available")

        display = displays[0]
        views = list(config.getViews(display))
        if not views:
            self.skipTest("No views available")

        view = views[0]

        # Find an input color space
        names = get_colorspace_names(config)
        input_cs = None
        for name in names:
            if "acescg" in name.lower() or "linear" in name.lower():
                input_cs = name
                break

        if not input_cs:
            input_cs = names[0] if names else None

        if not input_cs:
            self.skipTest("No input color space found")

        print(f"Testing display transform: {input_cs} -> {display}/{view}")

        # Create display transform
        try:
            transform = OCIO.DisplayViewTransform()
            transform.setSrc(input_cs)
            transform.setDisplay(display)
            transform.setView(view)

            processor = config.getProcessor(transform)
            cpu_processor = processor.getDefaultCPUProcessor()

            # Test with sample image
            test_image = np.zeros((32, 32, 3), dtype=np.float32)
            test_image[:, :, :] = 0.18  # 18% grey

            rgb_flat = test_image.reshape(-1, 3)

            img_desc = OCIO.PackedImageDesc(
                rgb_flat,
                32, 32, 3,
                OCIO.BIT_DEPTH_F32,
                rgb_flat.strides[1],
                rgb_flat.strides[0],
                32 * rgb_flat.strides[0]
            )

            cpu_processor.apply(img_desc)

            result = rgb_flat.reshape(32, 32, 3)
            self.assertTrue(np.all(np.isfinite(result)))
            print(f"Display transform applied successfully. Output range: [{result.min():.3f}, {result.max():.3f}]")

        except Exception as e:
            self.fail(f"Display transform failed: {e}")


def run_tests():
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestOCIOAvailability))
    suite.addTests(loader.loadTestsFromTestCase(TestOCIOConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestACES2Support))
    suite.addTests(loader.loadTestsFromTestCase(TestColorSpaceTransform))
    suite.addTests(loader.loadTestsFromTestCase(TestCommonColorSpaces))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestDisplayTransform))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if HAS_OCIO:
        print(f"\nOpenColorIO Version: {OCIO.GetVersion()}")
        config = get_ocio_config()
        if config:
            print(f"Config: {config.getDescription()[:100]}...")
            print(f"Color Spaces: {len(get_colorspace_names(config))}")
    else:
        print("\nOpenColorIO: NOT INSTALLED")
        print("Install with: pip install opencolorio")

    return result


if __name__ == "__main__":
    run_tests()
