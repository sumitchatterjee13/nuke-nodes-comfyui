"""
Tests for merge nodes
"""

import os
import sys

import pytest
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merge_nodes import NukeMerge, NukeMix
from tests.test_utils import assert_image_valid, create_test_image


class TestNukeMerge:
    """Test cases for NukeMerge node"""

    def test_merge_over_operation(self):
        """Test basic over operation"""
        merge_node = NukeMerge()

        # Create test images
        image_a = create_test_image(64, 64, 4)  # With alpha
        image_b = create_test_image(64, 64, 4)

        # Test over operation
        result = merge_node.merge(image_a, image_b, operation="over", mix=1.0)

        assert len(result) == 1
        output_image = result[0]
        assert_image_valid(output_image, expected_shape=(1, 64, 64, 4))

    def test_merge_add_operation(self):
        """Test add blend mode"""
        merge_node = NukeMerge()

        image_a = create_test_image(32, 32, 3)
        image_b = create_test_image(32, 32, 3)

        result = merge_node.merge(image_a, image_b, operation="add", mix=1.0)

        output_image = result[0]
        assert_image_valid(output_image, expected_shape=(1, 32, 32, 3))

    def test_merge_with_mask(self):
        """Test merge operation with mask"""
        merge_node = NukeMerge()

        image_a = create_test_image(32, 32, 3)
        image_b = create_test_image(32, 32, 3)
        mask = create_test_image(32, 32, 1)  # Single channel mask

        result = merge_node.merge(
            image_a, image_b, operation="multiply", mix=0.5, mask=mask
        )

        output_image = result[0]
        assert_image_valid(output_image, expected_shape=(1, 32, 32, 3))

    def test_merge_mix_parameter(self):
        """Test mix parameter functionality"""
        merge_node = NukeMerge()

        image_a = torch.zeros(1, 32, 32, 3)
        image_b = torch.ones(1, 32, 32, 3)

        # Test mix=0 (should return image_a)
        result_0 = merge_node.merge(image_a, image_b, "over", 0.0)
        torch.testing.assert_close(result_0[0], image_a, atol=1e-6, rtol=1e-6)

        # Test mix=1 (should return blended result)
        result_1 = merge_node.merge(image_a, image_b, "over", 1.0)
        assert not torch.allclose(result_1[0], image_a)

    def test_all_blend_modes(self):
        """Test all available blend modes"""
        merge_node = NukeMerge()

        image_a = create_test_image(16, 16, 3) * 0.5
        image_b = create_test_image(16, 16, 3) * 0.7

        blend_modes = [
            "over",
            "add",
            "multiply",
            "screen",
            "overlay",
            "soft_light",
            "hard_light",
            "color_dodge",
            "color_burn",
            "darken",
            "lighten",
            "difference",
            "exclusion",
            "subtract",
            "divide",
        ]

        for mode in blend_modes:
            result = merge_node.merge(image_a, image_b, mode, 1.0)
            assert_image_valid(result[0], expected_shape=(1, 16, 16, 3))


class TestNukeMix:
    """Test cases for NukeMix node"""

    def test_basic_mix(self):
        """Test basic mixing functionality"""
        mix_node = NukeMix()

        image_a = torch.zeros(1, 32, 32, 3)
        image_b = torch.ones(1, 32, 32, 3)

        # Test 50% mix
        result = mix_node.mix(image_a, image_b, 0.5)
        expected = torch.full_like(image_a, 0.5)

        torch.testing.assert_close(result[0], expected, atol=1e-6, rtol=1e-6)

    def test_mix_extremes(self):
        """Test mix at extreme values"""
        mix_node = NukeMix()

        image_a = torch.zeros(1, 16, 16, 3)
        image_b = torch.ones(1, 16, 16, 3)

        # Test mix=0 (should return image_a)
        result_0 = mix_node.mix(image_a, image_b, 0.0)
        torch.testing.assert_close(result_0[0], image_a, atol=1e-6, rtol=1e-6)

        # Test mix=1 (should return image_b)
        result_1 = mix_node.mix(image_a, image_b, 1.0)
        torch.testing.assert_close(result_1[0], image_b, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
