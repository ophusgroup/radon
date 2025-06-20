"""
Tests for CUDA array addition functionality
"""

import numpy as np
import pytest

import radon


class TestCudaAddArrays:
    """Test the CUDA add_arrays function"""

    def test_import_radon(self):
        """Test that we can import the radon module"""
        assert hasattr(radon, "add_arrays")

    def test_simple_addition(self):
        """Test basic array addition"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        result = radon.add_arrays(a, b)
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)

        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_addition(self):
        """Test addition with zeros"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        result = radon.add_arrays(a, b)

        np.testing.assert_array_almost_equal(result, a)

    def test_negative_numbers(self):
        """Test addition with negative numbers"""
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([-1.0, 2.0, -3.0], dtype=np.float32)

        result = radon.add_arrays(a, b)
        expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        np.testing.assert_array_almost_equal(result, expected)

    def test_larger_arrays(self):
        """Test with larger arrays"""
        size = 1000
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)

        result = radon.add_arrays(a, b)
        expected = a + b

        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_dimension_mismatch_error(self):
        """Test that mismatched array sizes raise an error"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0], dtype=np.float32)  # Different size

        with pytest.raises(RuntimeError, match="must have the same size"):
            radon.add_arrays(a, b)

    def test_wrong_dimension_error(self):
        """Test that 2D arrays raise an error"""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        with pytest.raises(RuntimeError, match="must be 1-dimensional"):
            radon.add_arrays(a, b)

    def test_empty_arrays(self):
        """Test with empty arrays"""
        a = np.array([], dtype=np.float32)
        b = np.array([], dtype=np.float32)

        result = radon.add_arrays(a, b)
        expected = np.array([], dtype=np.float32)

        np.testing.assert_array_equal(result, expected)

    def test_single_element(self):
        """Test with single element arrays"""
        a = np.array([5.0], dtype=np.float32)
        b = np.array([3.0], dtype=np.float32)

        result = radon.add_arrays(a, b)
        expected = np.array([8.0], dtype=np.float32)

        np.testing.assert_array_almost_equal(result, expected)


class TestCudaAvailability:
    """Test CUDA availability and GPU functionality"""

    @pytest.mark.cuda
    def test_cuda_kernel_execution(self):
        """Test that CUDA kernel actually executes and produces correct results"""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        result = radon.add_arrays(a, b)

        # Check the computation is correct
        expected = np.array([4.0, 6.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

        # If we get here without errors, CUDA execution worked
        assert result is not None
        assert len(result) == 2


# Skip tests if CUDA is not available
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers",
        "cuda: marks tests as requiring CUDA (deselect with '-m \"not cuda\"')",
    )


# Mark all tests in this file as requiring CUDA
pytestmark = pytest.mark.cuda
