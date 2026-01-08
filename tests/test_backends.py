"""
Test backend implementations
"""

import pytest
import numpy as np
from genframes.backends.base import detect_best_backend, get_backend


def test_detect_best_backend():
    """Test backend detection"""
    backend = detect_best_backend()
    assert backend in ["mlx", "mps", "cuda", "cpu"]


def test_get_backend_auto():
    """Test getting backend with auto selection"""
    backend = get_backend("auto")
    assert backend is not None
    assert backend.is_available()


def test_backend_tensor_conversion():
    """Test tensor conversion"""
    backend = get_backend("cpu")
    backend.initialize()

    # Test numpy to tensor and back
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = backend.to_tensor(arr)
    arr_back = backend.to_numpy(tensor)

    np.testing.assert_array_almost_equal(arr, arr_back)


if __name__ == "__main__":
    pytest.main([__file__])
