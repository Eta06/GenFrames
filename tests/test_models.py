"""
Test model implementations
"""

import pytest
import numpy as np
from genframes.models.rife import RIFEModel
from genframes.backends.base import get_backend


def test_rife_model_initialization():
    """Test RIFE model initialization"""
    backend = get_backend("cpu")
    model = RIFEModel(backend=backend, model_version="rife-v4.6")

    assert model.model_name == "rife-v4.6"
    assert model.backend == backend


def test_preprocess_postprocess():
    """Test preprocessing and postprocessing"""
    backend = get_backend("cpu")
    model = RIFEModel(backend=backend)

    # Test frame [0, 255]
    frame_uint8 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Preprocess
    frame_float = model.preprocess(frame_uint8)
    assert frame_float.dtype == np.float32
    assert frame_float.min() >= 0.0
    assert frame_float.max() <= 1.0

    # Postprocess
    frame_back = model.postprocess(frame_float)
    assert frame_back.dtype == np.uint8

    # Should be close to original
    np.testing.assert_array_almost_equal(frame_uint8, frame_back, decimal=0)


if __name__ == "__main__":
    pytest.main([__file__])
