import numpy as np
import pytest

from StatTools.generators import generate_fbn
from StatTools.generators.kasdin_generator import create_kasdin_generator
from StatTools.generators.lbfbm_generator import LBFBmGenerator

# Test data
testdata = {
    "hurst_values": [0.5, 0.7, 0.9],
    "lengths": [2**12, 2**14],
    "methods": [
        "kasdin",
        # "lbfbm",
    ],
}


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("hurst", testdata["hurst_values"])
@pytest.mark.parametrize("length", testdata["lengths"])
@pytest.mark.parametrize("method", testdata["methods"])
def test_unified_generator_basic(hurst: float, length: int, method: str):
    """
    Test basic functionality of the unified generate_fbn function.

    Parameters:
        hurst: Hurst exponent to test
        length: Length of sequence to generate
        method: Generator method to use
    """
    # Test that the function runs without error
    result = generate_fbn(hurst, length, method=method)

    # Test return type and shape
    assert isinstance(
        result, np.ndarray
    ), f"Result should be numpy array, got {type(result)}"
    assert result.shape == (
        1,
        length,
    ), f"Expected shape (1, {length}), got {result.shape}"

    # Test that values are reasonable (not all zeros, not NaN)
    assert not np.all(result == 0), "Generated sequence should not be all zeros"
    assert not np.any(
        np.isnan(result)
    ), "Generated sequence should not contain NaN values"

    # Test that mean is close to zero (for normalized generators)
    mean_abs = np.abs(np.mean(result))
    assert mean_abs < 0.2, f"Mean should be close to zero, got {mean_abs}"

    # Test that std is reasonable (close to 1 for normalized generators)
    std = np.std(result)
    assert 0.5 < std < 2.0, f"Standard deviation should be reasonable, got {std}"


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("hurst", testdata["hurst_values"])
def test_unified_generator_default_method(hurst: float):
    """
    Test that default method (kasdin) works correctly.
    """
    length = 500
    result = generate_fbn(hurst, length)  # Should default to kasdin

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, length)

    # Compare with direct kasdin generator to ensure consistency
    kasdin_result = generate_fbn(hurst, length, method="kasdin")

    # Results should be similar (same generator, same parameters)
    assert result.shape == kasdin_result.shape


def test_unified_generator_invalid_method():
    """
    Test error handling for invalid method names.
    """
    with pytest.raises(ValueError, match="Unknown generator method"):
        generate_fbn(0.7, 100, method="invalid_method")


def test_unified_generator_kwargs_forwarding():
    """
    Test that additional kwargs are properly forwarded to underlying generators.
    """
    hurst = 0.7
    length = 500

    # Test kasdin with specific kwargs
    result_kasdin = generate_fbn(
        hurst, length, method="kasdin", normalize=False, filter_coefficients_length=250
    )
    assert result_kasdin.shape == (1, length)

    # Test lbfbm with specific kwargs
    result_lbfbm = generate_fbn(hurst, length, method="lbfbm", base=1.2)
    assert result_lbfbm.shape == (1, length)


def test_backward_compatibility():
    """
    Test that existing generators still work (backward compatibility).
    """
    hurst = 0.7
    length = 500

    # Test original kasdin generator
    generator = create_kasdin_generator(hurst, length)
    sequence = generator.get_full_sequence()
    assert sequence.shape == (length,)

    # Test original lbfbm generator
    generator = LBFBmGenerator(h=hurst, length=length)
    sequence = np.array(list(generator))
    assert sequence.shape == (length,)


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("hurst", [0.2, 0.5, 0.8, 1.2, 1.5])
def test_unified_generator_various_hurst(hurst: float):
    """
    Test unified generator with various Hurst exponent values.
    """
    length = 10000

    # Test both methods with different hurst values
    for method in ["kasdin", "lbfbm"]:
        result = generate_fbn(hurst, length, method=method)

        assert result.shape == (1, length)
        assert not np.any(np.isnan(result))
        assert not np.all(result == 0)
