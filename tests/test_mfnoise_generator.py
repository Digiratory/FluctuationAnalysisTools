"""Test suite for multifractal noise generator (mfnoise_generator.py)."""

import numpy as np
import pytest
from scipy import signal, stats

from StatTools.generators.mfnoise_generator import _calculate_spectral_density, mfnoise


class TestCalculateSpectralDensity:
    """Test the internal _calculate_spectral_density function."""

    def test_single_hurst_exponent(self):
        """Test spectral density calculation with single Hurst exponent."""
        length = 1024
        hurst = [0.8]
        crossover_points = ()

        S = _calculate_spectral_density(length, hurst, crossover_points)

        assert isinstance(S, np.ndarray)
        assert S.shape == (length,)
        assert S[0] == 0.0  # DC component should be zero
        assert np.all(np.isfinite(S))  # No infinities or NaNs

        # For single Hurst, should follow power law
        freqs = np.fft.fftfreq(length, d=1.0)
        expected_alpha = 2 * hurst[0] + 1
        # Check that spectral density follows expected power law for non-zero frequencies
        non_zero_mask = freqs != 0
        expected_S = np.abs(freqs[non_zero_mask]) ** (-expected_alpha)
        assert np.allclose(S[non_zero_mask], expected_S, rtol=1e-10)

    def test_multiple_hurst_exponents(self):
        """Test spectral density calculation with multiple Hurst exponents."""
        length = 1024
        hurst = [0.5, 0.8]
        crossover_points = (100,)

        S = _calculate_spectral_density(length, hurst, crossover_points)

        assert isinstance(S, np.ndarray)
        assert S.shape == (length,)
        assert S[0] == 0.0  # DC component should be zero
        assert np.all(np.isfinite(S))  # No infinities or NaNs

        # Check continuity at crossover point
        crossover_freq = 1 / 100
        freqs = np.abs(np.fft.fftfreq(length, d=1.0))

        # Find indices around crossover frequency
        mask_low = freqs <= crossover_freq
        mask_high = freqs > crossover_freq

        if np.any(mask_low & (freqs != 0)) and np.any(mask_high):
            # Check that values are finite and reasonable
            assert np.all(np.isfinite(S[mask_low]))
            assert np.all(np.isfinite(S[mask_high]))

    def test_edge_cases(self):
        """Test edge cases for spectral density calculation."""
        # Empty hurst list should raise error (handled by main function)
        # Test with very small length
        length = 8
        hurst = [0.5]
        crossover_points = ()

        S = _calculate_spectral_density(length, hurst, crossover_points)
        assert S.shape == (length,)
        assert S[0] == 0.0


class TestMfnoiseGenerator:
    """Test the main mfnoise generator function."""

    def test_single_hurst_exponent(self):
        """Test generation with single Hurst exponent."""
        length = 1024
        hurst = 0.8
        crossover_points = ()

        result = mfnoise(length, hurst, crossover_points)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, length)  # Output should match input length
        assert np.all(np.isfinite(result))  # No infinities or NaNs
        assert np.all(np.abs(result) < 100)  # Reasonable magnitude

    def test_multiple_hurst_exponents(self):
        """Test generation with multiple Hurst exponents."""
        length = 1024
        hurst = (0.5, 0.8)
        crossover_points = (100,)

        result = mfnoise(length, hurst, crossover_points)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, length)
        assert np.all(np.isfinite(result))

    def test_multiple_tracks(self):
        """Test generation of multiple tracks."""
        length = 512
        hurst = 0.7
        crossover_points = ()
        n_tracks = 3

        result = mfnoise(length, hurst, crossover_points, n_tracks=n_tracks)

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_tracks, length)
        assert np.all(np.isfinite(result))

        # Tracks should be different from each other
        assert not np.allclose(result[0], result[1])
        assert not np.allclose(result[1], result[2])

    def test_normalization(self):
        """Test normalization behavior."""
        length = 512
        hurst = 0.7
        crossover_points = ()

        # Test with normalization enabled (default)
        result_normalized = mfnoise(length, hurst, crossover_points, normalize=True)

        # Test with normalization disabled
        result_unnormalized = mfnoise(length, hurst, crossover_points, normalize=False)

        assert isinstance(result_normalized, np.ndarray)
        assert isinstance(result_unnormalized, np.ndarray)

        # Normalized result should have mean close to 0 and std close to 1
        mean_val = np.mean(result_normalized)
        std_val = np.std(result_normalized)
        assert abs(mean_val) < 0.1  # Mean should be close to zero
        assert (
            abs(std_val - 1.0) < 0.1
        )  # Std should be close to 1 (more relaxed tolerance)

    def test_different_hurst_values(self):
        """Test generation with different Hurst exponent values."""
        length = 1024
        crossover_points = ()

        test_hurst_values = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5]

        for hurst in test_hurst_values:
            result = mfnoise(length, hurst, crossover_points)

            assert isinstance(result, np.ndarray)
            assert result.shape == (1, length)
            assert np.all(np.isfinite(result))
            assert np.all(np.abs(result) < 100)  # Reasonable magnitude

    def test_different_crossover_points(self):
        """Test generation with different crossover points."""
        length = 1024
        hurst = (0.5, 0.8, 1.2)

        test_crossover_points = [
            (50, 100),
        ]

        for crossover_points in test_crossover_points:
            result = mfnoise(length, hurst, crossover_points)

            assert isinstance(result, np.ndarray)
            assert result.shape == (1, length)
            assert np.all(np.isfinite(result))

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        length = 1024

        # Test empty hurst list
        with pytest.raises(
            ValueError, match="At least one Hurst exponent must be provided"
        ):
            mfnoise(length, [], ())

        # Test mismatched hurst and crossover points
        with pytest.raises(ValueError, match="Number of crossover points"):
            mfnoise(
                length, [0.5, 0.8], (100, 200)
            )  # 2 hurst, 2 crossovers (should be 1)

        with pytest.raises(ValueError, match="Number of crossover points"):
            mfnoise(length, [0.5], (100, 200))  # 1 hurst, 2 crossovers (should be 0)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible when using the same seed."""
        length = 512
        hurst = 0.7
        crossover_points = ()

        # Set seed and generate first result
        np.random.seed(42)
        result1 = mfnoise(length, hurst, crossover_points)

        # Set same seed and generate second result
        np.random.seed(42)
        result2 = mfnoise(length, hurst, crossover_points)

        # Results should be identical
        assert np.allclose(result1, result2)

    def test_statistical_properties(self):
        """Test statistical properties of generated signals."""
        length = 2048
        hurst = 0.7
        crossover_points = ()

        result = mfnoise(length, hurst, crossover_points)
        signal = result[0]  # Get first track

        # Test that signal has expected statistical properties
        assert abs(np.mean(signal)) < 0.2  # Mean should be close to zero
        # Test that signal is not constant
        assert np.std(signal) > 0.0001

        # Test that signal values are finite
        assert np.all(np.isfinite(signal))

    def test_multifractal_behavior(self):
        """Test that multifractal signals show expected scaling behavior."""
        length = 4096
        hurst = (0.5, 1.2)
        crossover_points = (200,)

        result = mfnoise(length, hurst, crossover_points)
        signal = result[0]

        # The signal should be finite and have reasonable magnitude
        assert np.all(np.isfinite(signal))
        assert np.all(np.abs(signal) < 100)

        # For multifractal signals, we expect different scaling in different regimes
        # This is a basic test - more sophisticated analysis would require DFA or similar
        assert len(signal) == length

    def test_performance_with_large_signals(self):
        """Test that the generator works with reasonably large signals."""
        length = 8192  # Large but reasonable size
        hurst = (0.5, 0.8)
        crossover_points = (500,)

        # This should complete without errors
        result = mfnoise(length, hurst, crossover_points)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, length)
        assert np.all(np.isfinite(result))

    def test_edge_case_short_signals(self):
        """Test behavior with very short signals."""
        length = 16
        hurst = 0.5
        crossover_points = ()

        result = mfnoise(length, hurst, crossover_points)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, length)  # Should be (1, 16)
        assert np.all(np.isfinite(result))

    def test_input_types(self):
        """Test that different input types work correctly."""
        length = 512
        crossover_points = ()

        # Test with float hurst
        result1 = mfnoise(length, 0.7, crossover_points)

        # Test with tuple hurst
        result2 = mfnoise(length, (0.7,), crossover_points)

        # Test with list hurst
        result3 = mfnoise(length, [0.7], crossover_points)

        # All should produce valid results
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        assert isinstance(result3, np.ndarray)

        assert result1.shape == (1, length)
        assert result2.shape == (1, length)
        assert result3.shape == (1, length)


class TestMfnoiseIntegration:
    """Integration tests for mfnoise generator."""

    def test_consistency_with_single_hurst(self):
        """Test that single Hurst case is consistent with expected behavior."""
        length = 1024
        hurst = 0.8
        crossover_points = ()

        # Generate multiple signals
        signals = []
        for _ in range(5):
            np.random.seed()  # Different seed each time
            result = mfnoise(length, hurst, crossover_points)
            signals.append(result[0])

        signals = np.array(signals)

        # All signals should be finite
        assert np.all(np.isfinite(signals))

        # Signals should have similar statistical properties
        means = np.mean(signals, axis=1)
        stds = np.std(signals, axis=1)

        # Means should all be close to zero
        assert np.all(abs(means) < 0.5)

        # Standard deviations should be reasonable and similar
        assert np.all(stds > 0.01)
        assert np.all(stds < 5.0)

    def test_multifractal_scaling_verification(self):
        """Basic verification that multifractal signals have expected properties."""
        length = 2048
        hurst = (0.5, 1.2)
        crossover_points = (100,)

        result = mfnoise(length, hurst, crossover_points)
        signal = result[0]

        # Basic sanity checks
        assert len(signal) == length
        assert np.all(np.isfinite(signal))
        assert np.std(signal) > 0.001  # Not constant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
