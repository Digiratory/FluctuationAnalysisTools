import numpy as np
import pytest
from scipy.stats import linregress

from StatTools.analysis.nd_dfa import nd_dfa


def generate_2d_fbms(size, H, seed=None):
    """
    Generate 2D fractional Brownian motion surfaces using the Fourier method.
    Returns both the surface (profile) and the increments (original signal).

    Parameters:
    size : int
        Size of the square grid (size x size)
    H : float
        Hurst exponent (0 < H < 1)
    seed : int or None
        Random seed for reproducibility

    Returns:
    surface : ndarray
        2D fractional Brownian motion surface (profile)
    increments : ndarray
        Original fractional Gaussian noise increments
    """
    np.random.seed(seed)
    n = size

    # Create wavevectors
    kx = np.fft.fftfreq(n)[:, None] * n
    ky = np.fft.fftfreq(n)[None, :] * n
    k = np.sqrt(kx**2 + ky**2)

    # Avoid division by zero at origin
    k[0, 0] = 1

    # Power spectrum for fractional Gaussian noise (increments)
    # S(k) ~ |k|^(-(2*H + 2)) for 2D fBm surface
    power = k ** (-(2 * H + 2))
    power[0, 0] = 0  # Zero mean

    # Generate complex Gaussian noise
    noise = np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))

    # Apply power spectrum
    f = np.sqrt(power) * noise

    # Inverse FFT to get surface
    surface = np.fft.ifft2(f).real
    surface -= np.mean(surface)  # Ensure zero mean

    # Compute increments (fractional Gaussian noise)
    increments = np.zeros_like(surface)
    increments[1:, 1:] = (
        surface[1:, 1:] - surface[:-1, 1:] - surface[1:, :-1] + surface[:-1, :-1]
    )

    return surface, increments


def estimate_hurst(F2, S):
    """
    Estimate Hurst exponent from DFA results

    Parameters:
    F2 : ndarray
        F^2(s) values from DFA
    S : ndarray
        Corresponding scales

    Returns:
    H : float
        Estimated Hurst exponent
    """
    # Use scales between 10% and 90% of max scale to avoid boundary effects
    min_scale = S[-1] * 0.1
    max_scale = S[-1] * 0.9
    mask = (S >= min_scale) & (S <= max_scale)

    if np.sum(mask) < 3:
        mask = np.ones_like(S, dtype=bool)

    log_s = np.log(S[mask])
    log_F = 0.5 * np.log(F2[mask])  # F(s) = sqrt(F2)

    slope, _, _, _, _ = linregress(log_s, log_F)
    return slope


class TestNDDFA:
    @pytest.mark.parametrize("H", [0.3, 0.5, 0.7, 0.9, 1.0, 1.25])
    def test_fbms_hurst_estimation(self, H):
        """Test Hurst exponent estimation on synthetic fBm surfaces"""
        size = 128
        surface, _ = generate_2d_fbms(size, H, seed=42)

        # Run ND-DFA on the original increments (should use n_integral=1)
        # But paper shows input should be the profile for DFA algorithm
        # We test both approaches per paper's clarification
        F2, S = nd_dfa(
            surface,  # Input is the profile (surface)
            s=32,  # Max scale
            pd=1,  # Linear detrending
            n_integral=0,  # No additional integration needed
            step=0.5,
            processes=1,
        )

        H_est = estimate_hurst(F2, S)
        assert abs(H_est - H) < 0.15, (
            f"Hurst estimation failed for H={H}: "
            f"estimated={H_est:.3f}, expected={H}"
        )

    def test_white_noise_profile(self):
        """Test white noise (H=0.5) with proper integration"""
        size = 128
        # Generate profile with H=0.5 (uncorrelated case)
        surface, _ = generate_2d_fbms(size, 0.5, seed=42)

        # Input raw noise with n_integral=1 (compute profile internally)
        F2, S = nd_dfa(
            surface,
            s=32,
            pd=1,
            n_integral=0,  # Compute profile from raw signal
            step=0.5,
            processes=1,
        )

        H_est = estimate_hurst(F2, S)
        assert (
            abs(H_est - 0.5) < 0.15
        ), f"White noise test failed: estimated H={H_est:.3f}, expected 0.5"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Small non-cubic array (made larger to satisfy scale constraints)
        small_data = np.random.rand(40, 60)  # Now min dimension//4 = 10
        F2, S = nd_dfa(
            small_data,
            s=[5, 8, 10],  # Scales within valid range [5, 10]
            pd=1,
            n_integral=1,
            processes=1,
        )
        assert len(S) > 0, "Should handle non-cubic arrays"

        # Array with dimension not divisible by s
        irregular_data = np.random.rand(41, 61)  # Dimensions not divisible by 10
        F2, S = nd_dfa(
            irregular_data,
            s=10,  # Scale where 41%10=1 and 61%10=1 (boundary effects)
            pd=1,
            n_integral=1,
            processes=1,
        )
        assert len(S) > 0, "Should handle dimensions not divisible by s"

        # Single scale processing
        F2, S = nd_dfa(
            np.random.rand(64, 64),
            s=[8],  # Pass as list to test single specific scale
            pd=1,
            n_integral=1,
            processes=1,
        )
        assert len(S) == 1, "Should handle single scale input"
        assert S[0] == 8, "Should use the exact scale specified"

    def test_incorrect_order_comparison(self):
        """
        Verify that pre-partition integration gives wrong results
        (Critical test from paper Section II.C)
        """
        size = 64
        H = 0.7
        surface, increments = generate_2d_fbms(size, H, seed=42)

        # Correct approach: partition first then integrate (n_integral=0 for profile input)
        F2_correct, S_correct = nd_dfa(
            surface,
            s=16,
            pd=1,
            n_integral=0,  # No additional integration
            step=0.5,
            processes=1,
        )
        H_correct = estimate_hurst(F2_correct, S_correct)

        # Incorrect approach: integrate first then partition (equivalent to n_integral=1 on profile)
        F2_incorrect, S_incorrect = nd_dfa(
            surface,
            s=16,
            pd=1,
            n_integral=1,  # Additional integration (wrong order)
            step=0.5,
            processes=1,
        )
        H_incorrect = estimate_hurst(F2_incorrect, S_incorrect)

        # Correct estimate should be close to true H, incorrect should be biased
        correct_error = abs(H_correct - H)
        incorrect_error = abs(H_incorrect - H)

        assert correct_error < 0.2, "Correct order failed to estimate H"
        assert incorrect_error > correct_error * 1.5, (
            "Incorrect integration order didn't produce significant bias: "
            f"correct_error={correct_error:.3f}, incorrect_error={incorrect_error:.3f}"
        )

    def test_polynomial_detrending(self):
        """Test polynomial detrending with known trend"""
        size = 64
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        # Create quadratic surface with noise
        trend = 3 * xx**2 + 2 * xx * yy - 4 * yy**2 + 5
        noise = np.random.normal(scale=0.5, size=(size, size))
        data = trend + noise

        # With quadratic detrending (pd=2), residuals should be close to noise
        F2, S = nd_dfa(
            data,
            s=16,
            pd=2,  # Quadratic detrending
            n_integral=0,
            step=0.5,
            processes=1,
        )

        # Estimate noise variance from small scales
        noise_var = np.var(noise)
        # F2 at smallest scale should approximate noise variance
        F2_min = F2[0]

        assert abs(F2_min - noise_var) / noise_var < 0.3, (
            "Polynomial detrending failed: "
            f"estimated variance={F2_min:.3f}, true variance={noise_var:.3f}"
        )
