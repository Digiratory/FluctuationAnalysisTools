"""
Comprehensive test suite for pybind11 bindings.

This test module verifies that:
1. All C++ functions work correctly through Python bindings
2. Backward compatibility with existing C API
3. All functionality works through new bindings

"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import both old and new modules to test compatibility
try:
    from StatTools import C_StatTools

    OLD_API_AVAILABLE = True
except ImportError:
    OLD_API_AVAILABLE = False
    print("Warning: Original C API not available for comparison")

try:
    from StatTools import StatTools_bindings

    NEW_API_AVAILABLE = True
except ImportError:
    NEW_API_AVAILABLE = False
    print("Warning: New pybind11 bindings not available")


class TestPybind11Bindings:
    """Test pybind11 bindings for StatTools functions"""

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing"""
        np.random.seed(42)  # For reproducible tests
        return {
            "input_vector": np.random.exponential(2.0, 100),
            "U": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            "C0_input": -1.0,
        }

    def test_exponential_distribution(self, sample_data):
        """Test exponential distribution value generation"""
        if not NEW_API_AVAILABLE:
            pytest.skip("New API not available")

        # Test single value generation
        result = StatTools_bindings.get_exponential_dist_value(2.0)
        assert isinstance(result, float)
        assert result > 0  # Exponential values should be positive

        # Test vector generation
        vector_result = StatTools_bindings.get_exp_dist_vector(2.0, 10)
        assert len(vector_result) == 10
        assert all(x > 0 for x in vector_result)

    def test_gaussian_distribution(self, sample_data):
        """Test Gaussian distribution value generation"""
        if not NEW_API_AVAILABLE:
            pytest.skip("New API not available")

        # Generate 50 values and test statistical properties
        values = [StatTools_bindings.get_gauss_dist_value() for _ in range(100)]

        # All values should be floats
        assert all(isinstance(x, float) for x in values)

        # Test for normality using Shapiro-Wilk test
        # Null hypothesis: the sample comes from a normal distribution
        # We want to fail to reject the null hypothesis (p-value > 0.05)
        from scipy.stats import shapiro

        stat, p_value = shapiro(values)
        assert (
            p_value > 0.05
        ), f"Values do not appear to be normally distributed (p-value: {p_value:.4f})"

        # Calculate mean and standard deviation
        sample_mean = np.mean(values)
        sample_std = np.std(values, ddof=1)  # Sample standard deviation

        # For standard normal distribution, mean should be close to 0
        # and std should be close to 1
        # Using reasonable tolerances for statistical testing
        assert_allclose(sample_mean, 0.0, atol=0.3)  # Allow 0.3 tolerance for mean
        assert_allclose(sample_std, 1.0, atol=0.2)  # Allow 0.2 tolerance for std

    def test_cumulative_sum(self, sample_data):
        """Test cumulative sum functionality"""
        if not NEW_API_AVAILABLE:
            pytest.skip("New API not available")

        test_array = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.array([1.0, 3.0, 6.0, 10.0])

        result = StatTools_bindings.cumsum(test_array)
        assert_allclose(result, expected)

    def test_waiting_time_calculation(self, sample_data):
        """Test main waiting time calculation function"""
        if not NEW_API_AVAILABLE:
            pytest.skip("New API not available")

        input_vector = sample_data["input_vector"]
        U = sample_data["U"]
        C0_input = sample_data["C0_input"]

        result = StatTools_bindings.get_waiting_time(input_vector, U, C0_input)

        # Verify result properties
        assert len(result) == len(U)
        assert all(isinstance(x, (float, np.floating)) for x in result)
        assert all(x >= 0 for x in result)  # Waiting times should be non-negative

    def test_poisson_thread_generation(self, sample_data):
        """Test Poisson thread generation"""
        if not NEW_API_AVAILABLE:
            pytest.skip("New API not available")

        input_vector = sample_data["input_vector"][:10]  # Smaller subset for testing
        result = StatTools_bindings.get_poisson_thread(input_vector, 1.0)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert all(x > 0 for x in result)  # Exponential values should be positive

    def test_model_function(self, sample_data):
        """Test main model function"""
        if not NEW_API_AVAILABLE:
            pytest.skip("New API not available")

        input_vector = sample_data["input_vector"]
        U = sample_data["U"]
        C0_global = sample_data["C0_input"]

        result = StatTools_bindings.model(input_vector, U, C0_global)

        # Verify result properties
        assert len(result) == len(U)
        assert all(isinstance(x, (float, np.floating)) for x in result)

    def test_backward_compatibility(self, sample_data):
        """Test that new API has same interface and produces valid results like old API"""
        if not (OLD_API_AVAILABLE and NEW_API_AVAILABLE):
            pytest.skip("Both old and new APIs required for comparison")

        input_vector = sample_data["input_vector"]
        U = sample_data["U"]
        C0_input = sample_data["C0_input"]

        # Run old API
        old_result = C_StatTools.get_waiting_time(input_vector, U, C0_input)

        # Run new API
        new_result = StatTools_bindings.get_waiting_time(input_vector, U, C0_input)

        # Both APIs use internal random number generation (get_poisson_thread),
        # so exact numerical comparison is not meaningful. Instead, verify:
        # 1. Same output shape
        assert len(old_result) == len(new_result) == len(U)

        # 2. Both produce valid (non-negative) waiting times
        assert all(x >= 0 for x in old_result)
        assert all(x >= 0 for x in new_result)

        # 3. Both produce finite values
        assert all(np.isfinite(x) for x in old_result)
        assert all(np.isfinite(x) for x in new_result)

        # 4. Waiting times should increase with utilization (general trend)
        # Higher U means more congestion, typically leading to higher wait times
        assert (
            old_result[-1] > old_result[0]
        )  # Last U (0.9) should have higher wait than first (0.5)
        assert new_result[-1] > new_result[0]


class TestIntegration:
    """Integration tests for the complete binding system"""

    def test_module_imports(self):
        """Test that all modules can be imported"""
        modules_available = []

        if OLD_API_AVAILABLE:
            modules_available.append("StatTools.C_StatTools")
        if NEW_API_AVAILABLE:
            modules_available.append("StatTools.StatTools_bindings")

        assert len(modules_available) > 0, "No binding modules available"
        print(f"Available modules: {modules_available}")

    def test_data_type_consistency(self):
        """Test that data types are consistent across bindings"""
        if not NEW_API_AVAILABLE:
            pytest.skip("New API not available")

        test_array = np.array([1.0, 2.0, 3.0])

        # Test cumsum function
        result = StatTools_bindings.cumsum(test_array)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
