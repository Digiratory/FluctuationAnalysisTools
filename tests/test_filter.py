import os

import numpy as np
import pytest

from StatTools.analysis.dfa import DFA
from StatTools.generators.base_filter import Filter

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
if IN_GITHUB_ACTIONS:
    h_list = [0.5, 1]
    target = [0.7]
else:
    h_list = np.arange(0.5, 1, 0.1)
    target = [0.5, 0.7, 0.9]

testdata = {
    "target_mean": target,
    "target_std": target,
    "h": h_list,
    "length": [6000],
}

SCALES = np.array([2**i for i in range(3, 9)])
TARGET_LEN = 2000


@pytest.mark.parametrize("h", testdata["h"])
@pytest.mark.parametrize("length", testdata["length"])
@pytest.mark.parametrize("target_std", testdata["target_std"])
@pytest.mark.parametrize("target_mean", testdata["target_mean"])
def test_filter_generator(h, length, target_std, target_mean):
    """
    Test that the generated data has the specified mean and standard deviation.
    """
    generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
    mean_difference_mean = 0
    mean_difference_std = 0
    mean_difference_h = 0
    times = 5
    for _ in range(times):
        trajectory = list(generator)
        mean_difference_mean += np.mean(trajectory)
        mean_difference_std += np.std(trajectory, ddof=1)
        mean_difference_h += DFA(trajectory).find_h()

    assert (
        abs(mean_difference_mean / times - target_mean) < 0.001
    ), f"Mean deviation too large: expected {target_mean}, got {mean_difference_mean}"
    assert (
        abs(mean_difference_std / times - target_std) < 0.001
    ), f"Std deviation too large: expected {target_std}, got {mean_difference_std}"
    assert abs(mean_difference_h / times - h) < (
        h * 0.15
    ), f"Hurst deviation too large: expected {h}, got {mean_difference_h}"


@pytest.mark.parametrize("h", testdata["h"])
@pytest.mark.parametrize("length", testdata["length"])
@pytest.mark.parametrize("target_std", testdata["target_std"])
@pytest.mark.parametrize("target_mean", testdata["target_mean"])
def test_filter(h, length, target_std, target_mean):
    """
    Test that the generated data has the specified mean and standard deviation.
    """
    generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
    mean_difference_mean = 0
    mean_difference_std = 0
    mean_difference_h = 0
    times = 3
    for _ in range(times):
        trajectory = generator.generate(n_vectors=1)
        mean_difference_mean += np.mean(trajectory)
        mean_difference_std += np.std(trajectory, ddof=1)
        mean_difference_h += DFA(trajectory).find_h()

    assert (
        abs(mean_difference_mean / times - target_mean) < 0.001
    ), f"Mean deviation too large: expected {target_mean}, got {mean_difference_mean}"
    assert (
        abs(mean_difference_std / times - target_std) < 0.001
    ), f"Std deviation too large: expected {target_std}, got {mean_difference_std}"
    assert abs(mean_difference_h / times - h) < (
        h * 0.15
    ), f"Hurst deviation too large: expected {h}, got {mean_difference_h}"


@pytest.mark.parametrize("h", testdata["h"])
@pytest.mark.parametrize("length", testdata["length"])
@pytest.mark.parametrize("target_std", testdata["target_std"])
@pytest.mark.parametrize("target_mean", testdata["target_mean"])
def test_filter_2d(h, length, target_std, target_mean):
    """
    Test that the generated data has the specified mean and standard deviation.
    """
    count = 3
    generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
    trajectories = generator.generate(n_vectors=count)

    mean_difference_mean = 0
    mean_difference_std = 0
    mean_difference_h = 0

    for i in range(count):
        trajectory = trajectories[i]

        mean_difference_mean += np.mean(trajectory)
        mean_difference_std += np.std(trajectory, ddof=1)
        mean_difference_h += DFA(trajectory).find_h()

    assert (
        abs(mean_difference_mean / count - target_mean) < 0.001
    ), f"Mean deviation too large: expected {target_mean}, got {mean_difference_mean}"
    assert (
        abs(mean_difference_std / count - target_std) < 0.001
    ), f"Std deviation too large: expected {target_std}, got {mean_difference_std}"
    assert abs(mean_difference_h / count - h) < (
        h * 0.15
    ), f"Hurst deviation too large: expected {h}, got {mean_difference_h}"
