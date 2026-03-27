"""Tests for pertutbations module."""

import os

import numpy as np
import pytest

from StatTools.experimental.augmentation.perturbations import add_exponential_gaps

SEED = 42
N = 2**16

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
if IN_GITHUB_ACTIONS:
    _CASES = [
        #  rq    fg    tol   (l_avg)
        (8, 0.2, 0.1),  # 1.6
        (16, 0.2, 0.1),  # 3.2
        (32, 0.2, 0.1),  # 6.4
    ]
else:
    _CASES = [
        #  rq    fg    tol   (l_avg)
        (8, 0.1, 0.1),  # 0.8
        (8, 0.2, 0.1),  # 1.6
        (8, 0.3, 0.1),  # 2.4
        (16, 0.1, 0.1),  # 1.6
        (16, 0.2, 0.1),  # 3.2
        (16, 0.3, 0.1),  # 4.8
        (32, 0.1, 0.1),  # 3.2
        (32, 0.2, 0.1),  # 6.4
        (32, 0.3, 0.1),  # 9.6
    ]


@pytest.mark.parametrize("rq,fg,tol", _CASES)
def test_add_exponential_gaps_distribution(rq, fg, tol):
    """NaN fraction, mean interval and mean gap length stay within tol of theoretical values."""
    np.random.seed(SEED)
    for _ in range(50):
        result, gaps = add_exponential_gaps(np.ones(N), rq=rq, fg=fg)

        starts = np.array([s for s, e in gaps])
        ends = np.array([e for s, e in gaps])
        inter_gap = np.diff(starts)
        lengths = ends - starts
        real_fg = 1 - np.exp(-fg)
        l_avg = fg * rq

        assert abs(np.isnan(result).mean() - real_fg) / real_fg < tol
        assert abs(np.mean(inter_gap) - rq) / rq < tol
        assert abs(np.mean(lengths) - l_avg) / l_avg < tol
