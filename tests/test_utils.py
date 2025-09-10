import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def win_generator(base, s_min, L):
    """
    Generates windows for DFA analyse
    """
    windows = []
    s_max = L / 4
    for degree in range(
        int(math.log2(s_min) / math.log2(base)), int(math.log2(s_max) / math.log2(base))
    ):
        new = int(base**degree)
        if not new in windows:
            windows.append(new)
    return windows


class Crossover:
    def __init__(self):
        """
        cross-slope scale,
        slope_l-slope before crossover
        slope_h -slope after crossover
        ridigity-ridigity of slope
        stderr- std
        intercept- vertical displacement
        """
        self.intercept = None
        self.cross = None
        self.slope_l = None
        self.slope_h = None
        self.rigidity = None

        self.stderr_intercept = None
        self.stderr_cross = None
        self.stderr_slope_l = None
        self.stderr_slope_h = None
        self.stderr_rigidity = None
        self.S = None
        self.hs = None

    def f_fcn(self, x, R, C):
        """
        ReLu type function with ridigity-R, C-slope scale
        """
        return np.log(1 + np.exp(R * (x - C))) / R * (x - C) / np.sqrt(1 + (x - C) ** 2)

    def rev_f_fcn(self, x, R, C):
        return (
            np.log(1 + np.exp(R * (x - C))) / R * (-(x - C)) / np.sqrt(1 + (x - C) ** 2)
        )

    def tf(self, x, R, C1, C2):
        """
        Creates function with two slopes
        """
        return -self.f_fcn(x, R, C2) - self.rev_f_fcn(x, R, C1)

    def single_cross_fcn(self, x, y_0, C_12, slope_1, slope_2, R_12):
        """
        Creates piesewise linear function with slopes definition
        """
        return (
            y_0
            + slope_1 * self.tf(x, R_12, -100, C_12)
            + slope_2 * self.tf(x, R_12, C_12, 100)
        )

    def analyse_dfa(self, hs, S):
        """
        Analyses real data: F(s) and s and simulated data with linear regression's model.
        """
        self.hs = hs
        self.S = S
        s = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T

        x_data = np.log10(s.flatten())
        y_data = np.log10(hs.flatten())

        def fit_func(x, intercept, cross, slope_l, slope_h, rigidity):
            """
            Сreates linear regression's model with single_cross_fcn function
            """
            return self.single_cross_fcn(
                x, intercept, cross, slope_l, slope_h, rigidity
            )

        p0 = (0, np.log10(S[len(S) // 2]), 1, 1, 5)
        bounds = (
            [-np.inf, np.log10(S[0]), 0, 0, 1],
            [+np.inf, np.log10(S[-1]), 5, 5, +np.inf],
        )

        try:
            params, pcov = curve_fit(
                f=fit_func,
                xdata=x_data,
                ydata=y_data,
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )
        except Exception as e:
            raise RuntimeError(f"curve_fit failed: {e}")

        self.intercept, cross_log, self.slope_l, self.slope_h, self.rigidity = params

        self.stderr_intercept = np.sqrt(pcov[0, 0])
        self.stderr_cross = np.sqrt(pcov[1, 1]) * np.log(10) * 10**cross_log
        self.stderr_slope_l = np.sqrt(pcov[2, 2])
        self.stderr_slope_h = np.sqrt(pcov[3, 3])
        self.stderr_rigidity = np.sqrt(pcov[4, 4])

        self.cross = 10**cross_log

        model = 10 ** fit_func(np.log10(S), *params)

        return self

    def plot_results(self):
        """
        Compares linear regression's model's data plot, real data: F(s)~s where F(s)-fluctuation functions
        and F(s)~s where F(s)-one array with mean from al fluctuation functions
        """
        if self.S is None or self.hs is None:
            raise ValueError("No plot")

        plt.figure(figsize=(12, 8))
        plt.loglog(self.S, self.hs.T, ".", color="gray", alpha=0.5, label="F(S)")
        plt.loglog(
            self.S, np.mean(self.hs, axis=0), "o", color="blue", label="Mean F(S)"
        )

        model = 10 ** self.single_cross_fcn(
            np.log10(self.S),
            self.intercept,
            np.log10(self.cross),
            self.slope_l,
            self.slope_h,
            self.rigidity,
        )
        plt.loglog(self.S, model, "r-", linewidth=2, label="Fit")

        plt.axvline(
            self.cross, color="red", linestyle="--", label=f"Cross: {self.cross:.2f}"
        )

        plt.xlabel("S")
        plt.ylabel("F(S)")
        plt.title("DFA Analysis with Crossover")
        plt.legend()
        plt.grid(True, which="both", alpha=0.5)
        plt.show()


if __name__ == "__main__":
    path = Path("/Users/majagavricenkova/Desktop/Александр проект/pigeons_repo/")
    path_res = path / "empirica/fluctuation_functions/csv"
    S = np.loadtxt(path_res / "S_hf1.csv", delimiter=";")
    print(f"S shape: {S.shape}")

    F_S = np.loadtxt(path_res / "hf1_X.csv", delimiter=";", skiprows=1)
    print(f"F(S) shape: {F_S.shape}")

    if F_S.ndim == 1:
        F_S = F_S.reshape(1, -1)
    elif F_S.shape[0] == len(S):
        F_S = F_S.T
    else:
        raise ValueError(f"Cannot match F_S shape {F_S.shape} with S length {len(S)}")

    analyzer = Crossover()
    analyzer.analyse_dfa(F_S, S)
    analyzer.plot_results()
