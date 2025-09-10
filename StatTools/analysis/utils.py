import numpy as np
from scipy.optimize import curve_fit


class Crossover:
    def __init__(self):
        """
        cross-slope scale,
        slope_l-slope before crossover
        slope_h -slope after crossover
        ridigity-ridigity of slope
        stderr- std
        """
        self.cross = None
        self.slope_l = None
        self.slope_h = None
        self.rigidity = None
        self.intercept = None

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
        """
        Invert Relu type function
        """
        return (
            np.log(1 + np.exp(R * (x - C))) / R * (-(x - C)) / np.sqrt(1 + (x - C) ** 2)
        )

    def tf(self, x, R, C1, C2):
        """
        Step function
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

        def fit_func(x, cross, slope_l, slope_h, rigidity):
            """
            Сreates linear regression's model
            """
            return self.single_cross_fcn(x, cross, slope_l, slope_h, rigidity)

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

        cross_log, self.slope_l, self.slope_h, self.rigidity = params

        self.stderr_cross = np.sqrt(pcov[1, 1]) * np.log(10) * 10**cross_log
        self.stderr_slope_l = np.sqrt(pcov[2, 2])
        self.stderr_slope_h = np.sqrt(pcov[3, 3])
        self.stderr_rigidity = np.sqrt(pcov[4, 4])

        self.cross = 10**cross_log
        return params, self
