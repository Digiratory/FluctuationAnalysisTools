"""Manual test for fluctuation function with zero crossovers visualization.

This script generates synthetic fluctuation function data with one Hurst coefficient
and zero crossover points, adds noise, fits the model using analyse_zero_cross_ff, and visualizes
the results using plot_ff.
"""

import matplotlib.pyplot as plt
import numpy as np

from StatTools.analysis.utils import analyse_zero_cross_ff
from StatTools.visualization.ff_plot import ff_plot

tst_s = np.array(
    [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
)

y_0 = 0
cross_single_crossover = np.array([])
slope_single_crossover = 1.5

tst_h_multiple_single_crossover = 10 ** (slope_single_crossover * np.log10(tst_s) + y_0)


cross_error = [0, 0]
slope_error = [0, 0, 0]
r_error = [0, 0]

tst_hr_multiple_approx = 1 + np.random.normal(
    0, 0.01, (20, len(tst_h_multiple_single_crossover))
)
tst_h_multiple = tst_hr_multiple_approx * tst_h_multiple_single_crossover


fig, axs = plt.subplots()
ff_parameters_approx, residuals_approx = analyse_zero_cross_ff(tst_h_multiple, tst_s)


ff_plot(
    tst_h_multiple,
    tst_s,
    ff_parameter=ff_parameters_approx,
    residuals=residuals_approx,
)

plt.show()
