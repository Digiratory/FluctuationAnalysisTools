"""Manual test for fluctuation function visualization.

This script generates synthetic fluctuation function data with multiple Hurst coefficients
and crossover points, adds noise, fits the model using analyse_cross_ff, and visualizes
the results using plot_ff. The left plot shows the data with the true parameters, and the
right plot shows the fitted model with residuals.
"""

import matplotlib.pyplot as plt
import numpy as np

from StatTools.analysis.utils import (
    analyse_cross_ff,
    cross_fcn_sloped,
    ff_params,
    var_estimation,
)
from StatTools.visualization.ff_plot import ff_plot

tst_s = np.array(
    [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
)

y_0 = 0
cross = np.array([5, 50])
cross_log = list(np.log10(cross))
slope = [1.5, 1, 0.5]
r = [1, 2.7]
all_values = cross_log + slope + r
tst_h_multiple = 10 ** cross_fcn_sloped(
    np.log10(tst_s), y_0, *all_values, crossover_amount=len(cross)
)


cross_error = [0, 0]
slope_error = [0, 0, 0]
r_error = [0, 0]

cross_list = [
    var_estimation(value=10**v, stderr=e) for v, e in zip(cross_log, cross_error)
]
slopes_list = [var_estimation(value=v, stderr=e) for v, e in zip(slope, slope_error)]
ridigity_list = [var_estimation(value=v, stderr=e) for v, e in zip(r, r_error)]

ff_params_new = ff_params(
    intercept=var_estimation(value=y_0, stderr=0),
    cross=cross_list,
    slopes=slopes_list,
    ridigity=ridigity_list,
)
tst_hr_multiple_approx = 1 + np.random.normal(0, 0.01, (20, len(tst_h_multiple)))
tst_h_multiple = tst_hr_multiple_approx * tst_h_multiple


fig, axs = plt.subplots(1, 2)
ff_plot(tst_h_multiple, tst_s, ff_params_new, ax=axs[0])
ff_parameters_approx, residuals_approx = analyse_cross_ff(
    tst_h_multiple, tst_s, crossover_amount=len(cross)
)


ff_plot(
    tst_h_multiple,
    tst_s,
    ff_parameter=ff_parameters_approx,
    residuals=residuals_approx,
    ax=axs[1],
)

plt.show()
