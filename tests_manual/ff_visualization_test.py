import matplotlib.pyplot as plt
import numpy as np

from StatTools.analysis.utils import (
    analyse_cross_ff,
    cross_fcn_sloped,
    ff_params,
    var_estimation,
)
from StatTools.visualization.ff_plot import plot_ff

tst_s = np.array(
    [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
)

y_0 = 0
cross = np.array([5])
slope = [1.5, 1.0]
r = [1, 1]
all_values = list(np.log10(cross)) + slope + r
tst_h_multiple = 10 ** cross_fcn_sloped(
    np.log10(tst_s), y_0, *all_values, crossover_amount=len(cross)
)

cross_error = [0, 0]
slope_error = [0, 0, 0]
r_error = [0, 0, 0]
# plt.plot(tst_s, tst_h_multiple)
# plt.loglog()
# plt.show()
cross_list = [var_estimation(value=v, stderr=e) for v, e in zip(cross, cross_error)]
slopes_list = [var_estimation(value=v, stderr=e) for v, e in zip(slope, slope_error)]
ridigity_list = [var_estimation(value=v, stderr=e) for v, e in zip(r, r_error)]

ff_params_new = ff_params(
    intercept=var_estimation(value=y_0, stderr=0),
    cross=cross_list,
    slopes=slopes_list,
    ridigity=ridigity_list,
)
tst_hr_multiple = 1 + np.random.normal(0, 0.3, (20, len(tst_h_multiple)))
tst_h_multiple = tst_hr_multiple * tst_h_multiple

plot_ff(tst_h_multiple, tst_s, ff_params_new)

#
# tst_hr_multiple *= tst_h_multiple
# ff_parameters, residuals = analyse_cross_ff(tst_hr_multiple, tst_s)
# plot_ff(tst_hr_multiple, tst_s,  ff_params_new,residuals)
# plot_ff(tst_hr_multiple, tst_s,  ff_parameters,residuals)
