from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from StatTools.analysis.utils import analyse_cross_ff, cross_fcn_sloped, ff_params


def plot_ff(
    hs: np.ndarray,
    S: np.ndarray,
    ff_parameter: ff_params,
    residuals=None,
    ax=None,
    title=None,
):
    """Plots the fluctuation function with fitted parameters and crossover points.

    This function visualizes the fluctuation function data along with the fitted model,
    including error bars if residuals are provided, and marks the crossover points.

    Args:
        hs (np.ndarray): The dependent data array, length M.
        S (np.ndarray): The independent variable array, shape (k, M)
        ff_parameter (ff_params): Fitted parameters from the fluctuation function analysis.
        residuals (np.ndarray, optional): Residuals for plotting error bars. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, creates a new figure. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The matplotlib axis containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 10))
    slopes = [slp.value for slp in ff_parameter.slopes]
    crossovers = [cross.value for cross in ff_parameter.cross]
    R = [r.value for r in ff_parameter.ridigity]
    intercept = (ff_parameter.intercept.value,)
    all_values = [np.log10(c) for c in crossovers] + slopes + R
    fit_func = 10 ** cross_fcn_sloped(
        np.log10(S),
        intercept,
        *all_values,
        crossover_amount=len(crossovers),
    )

    if residuals is not None:
        ax.errorbar(
            S,
            fit_func,
            fmt="g--",
            capsize=7,
            yerr=2 * np.std(residuals, axis=0),
            label=r"$F(S) \pm 2\sigma$",
        )
    else:
        ax.plot(S, fit_func, label=r"$F(S)")

    S_new = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    array_for_limits = [-np.inf] + list(crossovers) + [+np.inf]
    for plot_value in range(len(slopes)):
        current_lim = array_for_limits[plot_value]
        next_lim = array_for_limits[plot_value + 1]
        mask = (S_new > current_lim) & (S_new <= next_lim)
        ax.plot(
            S_new[mask],
            hs[mask],
            ".",
            label=rf"$H_0(S) \sim {slopes[plot_value]:.2f} \cdot S$",
        )

    for c in ff_parameter.cross:
        ax.axvline(
            c.value, color="k", linestyle="--", label=f"Cross at $S={c.value:.2f}$"
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.legend()

    return ax
