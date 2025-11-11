from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from StatTools.analysis.utils import analyse_cross_ff, cross_fcn_sloped, ff_params


def plot_cross_result():
    t = np.linspace(0, 50, num=101, endpoint=True)
    slope_ij_multiple = [1, 6, 3, 1, 6]
    C_ij_multiple = [5, 6, 15, 20]
    R_ij_multiple = [5, 7, 1, 2, 1]
    intercept = 0
    C = [4]
    slope = [2, 1]
    R = [5, 2]
    fig, axs = plt.subplots(1, 2, figsize=(30, 10), sharey=False)
    change_cross_value = partial(cross_fcn_sloped, crossover_amount=1)
    axs[0].axhline(y=0, color="r", linestyle="--", label="y0")
    axs[0].plot(
        t,
        change_cross_value(
            t,
            intercept,
            C_ij_multiple,
            slope_ij_multiple,
            R_ij_multiple,
            crossover_amount=1,
        ),
        label="multiple crossovers",
    )
    axs[0].axhline(y=0, color="b", linestyle="--", label="y0")
    axs[1].plot(
        t,
        change_cross_value(t, 0, C, slope, R, crossover_amount=1),
        label="single crossover",
    )

    plt.plot()
    plt.grid()
    plt.legend()
    plt.xlim(0, 45)
    plt.show()


def plot_ff(
    hs: np.ndarray,
    S: np.ndarray,
    ff_parameter: ff_params,
    residuals=None,
    # title="F(S)",
    ax=None,
):
    # if len(residuals.shape) == 1:
    #     residuals = np.expand_dims(residuals, -1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 10))
    # ax.set_title(title)
    # ax.set_title(title)
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
        ax.plot(
            S,
            fit_func,
            label=r"$F(S)",
        )

    S_new = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    # colors = ["blue", "green", "red", "purple"]
    array_for_limits = [-np.inf] + list(crossovers) + [+np.inf]
    for plot_value in range(len(slopes)):
        current_lim = array_for_limits[plot_value]
        next_lim = array_for_limits[plot_value + 1]
        mask = (S_new > current_lim) & (S_new <= next_lim)
        ax.plot(
            S_new[mask],
            hs[mask],
            ".",
            # color=colors[plot_value],
            label=rf"$H_0(S) \sim {slopes[plot_value]:.2f} \cdot S$",
        )

    # if len(crossovers) > 1:
    #     mask1 = (np.log10(S_new) > cross_log[0]) & (np.log10(S_new) <= cross_log[1])
    #     ax.plot(
    #         S_new[mask1],
    #         hs[mask1],
    #         ".",
    #         color=colors[1],
    #         label=rf"$H_1(S) \sim {slopes[1]:.2f} \cdot S$",
    #     )
    # mask2 = np.log10(S_new) > cross_log[-1]

    # ax.plot(
    #     S_new[mask2],
    #     hs[mask2],
    #     ".",
    #     color=colors[2],
    #     label=rf"$H_2(S) \sim {slopes[2]:.2f}  \cdot S$",
    # )
    for c in ff_parameter.cross:
        ax.axvline(
            c.value, color="k", linestyle="--", label=f"Cross at $S={c.value:.2f}$"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.legend()

    return ax
