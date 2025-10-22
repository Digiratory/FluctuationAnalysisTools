import matplotlib.pyplot as plt
import numpy as np

from StatTools.analysis.utils import (
    analyse_ff,
    analyse_multiple_cross_ff,
    multiple_cross_fcn_sloped,
    scalar_for_fitting,
    single_cross_fcn,
    single_cross_fcn_sloped,
)


def plot_single_cross_result():
    intercept = 0
    C_12 = 4
    slope_1 = 2
    slope_2_1 = 1
    slope_2_2 = 4
    R_1 = 5
    R_2 = 2
    x = np.linspace(0, 50, num=101, endpoint=True)
    y_1 = single_cross_fcn_sloped(x, intercept, C_12, slope_1, slope_2_1, R_1)
    y_2 = single_cross_fcn_sloped(x, intercept, C_12, slope_1, slope_2_2, R_2)

    fig, ax = plt.subplots()
    plt.plot(x, y_1, label="$R_{12} = 5$")
    plt.plot(x, y_2, label="$R_{12} = 1$")
    plt.grid(which="both")
    plt.legend()
    plt.show()


def plot_multiple_cross_result():
    t = np.linspace(0, 45, num=101, endpoint=True)
    y = np.linspace(-20, 100, num=101, endpoint=True)
    slope_ij = [1, 6, 3, 1, 6]
    C_ij = [5, 6, 15, 20]
    R_ij = [5, 7, 1, 2, 1]
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 5)
    plt.subplot(1, 1, 1)
    plt.axhline(y=0, color="r", linestyle="--", label="y0")
    plt.plot(
        t,
        multiple_cross_fcn_sloped(t, 0, C_ij, slope_ij, R_ij),
        label="sum of func sloped",
    )
    plt.grid()
    plt.legend()
    plt.xlim(0, 45)
    plt.show()


def analyse_plot_ff(hs: np.ndarray, S: np.ndarray, title="F(S)"):
    """
    Analyse plot that shows comparison
    linear regression's model's data plot and real data: F(s)~s where F(s)-fluctuation functions
    """
    params, residuals = analyse_ff(hs, S)
    if len(residuals.shape) == 1:
        residuals = np.expand_dims(residuals, -1)
    cross = params.cross.value
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    ax.errorbar(
        S,
        10
        ** (
            single_cross_fcn(
                np.log10(S),
                params.intercept.value,
                np.log10(params.cross.value),
                params.slope_l.value,
                params.slope_h.value,
                params.ridigity.value,
            )
        ),
        fmt="g--",
        capsize=7,
        yerr=2 * np.std(residuals, axis=0),
        label=f"$F(S) \pm 2 \sigma$",
    )
    S = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    ax.plot(
        S[S <= cross],
        hs[S <= cross],
        ".",
        color="blue",
        label=f"$H_l(S) \sim {params.slope_l.value:.2f} \pm {2*params.slope_l.stderr:.2f} S$",
    )
    ax.plot(
        S[S > cross],
        hs[S > cross],
        ".",
        color="red",
        label=f"$H_h(S)  \sim {params.slope_h.value:.2f} \pm {2*params.slope_h.stderr:.2f} S$",
    )
    ax.axvline(
        params.cross.value,
        label=f"Cross, $S = {params.cross.value:.2f} \pm {params.cross.stderr:.2f}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.legend()
    plt.plot()
    plt.show()


def analyse_multiple_plot_ff(hs: np.ndarray, S: np.ndarray, title="F(S)"):
    params, residuals = analyse_multiple_cross_ff(hs, S)
    if len(residuals.shape) == 1:
        residuals = np.expand_dims(residuals, -1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    y0 = params.multiple_intercept.multiple_value
    crossovers = params.multiple_cross.multiple_value
    slope_current = params.multiple_slope_current.multiple_value
    R = params.multiple_ridigity.multiple_value
    fit_func = 10 ** multiple_cross_fcn_sloped(
        np.log10(S), y0, np.log10(crossovers), slope_current, R
    )
    ax.errorbar(
        S,
        fit_func,
        fmt="g--",
        capsize=7,
        yerr=2 * np.std(residuals, axis=0),
        label=r"$F(S) \pm 2\sigma$",
    )

    S_new = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    colors = ["blue", "green", "red", "purple"]
    cross_log = np.log10(crossovers)
    mask = np.log10(S_new) <= cross_log[0]
    ax.plot(
        S_new[mask],
        hs[mask],
        ".",
        color=colors[0],
        label=rf"$H_0(S) \sim {slope_current[0]:.2f} \pm {2*params.multiple_slope_current.multiple_stderr[0]:.2f} \cdot S$",
    )
    if len(crossovers) > 1:
        mask1 = (np.log10(S_new) > cross_log[0]) & (np.log10(S_new) <= cross_log[1])
        ax.plot(
            S_new[mask1],
            hs[mask1],
            ".",
            color=colors[1],
            label=rf"$H_1(S) \sim {slope_current[1]:.2f} \pm {2*params.multiple_slope_current.multiple_stderr[1]:.2f} \cdot S$",
        )
    mask2 = np.log10(S_new) > cross_log[-1]
    ax.plot(
        S_new[mask2],
        hs[mask2],
        ".",
        color=colors[2],
        label=rf"$H_2(S) \sim {slope_current[2]:.2f} \pm {2*params.multiple_slope_current.multiple_stderr[2]:.2f} \cdot S$",
    )
    for c in crossovers:
        ax.axvline(c, color="k", linestyle="--", label=f"Cross at $S={c}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.legend()
    plt.show()


tst_s = np.array(
    [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
)
tst_one_cross = 4
tst_h = 10 ** single_cross_fcn(np.log10(tst_s), 0, np.log10(tst_one_cross), 1, 0.5, 1)
tst_hr = 1 + np.random.normal(0, 0.3, (20, len(tst_h)))
tst_hr *= tst_h
analyse_plot_ff(tst_hr, tst_s)
cross1_multiple = 5
cross2_multiple = 6
slope1_miltiple = 1
slope2_miltiple = 4
slope3_multiple = 3
r1_miltiple = 2
r2_multiple = 3
r3_multiple = 1
tst_h_multiple = 10 ** scalar_for_fitting(
    np.log10(tst_s),
    0,
    np.log10(cross1_multiple),
    np.log10(cross2_multiple),
    slope1_miltiple,
    slope2_miltiple,
    slope3_multiple,
    r1_miltiple,
    r2_multiple,
    r3_multiple,
)
tst_hr_multiple = 1 + np.random.normal(0, 0.3, (20, len(tst_h_multiple)))
tst_hr_multiple *= tst_h_multiple
analyse_multiple_plot_ff(tst_hr_multiple, tst_s)
