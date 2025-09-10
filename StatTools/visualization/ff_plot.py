import matplotlib.pyplot as plt
import numpy as np
from analysis import Crossover


def plot_results():
    """
    Compares linear regression's model's data plot, real data: F(s)~s where F(s)-fluctuation functions
    and F(s)~s where F(s)-one array with mean from al fluctuation functions
    """
    crossover = Crossover()
    S = []
    hs = []
    cross, slope_l, slope_h, ridigity = crossover.analyse_dfa()
    if S is None or hs is None:
        raise ValueError("No plot")

    plt.figure(figsize=(12, 8))
    plt.loglog(S, hs.T, ".", color="gray", alpha=0.5, label="F(S)")
    plt.loglog(S, np.mean(hs, axis=0), "o", color="blue", label="Mean F(S)")

    model = 10 ** crossover.single_cross_fcn(
        np.log10(S), np.log10(cross), slope_l, slope_h, ridigity
    )
    plt.loglog(S, model, "r-", linewidth=2, label="Fit")

    plt.axvline(cross, color="red", linestyle="--", label=f"Cross: {cross:.2f}")

    plt.xlabel("S")
    plt.ylabel("F(S)")
    plt.title("DFA Analysis with Crossover")
    plt.legend()
    plt.grid(True, which="both", alpha=0.5)
    plt.show()
