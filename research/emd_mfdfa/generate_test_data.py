"""
Test data generation script for EMD-MFDFA experiments.

This script generates synthetic time series (fractional Gaussian noise)
with known Hurst exponents for validating the EMD-based MFDFA implementation.

Usage:
    python generate_test_data.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from StatTools.generators.kasdin_generator import create_kasdin_generator


def generate_test_dataset(output_dir: str = "test_data"):
    """
    Generate a complete test dataset with various parameters.

    Generates fGn series for:
    - Different Hurst exponents: 0.2, 0.5, 0.8
    - Different lengths: 2^10, 2^12, 2^14

    Args:
        output_dir: Directory to save generated data
    """
    # Create output directory
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True, parents=True)

    # Test parameters
    hurst_values = [0.2, 0.5, 0.8]
    lengths = [2**10, 2**12, 2**14]

    print("=" * 60)
    print("Generating Test Dataset for EMD-MFDFA")
    print("=" * 60)

    # Generate data for each combination
    for h in hurst_values:
        for length in lengths:
            print(f"\nGenerating: H={h:.1f}, N={length}")

            # Generate signal using Kasdin method
            np.random.seed(42)
            generator = create_kasdin_generator(h=h, length=length, normalize=True)
            signal = generator.get_full_sequence()

            # Save to file
            filename = f"fgn_H{h:.1f}_N{length}.txt"
            filepath = output_path / filename
            np.savetxt(filepath, signal)

            print(f"  Saved to: {filepath}")
            print(f"  Mean: {np.mean(signal):.6f}")
            print(f"  Std: {np.std(signal):.6f}")
            print(f"  Min: {np.min(signal):.3f}, Max: {np.max(signal):.3f}")

    print("\n" + "=" * 60)
    print(f"Dataset generation complete!")
    print(f"Files saved in: {output_path}")
    print("=" * 60)


def plot_sample_series(save_dir: str = "figures"):
    """
    Plot sample time series for visual inspection.

    Args:
        save_dir: Directory to save figures
    """
    # Create figures directory
    fig_path = Path(__file__).parent / save_dir
    fig_path.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 60)
    print("Generating Sample Plots")
    print("=" * 60)

    # Parameters
    length = 2**10
    hurst_values = [0.2, 0.5, 0.8]

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        "Fractional Gaussian Noise with Different Hurst Exponents",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (h, ax) in enumerate(zip(hurst_values, axes)):
        # Generate signal
        np.random.seed(42)
        generator = create_kasdin_generator(h=h, length=length, normalize=True)
        signal = generator.get_full_sequence()

        # Plot
        ax.plot(signal, linewidth=0.5, alpha=0.8)
        ax.set_title(
            f'H = {h:.1f} ({"Anti-persistent" if h < 0.5 else "Random walk" if h == 0.5 else "Persistent"})',
            fontsize=12,
        )
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"Mean: {np.mean(signal):.3f}, Std: {np.std(signal):.3f}"
        ax.text(
            0.02,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    # Save figure
    output_file = fig_path / "sample_fgn_series.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_file}")

    plt.close()


def test_generator_basic():
    """
    Basic test to verify generator is working correctly.
    """
    print("\n" + "=" * 60)
    print("Testing Basic Generator Functionality")
    print("=" * 60)

    try:
        # Test with H=0.7, short sequence
        print("\nTest 1: Generating fGn with H=0.7, N=256")
        np.random.seed(123)
        generator = create_kasdin_generator(h=0.7, length=256, normalize=True)
        signal = generator.get_full_sequence()
        print(f"  Success. Shape: {signal.shape}")
        print(f"    Mean: {np.mean(signal):.6f}")
        print(f"    Std: {np.std(signal):.6f}")

        # Test with different H values
        print("\nTest 2: Multiple H values")
        for h in [0.3, 0.5, 0.8]:
            np.random.seed(456)
            generator = create_kasdin_generator(h=h, length=512, normalize=True)
            signal = generator.get_full_sequence()
            print(f"  H={h:.1f}: Generated {len(signal)} points")

        print("\nAll basic tests passed")
        return True

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run basic tests first
    if not test_generator_basic():
        print("\nBasic tests failed. Please fix errors before generating full dataset.")
        sys.exit(1)

    # Generate full test dataset
    generate_test_dataset()

    # Plot sample series
    plot_sample_series()

    print("\nAll done")
