import numpy as np

from StatTools.experimental.analysis.tools import get_extra_h_dfa

def adjust_hurst_to_range(signal: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Adjusts the Hurst exponent of a signal to fall within the range [0.5, 1.5]
    using cumulative sums and differences.
    
    The function uses mathematical transformations:
    - Cumulative sum (np.cumsum) increases the Hurst exponent by ~1.
    - Differencing (np.diff) decreases the Hurst exponent by ~1.
    
    Parameters:
        signal (array_like): Input 1D signal whose Hurst exponent needs adjustment
        
    Returns:
        adjusted_signal (ndarray): Signal with Hurst exponent adjusted to the target range
        applied_steps (int): Number of transformation steps applied:
            - Positive: number of cumulative sums applied (Hurst increased)
            - Negative: number of differences applied (Hurst decreased)
            - Zero: no transformation needed
        
    Notes:
        - For H > 1.5: applies differencing to reduce Hurst exponent.
        - For H < 0.5: applies cumulative sums to increase Hurst exponent.
        - The transformation follows approximate relationship: 
          H_transformed ≈ H_original + applied_steps.
    """
    h = get_extra_h_dfa(signal)
    adjusted_signal = signal
    applied_steps = 0
    effective_h = h
    if h > 1.5:
        while effective_h > 1.5:
            applied_steps -= 1
            effective_h -= 1
    elif h < 0.5:
        while effective_h < 0.5:
            applied_steps += 1
            effective_h += 1

    if applied_steps < 0:
        for _ in range(abs(applied_steps)):
            adjusted_signal = np.diff(adjusted_signal)

    elif applied_steps > 0:
        for _ in range(applied_steps):
            adjusted_signal = np.cumsum(adjusted_signal)
            
    return adjusted_signal, applied_steps


def reverse_hurst_adjustment(adjusted_signal: np.ndarray, applied_steps: int) -> np.ndarray:
    """
    Reverses the Hurst exponent adjustment applied by adjust_hurst_to_range().
    
    Applies the inverse transformations to recover the original signal
    with its initial Hurst exponent.
    
    Parameters:
        adjusted_signal (array_like): Signal that was previously processed by adjust_hurst_to_range().
        applied_steps (int): Number of transformation steps that were originally applied.
        
    Returns:
        original_signal (ndarray): Signal with the original Hurst exponent restored.
        
    Notes:
        - If applied_steps was negative (differencing applied originally),
          applies cumulative sums to reconstruct the signal.
        - If applied_steps was positive (cumulative sums applied originally),
          applies differencing to reconstruct the signal.
        - The length of the returned signal may differ from the original if
          multiple differencing operations were applied.
    """
    if applied_steps < 0:
        for _ in range(abs(applied_steps)):
            adjusted_signal = np.cumsum(adjusted_signal)

    elif applied_steps > 0:
        for _ in range(applied_steps):
            adjusted_signal = np.diff(adjusted_signal)
            
    return adjusted_signal