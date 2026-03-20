import numpy as np

from StatTools.analysis.dfa import dfa
from StatTools.analysis.dpcca import dpcca
from StatTools.analysis.utils import analyse_cross_ff_linregress
from StatTools.generators import chol2d_mult, generate_fbn

# Create a dataset with Hurst exponent H = 0.8 using the unified interface
hurst = 0.8
length = 2**12

# Generate fractional Brownian noise using the default Kasdin method
sig_1 = generate_fbn(hurst=hurst, length=length)
sig_2 = generate_fbn(hurst=hurst, length=length)
print(sig_2.shape)

s, F_s = dfa(sig_1, degree=2)
ff_params, residuals = analyse_cross_ff_linregress(np.sqrt(F_s), s)
print(f"estiamted Hurst value:{ff_params.slopes[0].value} true Hurst Value: {hurst}")

signal = np.vstack((sig_1, sig_2)).T
des_R0 = 0.89
R0 = np.array([[1.0, des_R0], [des_R0, 1.0]])
print(f"shape of signal:{signal.shape}")
correlated_signal = chol2d_mult(signal, R0)
s_list = [256, 512, 960]
for s_idx, s_val in enumerate(s_list):
    p, r, f, s = dpcca(
        correlated_signal.T,
        pd=1,
        step=1,
        s=s_val,
        time_delays=[-60, -50, -40, 0, 40, 50, 60],
        n_integral=1,
    )
    print(f"correlation matrix for current s value {s_list[s_idx]} with tds-dpcca:{r}")
