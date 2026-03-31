import numpy as np
import pytest
from scipy import signal, stats

from StatTools.analysis.dpcca import dpcca, tds_dpcca_worker
from StatTools.generators import generate_fbn
from StatTools.generators.multi_scale_fractional_generator import chol2d_mult

testdata = [
    (1.0),
    (1.25),
    (1.5),
    (1.7),
]


@pytest.fixture(scope="module")
def sample_signal():
    length = 2**13
    signals = {}
    for h in testdata:
        z = np.random.normal(size=length * 2)
        # B = (h - 0.5) * np.arange(1, length*2)**(h - 1.5)
        # sig = signal.lfilter(B, 1, z)
        # sig = sig[length//2: length//2+length]
        # assert sig.shape[0] == length
        # signals[h] = sig
        beta = 2 * h - 1
        L = length * 2
        A = np.zeros(length * 2)
        A[0] = 1
        for k in range(1, L):
            A[k] = (k - 1 - beta / 2) * A[k - 1] / k

        if h == 0.5:
            Z = z
        else:
            Z = signal.lfilter(1, A, z)
        signals[h] = Z

    return signals


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("h", testdata)
def test_dpcca_default(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]

    _, _, f1, s1 = dpcca(sig, 2, step, s, processes=threads)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))

    # plt.loglog(s1, np.sqrt(f1))
    # plt.grid()
    # plt.show()
    assert res.slope == pytest.approx(h, 0.1)


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("h", testdata)
def test_dpcca_default_buffer(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]

    _, _, f1, s1 = dpcca(sig, 2, step, s, processes=threads, buffer=True)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h, 0.1)


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("h", testdata)
def test_dpcca_cumsum_0_default(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]

    sig = np.cumsum(sig, axis=0)
    _, _, f1, s1 = dpcca(sig, 2, step, s, processes=threads, n_integral=0)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h, 0.1)


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("h", testdata)
def test_dpcca_cumsum_0_buffer(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]

    sig = np.cumsum(sig, axis=0)
    _, _, f1, s1 = dpcca(sig, 2, step, s, processes=threads, buffer=True, n_integral=0)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h, 0.1)


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("h", testdata)
def test_dpcca_cumsum_2_default(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]
    _, _, f1, s1 = dpcca(sig, 2, step, s, processes=threads, n_integral=2)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h + 1, 0.1)


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("h", testdata)
def test_dpcca_cumsum_2_buffer(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]
    _, _, f1, s1 = dpcca(sig, 2, step, s, processes=threads, buffer=True, n_integral=2)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h + 1, 0.1)


hurst_values = [0.25, 0.5, 0.75]
test_lags = [0, 1, 2, 3]


@pytest.fixture(scope="module")
def create_signals():
    length = 2**15
    signals = {}
    true_lag = 40
    for h in hurst_values:
        z = np.random.normal(size=length + 2 * true_lag)
        beta = 2 * h - 1
        L = len(z)
        A = np.zeros(L)
        A[0] = 1
        for k in range(1, L):
            A[k] = (k - 1 - beta / 2) * A[k - 1] / k

        if h == 0.5:
            Z_full = z
        else:
            Z_full = signal.lfilter(1, A, z)

        x = Z_full[2 * true_lag :]  # лидер самый первый
        y = Z_full[true_lag:-true_lag]
        z = Z_full[: -2 * true_lag]  # самый последний
        signals[h] = np.vstack([x, y, z])
    return signals


@pytest.mark.parametrize("h", hurst_values)
# @pytest.mark.parametrize("lag", test_lags)
def test_tds_dpcca_comparison_signals_worker(create_signals, h):
    arr = create_signals[h]
    true_lag_01 = 40
    true_lag_12 = 40
    true_lag_02 = 80
    _, r, _ = tds_dpcca_worker(
        s=[256, 512, 1024],
        arr=arr,
        step=10,
        pd=1,
        time_delays=[
            -90,
            -80,
            -70,
            -60,
            -50,
            -40,
            -30,
            -20,
            -10,
            0,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
        ],
        n_integral=1,
    )
    time_delays = [
        -90,
        -80,
        -70,
        -60,
        -50,
        -40,
        -30,
        -20,
        -10,
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
    ]
    s = [256, 512, 1024]
    for s_idx, s_val in enumerate(s):
        correlation_01 = r[:, s_idx, 0, 1]
        max_lag_idx_01 = np.argmax(correlation_01)
        estimated_lag_01 = time_delays[max_lag_idx_01]
        correlation_12 = r[:, s_idx, 1, 2]
        max_lag_idx_12 = np.argmax(correlation_12)
        estimated_lag_12 = time_delays[max_lag_idx_12]
        correlation_02 = r[:, s_idx, 0, 2]
        max_lag_idx_02 = np.argmax(correlation_02)
        estimated_lag_02 = time_delays[max_lag_idx_02]
        assert abs(estimated_lag_01 - true_lag_01) <= 1
        assert abs(estimated_lag_12 - true_lag_12) <= 1
        assert abs(estimated_lag_02 - true_lag_02) <= 1
