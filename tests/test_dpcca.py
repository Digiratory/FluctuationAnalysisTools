import numpy as np
import pytest
from scipy import signal, stats

from StatTools.analysis.dpcca import dpcca, tdc_dpcca_worker

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


@pytest.mark.parametrize("h", testdata)
def test_dpcca_default(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]

    p1, r1, f1, s1 = dpcca(sig, 2, step, s, processes=threads)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))

    # plt.loglog(s1, np.sqrt(f1))
    # plt.grid()
    # plt.show()
    assert res.slope == pytest.approx(h, 0.1)


@pytest.mark.parametrize("h", testdata)
def test_dpcca_default_buffer(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]

    p1, r1, f1, s1 = dpcca(sig, 2, step, s, processes=threads, buffer=True)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h, 0.1)


@pytest.mark.parametrize("h", testdata)
def test_dpcca_cumsum_0_default(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]

    sig = np.cumsum(sig, axis=0)
    p1, r1, f1, s1 = dpcca(sig, 2, step, s, processes=threads, n_integral=0)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h, 0.1)


@pytest.mark.parametrize("h", testdata)
def test_dpcca_cumsum_0_buffer(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]

    sig = np.cumsum(sig, axis=0)
    p1, r1, f1, s1 = dpcca(
        sig, 2, step, s, processes=threads, buffer=True, n_integral=0
    )
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h, 0.1)


@pytest.mark.parametrize("h", testdata)
def test_dpcca_cumsum_2_default(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]
    p1, r1, f1, s1 = dpcca(sig, 2, step, s, processes=threads, n_integral=2)
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h + 1, 0.1)


@pytest.mark.parametrize("h", testdata)
def test_dpcca_cumsum_2_buffer(sample_signal, h):
    s = [2**i for i in range(3, 20)]
    step = 0.5
    threads = 4
    sig = sample_signal[h]
    p1, r1, f1, s1 = dpcca(
        sig, 2, step, s, processes=threads, buffer=True, n_integral=2
    )
    f1 = np.sqrt(f1)
    res = stats.linregress(np.log(s1), np.log(f1))
    assert res.slope == pytest.approx(h + 1, 0.1)


hurst_values = [0.3, 0.5, 0.7, 0.9]
test_lags = [-5, -2, 0, 2, 5]


@pytest.fixture(scope="module")
def create_signal_pair():
    length = 2**10
    signals = {}
    for h in hurst_values:
        z = np.random.normal(size=length * 2)  # гауссовский белый шум
        beta = 2 * h - 1  # параметр для генерации  фрактального
        L = length * 2
        A = np.zeros(L)
        A[0] = 1
        for k in range(1, L):
            A[k] = (k - 1 - beta / 2) * A[k - 1] / k  # для регрессии

        if h == 0.5:
            Z = z
        else:
            Z = signal.lfilter(1, A, z)

        Z = Z[1000 : 1000 + length]
        lag = 10
        Z_lag = np.roll(Z, lag)

        Z = Z[lag:-lag]
        Z_lag = Z_lag[lag:-lag]

        signals[h] = np.vstack(
            [Z, Z_lag]
        )  # общая матрица для двух сишналов по всем окнам

    return signals


@pytest.mark.parametrize("h", hurst_values)
@pytest.mark.parametrize("lag", test_lags)
def test_tdc_dpcca_lags(create_signal_pair, h, lag):
    arr = create_signal_pair[h]
    new_z = arr.shape[1]
    s = [64, 128, 256]
    step = 0.6
    pd = 1
    n_integral = 1
    true_lag = 10
    lag_range = np.arange(true_lag - 5, true_lag + 6)
    P, R, F = tdc_dpcca_worker(
        s=s,
        arr=arr,
        step=step,
        pd=pd,
        time_delays=lag_range,
        flag_use_lags=True,
        n_integral=n_integral,
    )
    for s_idx in range(len(s)):
        corr_series = R[:, s_idx, 0, 1]
        max_lag_idx = np.argmax(corr_series)
        estimated_lag = lag_range[max_lag_idx]
        assert (
            estimated_lag == true_lag
        ), f"херст={h}, временной масштаб={s[s_idx]}: настоящий лаг={true_lag}, измеренный лаг {estimated_lag}"
