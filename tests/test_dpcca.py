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


@pytest.mark.timeout(300)  # 5 minutes timeout
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


@pytest.mark.timeout(300)  # 5 minutes timeout
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


@pytest.mark.timeout(300)  # 5 minutes timeout
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


@pytest.mark.timeout(300)  # 5 minutes timeout
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


@pytest.mark.timeout(300)  # 5 minutes timeout
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


@pytest.mark.timeout(300)  # 5 minutes timeout
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


hurst_values = [1.0, 1.25, 1.5]
test_lags = [0, 1, 2, 3]


@pytest.fixture(scope="module")
def create_signal_pair():
    length = 2**10
    signals = {}
    true_lag = 6
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

        x = Z_full[:-true_lag]
        y = Z_full[true_lag:]
        signals[h] = np.vstack([y, x])
    return signals


@pytest.mark.parametrize("h", hurst_values)
# @pytest.mark.parametrize("lag", test_lags)
def test_tdc_dpcca_lags(create_signal_pair, h):
    arr = create_signal_pair[h]
    s = [256, 512, 1024]
    step = 1
    pd = 1
    n_integral = 0
    true_lag = -6
    p, r, f = tdc_dpcca_worker(
        s=s,
        arr=arr,
        step=step,
        pd=pd,
        time_delays=None,
        max_time_delay=abs(true_lag),
        n_integral=n_integral,
    )
    lags_arr = np.arange(true_lag, -true_lag + 1)
    for s_idx in range(len(s)):
        correlation = r[:, s_idx, 0, 1]
        max_lag_idx = np.argmax(correlation)
        estimated_lag = lags_arr[max_lag_idx]
        assert abs(estimated_lag - true_lag) <= 1


@pytest.mark.parametrize("h", hurst_values)
def test_dpcca_with_time_lag(create_signal_pair, h):
    arr = create_signal_pair[h]
    s = [256, 512, 1024]
    step = 1
    pd = 1
    n_integral = 0
    true_lag = -6
    lags_arr = np.arange(true_lag, -true_lag + 1)
    p, r, f, s_current = dpcca(
        arr,
        pd,
        step,
        s,
        abs(true_lag),
        buffer=False,
        gc_params=None,
        n_integral=n_integral,
        processes=1,
    )
    for s_idx in range(len(s_current)):
        correlation = r[:, s_idx, 0, 1]
        max_lag_idx = np.argmax(correlation)
        estimated_lag = lags_arr[max_lag_idx]
        assert abs(estimated_lag - true_lag) <= 1
