import numpy as np
import pytest
from scipy import signal, stats

from StatTools.analysis.dpcca import dpcca

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
