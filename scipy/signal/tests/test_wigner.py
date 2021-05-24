import numpy as np
from scipy.signal import gausspulse

from scipy.signal import wigner


class TestWigner:
    def test_wvd_shapes(self):
        t, f, wv = wigner.wvd(np.sin(np.linspace(0, 100, 1024)),
                              resolution=4, win_size=128)
        assert len(t) == 256
        assert len(f) == 128
        assert wv.shape == (128, 256)

    def test_wvd_tc_fc(self):
        fs = 100
        fc = 4.2
        tc = 512 / fs
        t = np.arange(0, 1024) / fs
        x_re, x_im = gausspulse(t - tc, fc, bw=0.2, retquad=True)
        t, f, wv = wigner.wvd(x_re + 1j * x_im, fs)
        max_idx = np.unravel_index(np.argmax(wv), wv.shape)
        assert np.allclose(fc, f[max_idx[0]], atol=f[1] - f[0])
        assert np.allclose(tc, t[max_idx[1]], atol=t[1] - t[0])


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# parameter
f_s = 100
t = np.arange(0, 1024) / 100
x = np.sin(2*np.pi*1*t) + np.sin(2*np.pi*5*t)

# Wigner-Ville-Distribution
t, f, wv = signal.wvd(signal.hilbert(x), f_s)

plt.figure()
plt.pcolormesh(t, f, wv, shading='nearest')
plt.xlabel('Time $t$ / s')
plt.ylabel('Frequency $f$ / Hz')
plt.ylim([0, 6])
plt.show()
