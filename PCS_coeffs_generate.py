import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, firls
import os

TAPS = 127
GAIN_SMOOTHING = 0.2
OUTDIR = 'generated_freq_response'

def init_PCS_params():
    PCS_curve = np.ones(257)      # Perceptual Contrast Stretching
    PCS_curve[0:3] = 1
    PCS_curve[3:6] = 1.070175439
    PCS_curve[6:9] = 1.182456140
    PCS_curve[9:12] = 1.287719298
    PCS_curve[12:138] = 1.4       # Pre Set
    PCS_curve[138:166] = 1.322807018
    PCS_curve[166:200] = 1.238596491
    PCS_curve[200:241] = 1.161403509
    PCS_curve[241:256] = 1.077192982

    PCS_params = {
        'Band0': dict(band=[0, 3/256], gain=1.0),
        'Band1': dict(band=[3/256, 6/256], gain=1.070175439),
        'Band2': dict(band=[6/256, 9/256], gain=1.182456140),
        'Band3': dict(band=[9/256, 12/256], gain=1.287719298),
        'Band4': dict(band=[12/256, 138/256], gain=1.4),
        'Band5': dict(band=[138/256, 166/256], gain=1.322807018),
        'Band6': dict(band=[166/256, 200/256], gain=1.238596491),
        'Band7': dict(band=[200/256, 241/256], gain=1.161403509),
        'Band8': dict(band=[241/256, 1], gain=1.077192982),
    }

    return PCS_curve, PCS_params

def get_multiband_filter(PCS_params, numtaps):
    bands = []
    gains = []
    for _, params in PCS_params.items():
        bands.extend(params['band'])
        gains.append(np.exp(params['gain']))

    min_gain = min(gains)
    for idx in range(len(gains)):
        gains[idx] = gains[idx] / min_gain

    desired = []
    for idx in range(len(gains)):
        if idx == 0:
            gain_2 = (gains[idx+1]-gains[idx]) * GAIN_SMOOTHING + gains[idx]
            desired.append(gains[idx])
            desired.append(gain_2)
        elif idx == len(gains)-1:
            gain_1 = (gains[idx-1]-gains[idx]) * GAIN_SMOOTHING + gains[idx]
            desired.append(gain_1)
            desired.append(gains[idx])
        else:
            gain_1 = (gains[idx-1]-gains[idx]) * GAIN_SMOOTHING + gains[idx]
            gain_2 = (gains[idx+1]-gains[idx]) * GAIN_SMOOTHING + gains[idx]
            desired.append(gain_1)
            desired.append(gain_2)

    multiband_coeffs = firls(numtaps, bands, desired, weight=None, nyq=None, fs=2)

    return multiband_coeffs

def plot_fir_response(w, h):
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    ax1 = fig.add_subplot(111)
    plt.plot(w, abs(h), 'b')
    plt.ylabel('Amplitude [linear]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.savefig(os.path.join(OUTDIR,'PCS_coeffs_freqz.png'))
    plt.show()
    ax1.clear()
    ax2.clear()

PCS_curve, PCS_params = init_PCS_params()

cascade_coeffs = get_multiband_filter(PCS_params, TAPS)
np.save(os.path.join(OUTDIR,'PCS_coeffs.npy'), cascade_coeffs)

w, h = freqz(cascade_coeffs)
plot_fir_response(w, h)