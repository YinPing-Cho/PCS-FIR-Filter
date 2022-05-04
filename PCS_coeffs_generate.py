import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, firls
import os
import sys
import argparse
from utils import *
from original_PCS_spectral import process_wav
from tqdm import tqdm

TAPS = 127
BINS = 257
GAIN_SMOOTHING = 0.2
OUTDIR = 'generated_freq_response'
STAT_DIR = 'statistical'

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

def smooth_gains(gains):
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

    return desired

def get_multiband_filter(PCS_params, numtaps):
    bands = []
    gains = []
    for _, params in PCS_params.items():
        bands.extend(params['band'])
        gains.append(np.exp(params['gain']))

    min_gain = min(gains)
    for idx in range(len(gains)):
        gains[idx] = gains[idx] / min_gain

    desired = smooth_gains(gains)

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

def load_and_filter(audio_path=None):
    if audio_path is not None:
        audio, sr = load_wav(audio_path)
    else:
        sr=22050
        audio = generate_noise(sr*10)
    filtered_audio = process_wav(audio)
    audio = torch.FloatTensor(audio)
    filtered_audio = torch.FloatTensor(filtered_audio)
    return audio, filtered_audio

def adaptive_smoothing(curve, target_length):
    smoother = torch.nn.AdaptiveAvgPool1d(target_length)
    smoothed = smoother(curve.view(1,1,curve.size(-1)))
    return smoothed.squeeze()

def moving_avg_spectra(spectrum_avg, spectrum, count):
    spectrum = adaptive_smoothing(spectrum, BINS)
    if spectrum_avg is None:
        assert count == 0
        return spectrum
    return (spectrum_avg * count + spectrum) / (count+1)

def record_spectrum_avg(spectrum_avg, audio, count):
    spectrum = onesided_spectrum(audio, dB=False)
    spectrum_avg = moving_avg_spectra(spectrum_avg, spectrum, count)
    return spectrum_avg

def statistical_response(mode='noise', num_samples=100, wav_dir=None):
    original_spectrum_avg = None
    filtered_spectrum_avg = None

    if mode == 'noise':
        audio_path = None
    elif mode == 'wav':
        assert wav_dir is not None
        filepaths = list()
        for file in os.listdir(wav_dir):
            if file.endswith('.wav'):
                wav_path = os.path.join(wav_dir, file)
                audio, _ = load_wav(wav_path)
                if audio.shape[-1] > BINS:
                    filepaths.append(os.path.join(wav_dir, file))
        if num_samples is not None:
            num_samples = min(num_samples, len(filepaths))
        else:
            num_samples = len(filepaths)

    with tqdm(total=num_samples) as pbar:
        for idx in range(num_samples):
            if mode == 'noise':
                audio_path = None
            elif mode == 'wav':
                audio_path = filepaths[idx]

            audio, filtered_audio = load_and_filter(audio_path)

            original_spectrum_avg = record_spectrum_avg(original_spectrum_avg, audio, idx)
            filtered_spectrum_avg = record_spectrum_avg(filtered_spectrum_avg, filtered_audio, idx)

            pbar.update(1)

    pointwise_gains = filtered_spectrum_avg / original_spectrum_avg

    plot_response_curves([original_spectrum_avg, filtered_spectrum_avg])
    plot_response_curves([pointwise_gains])
    return pointwise_gains

def manual_PCS():
    _, PCS_params = init_PCS_params()

    cascade_coeffs = get_multiband_filter(PCS_params, TAPS)
    np.save(os.path.join(OUTDIR,'PCS_coeffs.npy'), cascade_coeffs)

    w, h = freqz(cascade_coeffs)
    plot_fir_response(w, h)

def statistical_PCS(args):
    pointwise_gains = statistical_response(mode=args.stat_mode, num_samples=args.num_samples, wav_dir=args.wav_dir)
    np.save(os.path.join(STAT_DIR,'stat_gains.npy'), pointwise_gains)

    bands = list()
    gains = list()
    for idx in range(len(pointwise_gains)):
        bands.extend([idx/BINS, (idx+1)/BINS])
        gains.append(pointwise_gains[idx])

    desired = smooth_gains(gains)

    cascade_coeffs = firls(TAPS, bands, desired, weight=None, nyq=None, fs=2)
    np.save(os.path.join(STAT_DIR,'PCS_coeffs.npy'), cascade_coeffs)

    w, h = freqz(cascade_coeffs)
    plot_fir_response(w, h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='statistical',
                        help='`statistical` or `manual`.\n\
                            `statistical` uses specified signals to measure the original spectral PCS as a LTI system to obtain gains.\n\
                            `manual` uses default gains in the original spectroal PCS.')
    parser.add_argument('-stm', '--stat_mode', type=str, default='noise', \
                        help='noise` or `wav`.\n `spcifies the measuring signal if mode==`statistical`.\n\
                            if `noise`, generates Gaussian noise as measuring signals.\n\
                            if `wav` load .wav files from specified directory as measuring signals',
                        required=False)
    parser.add_argument('-wd', '--wav_dir', type=str, default=None,
                        required=False, help='specifies where the .wav files are located if mode==`statistical` and --stat_mode==wav')
    parser.add_argument('-n', '--num_samples', type=int, default=100,
                        required=False, help='if mode==`statistical`, the measuring process will be performed num_samples times.\n\
                        if --stat_mode==`wav`, the process will be performed min(num_samples, num_wavs_loaded) times')

    args = parser.parse_args()
    if args.mode == 'manual':
        manual_PCS()
    elif args.mode == 'statistical':
        assert args.stat_mode == 'noise' or args.stat_mode == 'wav', 'args.stat_mode: {}'.format(args.stat_mode)
        if args.stat_mode == 'wav':
            assert os.path.isdir(args.wav_dir), 'Error, is not dir; args.wav_dir: {}'.format(args.wav_dir)

        statistical_PCS(args)
    else:
        parser.print_help()