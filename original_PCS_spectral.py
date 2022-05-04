import numpy as np
import librosa
import scipy
from utils import *

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

def Sp_and_phase(signal):
    signal_length = signal.shape[0]
    n_fft = 512
    y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)

    F = librosa.stft(y_pad, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)

    Lp = PCS_curve * np.transpose(np.log1p(np.abs(F)), (1, 0))
    phase = np.angle(F)

    NLp = np.transpose(Lp, (1, 0))

    return NLp, phase, signal_length


def SP_to_wav(mag, phase, signal_length):
    mag = np.expm1(mag)
    Rec = np.multiply(mag, np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming, length=signal_length)
    return result

def process_wav(signal):
    noisy_LP, Nphase, signal_length = Sp_and_phase(signal.squeeze().numpy())

    enhanced_wav = SP_to_wav(noisy_LP, Nphase, signal_length)

    return enhanced_wav

def test(audio_path=None):
    if audio_path is not None:
        audio, sr = load_wav(audio_path)
    else:
        sr=22050
        audio = generate_noise(22050*10)
    filtered_audio = process_wav(audio)
    filtered_audio = torch.tensor(filtered_audio).unsqueeze(0)
    plot_before_after_spectra(audio, filtered_audio, min_max=[-60, 100])
    dump_audio_samples(filtered_audio, sampling_rate=sr, output_dir='roychao_audio_PCSed')

#test(audio_path=None)
#test(audio_path='audio_original/f1_001_7.wav')