import torch
import torchaudio
import soundfile as sf
import os
import numpy as np
import matplotlib.pyplot as plt

def norm_audio(audio):
    return audio / np.max(audio) * 0.9

def generate_noise(length):
    ones = torch.ones(length)
    noise = torch.normal(mean=ones*0, std=ones*0.1)
    return noise

def load_wav(full_path):
    data, sampling_rate = sf.read(full_path)
    #data = norm_audio(data)
    data = torch.FloatTensor(data).unsqueeze(0)
    return data, sampling_rate

def dump_audio_samples(audio_batch, output_dir, suffix=None, sampling_rate=22050):
    assert len(list(audio_batch.size())) == 2, audio_batch.size()
    for i, a in enumerate(audio_batch):
        if suffix is not None:
            filename = os.path.join(output_dir,str(i) + "_PCSed_{}.wav".format(suffix))
        else:
            filename = os.path.join(output_dir,str(i) + "_PCSed.wav")
        torchaudio.save(filename, a.unsqueeze(0), sample_rate=sampling_rate, encoding="PCM_F",bits_per_sample=32)

def onesided_spectrum(x, dB=True):
    x = torch.fft.fft(x).square().abs().float()
    x = x[:x.size(-1)//2]

    if dB:
        return 20 * torch.log10(x)
    else:
        return x

def plot_before_after_spectra(audio_before, audio_after, min_max=None):
    audio_before = audio_before.squeeze()
    audio_after = audio_after.squeeze()

    spectrum_before = onesided_spectrum(audio_before)
    spectrum_after = onesided_spectrum(audio_after)

    length = min(spectrum_before.size(-1), spectrum_after.size(-1))
    spectrum_before = spectrum_before[...,:length]
    spectrum_after = spectrum_after[...,:length]

    t = np.linspace(0, length, length)
    t = t / length
    plt.plot(t, spectrum_before, 'b', label='before')
    plt.plot(t, spectrum_after, 'r', label='after')
    plt.xlabel('Nyquist Frequency')
    plt.ylabel('Decibel (dB)')

    if min_max is not None:
        ax = plt.gca()
        ax.set_ylim(min_max)

    plt.title('Before and After spectra')
    plt.legend(loc='best')
    plt.show()
    plt.close()

def plot_response_curves(curves, min_max=None):
    assert isinstance(curves, list)

    length = len(curves[0])
    t = np.linspace(0, length, length)
    t = t / length

    for idx, c in enumerate(curves):
        plt.plot(t, c, label='curve {}'.format(idx))
    plt.xlabel('Nyquist Frequency')
    plt.ylabel('Gain (linear)')

    if min_max is not None:
        ax = plt.gca()
        ax.set_ylim(min_max)
        
    plt.title('Magnitude Response Curves')
    plt.legend(loc='best')
    plt.show()
    plt.close()