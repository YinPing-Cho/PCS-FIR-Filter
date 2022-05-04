import numpy as np
from utils import *
from PCS_filter import PCS_FIR_filter

OUTDIR = 'statistical'
PCS_FIR_Filter = PCS_FIR_filter(coeffs_path=os.path.join(OUTDIR,'PCS_coeffs_mpop600.npy'))

def test(audio_path=None):
    if audio_path is not None:
        audio, sr = load_wav(audio_path)
    else:
        sr=22050
        audio = generate_noise(22050*10)
    filtered_audio = PCS_FIR_Filter(audio)
    plot_before_after_spectra(audio, filtered_audio, min_max=[-120, 160])
    dump_audio_samples(filtered_audio, sampling_rate=sr, output_dir='statistical')

#test(audio_path=None)
test(audio_path='audio_original/f1_001_7.wav')