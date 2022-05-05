# PCS-FIR-Filter

Based on the spectral perceptual gains from the [official PCS repo](https://github.com/RoyChao19477/PCS/blob/main/PCS/PCS.py), this project aims to derive the equivalent linear-time-invariant (LTI) finite-impulse-response (FIR) filter coefficients to allow Perceptual Contrast Stretching (PCS) be performed directly on waveforms.

FIR filtering is a differentiable operation, which makes it ideal for Deep Learning applications working directly on waveforms. The FIR filtering example in this project is performed with PyTorch 1-D convolution layer. Of course, the derived filter coefficients (in numpy format) can also be easily applied to other backends.

## Requirements
```
torch >= 1.8
torchaudio
matplotlib
Soundfile
numpy
scipy
```
Available in `requirements.txt`

## Usage
1. Filter design:
- `python PCS_coeffs_generate.py --mode='manual'` generates FIR filter coefficients (in `*.npy` format) and impulse response plot under directory `generated_freq_response/` with default spectral PCS coefficients.
- Since the original PCS (spectral PCS) works on log-1-p spectrograms, the nonlinearity cannot be reproduced directly with LTI FIR filters; therefore, `python PCS_coeffs_generate.py` provides two additional statistical filter design methods to approximate the behavior of spectral PCS:
  - `python PCS_coeffs_generate.py --mode='statistical' --stat_mode='gaussian'` measures and approximate spectral PCS's equivalent LTI impulse response with Gaussian signals of varying standard deviations.
  - `python PCS_coeffs_generate.py --mode='statistical' --stat_mode='wav' --wav_dir='*'` measures and approximate spectral PCS's equivalent LTI impulse response with the .wav files you placed in `wav_dir`.
2. Filtering:
- `python test_PCS_wave.py` performs PCS with the FIR filter coefficients derived by `PCS_coeffs_generate.py` and outputs filtered audio.

## Example Results
- Frequency response of the FIR filter coefficients derived from the default PCS settings with `GAIN_SMOOTHING = 0.2`:
<p align="center">
<img src="https://github.com/YinPing-Cho/PCS-FIR-Filter/blob/main/generated_freq_response/PCS_coeffs_freqz.png" height="384">
</p>
- Spectra comparison of befer and after PCS:
<p align="center">
<img src="https://github.com/YinPing-Cho/PCS-FIR-Filter/blob/main/audio_PCSed/before_after.png" height="384">
</p>

- Frequency response of the FIR filter coefficients derived with audio-wav-based statistical method with [Mpop600](https://ieeexplore.ieee.org/document/9306461) Mandarin singing voice dataset:
<p align="center">
<img src="https://github.com/YinPing-Cho/PCS-FIR-Filter/blob/dev/statistical/mpop600_fr.png" height="384">
</p>
- Spectra comparison of befer and after PCS:
<img src="https://github.com/YinPing-Cho/PCS-FIR-Filter/blob/dev/statistical/sing_pcs_mp600.png" height="384">
</p>

## Reference
- The official repo of PCS (https://github.com/RoyChao19477/PCS).
- The original PCS paper: Rong Chao, Cheng Yu, Szu-Wei Fu, Xugang Lu, Yu Tsao, "Perceptual Contrast Stretching on Target Feature for Speech Enhancement," (http://arxiv.org/abs/2203.17152)
- Mpop600 Mandarin singing voice dataset: C. -C. Chu, F. -R. Yang, Y. -J. Lee, Y. -W. Liu and S. -H. Wu, "MPop600: A Mandarin Popular Song Database with Aligned Audio, Lyrics, and Musical Scores for Singing Voice Synthesis," 2020 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), 2020, pp. 1647-1652.
