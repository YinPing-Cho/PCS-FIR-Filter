# PCS-FIR-Filter

Based on the spectral perceptual gains from the [official PCS repo](https://github.com/RoyChao19477/PCS/blob/main/PCS/PCS.py), this project aims to derive the equivalent finite-impulse-response (FIR) filter coefficients to allow Perceptual Contrast Stretching (PCS) be performed directly on waveforms.

The FIR filtering in this project is performed with PyTorch convolution as a demonstration for Deep Learning applications. Of course, the derived filter coefficients (in numpy format) can be applied to other backends.

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
- `python PCS_coeffs_generate.py` generates FIR filter coefficients (in `*.npy` format) and impulse response plot under directory `generated_freq_response/`.
- `python test_PCS_wave.py` performs PCS with the FIR filter coefficients derived by `PCS_coeffs_generate.py` and outpus filtered audio under directory `audio_PCSed/`.

## Reference
- The official repo of PCS (https://github.com/RoyChao19477/PCS).
- The original PCS paper: Rong Chao, Cheng Yu, Szu-Wei Fu, Xugang Lu, Yu Tsao, "Perceptual Contrast Stretching on Target Feature for Speech Enhancement," (http://arxiv.org/abs/2203.17152)
