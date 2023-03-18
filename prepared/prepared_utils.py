"""Utilities for prepared manipulation."""

import librosa
import numpy as np
from scipy.signal import lfilter, get_window
from scipy import signal


def log_mel_spectrogram(
    x: np.ndarray,
    preemph: float,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    f_min: int,
) -> np.ndarray:
    """Create a log Mel spectrogram from a raw audio signal."""
    x = lfilter([1, -preemph], [1], x)
    magnitude = np.abs(
        librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_mels, fmin=f_min)
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel_spec = np.log(mel_spec + 1e-9)
    return log_mel_spec.T


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # 巴特沃斯数字和模拟滤波器设计。设计一个 N 阶数字或模拟巴特沃斯滤波器并返回滤波器系数。
    # IIR 滤波器的分子 (`b`) 和分母 (`a`) 多项式
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def pySTFT(x, fft_length=1024, hop_length=256):
    """ 实现给定样本片段的短时傅里叶变换
        采样率16kHz：1s信号包含16000个采样点
        x         :音频样本
         采样率 * 语音时长 = 语音采样点数
         16000*0.064=1024, 16000*0.016=256
        fft_length:傅里叶变换的长度，窗长,50ms*16000=800
        hop_length:跳跃的长度，帧移,12.5ms*16000=200
        return 傅里叶变换的结果
    """
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)


def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    # index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0
