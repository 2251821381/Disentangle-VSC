import os
import pickle

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from prepared.prepared_utils import butter_highpass, speaker_normalization, pySTFT

fft_length = 1024
hop_length = 256
# 创建一个 Filterbank 矩阵以将FFT bins合并到mel-frequency bins
# sr:输入信号的采样频率， n_fft:FFT分量的数量， n_mels:生成mel bands的数量，
# fmin:最低频率（in Hz），fmax:最高频率，
# return: mel transform matrix, np.ndarray[shape=(n_mels, 1 + n_fft/2)]
# 创建一个 Mel 过滤器组
# 这将产生一个线性变换矩阵来将 FFT 箱投射到梅尔频率箱上
# (n_mels, 1 + n_fft//2),(80, 513)
mel_basis = mel(sr=16000, n_fft=fft_length, n_mels=80, fmin=90, fmax=7600).T
# mel_basis = mel(hparams.sample_rate, hparams.n_fft, fmin=hparams.fmin, fmax=hparams.fmax, n_mels=hparams.n_mels).T
# 设计一个最小等级的标量
min_level = np.exp(-100 / 20 * np.log(10))
# 调用 butter_highpass 创建一个高通滤波器，得到滤波器系数：分子(' b ')和分母(' a ')多项式
b, a = butter_highpass(30, 16000, order=5)

spk2gen = pickle.load(open('../dataset/VCTK/spk2gen_nikl.pkl', "rb"))

# Modify as needed
# audio file directory
rootDir = '../dataset/VCTK/dataset/wav16'
# f0 directory
targetDir_f0 = '../dataset/VCTK/dataset/raptf0'
# spectrogram directory
targetDir = '../dataset/VCTK/dataset/spmel'


dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):
    print(subdir)

    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    if not os.path.exists(os.path.join(targetDir_f0, subdir)):
        os.makedirs(os.path.join(targetDir_f0, subdir))
    _, _, fileList = next(os.walk(os.path.join(dirName, subdir)))

    gender, index = spk2gen[subdir]
    print(f'gender: %s, index: %s' % (gender, index))
    if gender == 'M':
        lo, hi = 50, 250
    elif gender == 'F':
        lo, hi = 100, 600
    else:
        lo, hi = 100, 600
        # raise ValueError

    # 同一个音频文件子目录下，设定生成的随机数序列相同
    prng = RandomState(int(subdir[1:]))
    for fileName in sorted(fileList):
        # read audio file
        # 以 NumPy 数组的形式从声音文件中提供音频数据
        x, fs = sf.read(os.path.join(dirName, subdir, fileName))
        x = x.T
        # 重采样，指定原采样频率和目标采样频率
        x = librosa.resample(x, fs, 16000)
        assert fs == 16000
        if x.shape[0] % hop_length == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        # Ddd a little random noise for models robustness
        wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

        # compute spectrogram, (1 + n_fft//2, T), (513, ),幅度谱
        D = pySTFT(wav).T
        # Convert to mel and normalize,(n_mels, T)，梅尔谱
        D_mel = np.dot(D, mel_basis)
        # to decibel,转换成分贝
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100

        # extract f0
        f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768, fs, hop_length, min=lo, max=hi, otype=2)
        index_nonzero = (f0_rapt != -1e10)
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
        f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

        assert len(S) == len(f0_rapt)
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)
        np.save(os.path.join(targetDir_f0, subdir, fileName[:-4]),
                f0_norm.astype(np.float32), allow_pickle=False)