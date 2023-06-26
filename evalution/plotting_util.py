import numpy
import numpy as np
import wave
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
from librosa.filters import mel
from scipy import signal
from numpy.random import RandomState
from pysptk import sptk
from prepared.prepared_utils import butter_highpass, speaker_normalization, pySTFT
from evalution.yin import pitch_calc, compute_yin

wlen = 512
inc = 128
# plt.figure(dpi=600)  # 将显示的所有图分辨率调高
matplotlib.rc("font", family='SimHei')  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示符号

mel_basis = mel(sr=16000, n_fft=1024, n_mels=80, fmin=90, fmax=7600).T
# mel_basis = mel(hparams.sample_rate, hparams.n_fft, fmin=hparams.fmin, fmax=hparams.fmax, n_mels=hparams.n_mels).T
# 设计一个最小等级的标量
min_level = np.exp(-100 / 20 * np.log(10))
# 调用 butter_highpass 创建一个高通滤波器，得到滤波器系数：分子(' b ')和分母(' a ')多项式
b, a = butter_highpass(30, 16000, order=5)

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        }


def short_time_energy(file_path, file_name, target_path):
    """
    draw short-time energy
    @param file_path:
    @param file_name:
    @return:
    """
    file = wave.open(file_path, "rb")
    params = file.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = file.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    signal_length = len(wave_data)  # 信号总长度
    if signal_length <= wlen:  # 若信号长度小于一个帧的长度，则帧数定义为1
        frame_num = 1
    else:  # 否则，计算帧的总长度
        frame_num = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))
    pad_length = int((frame_num - 1) * inc + wlen)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((wave_data, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, wlen), (frame_num, 1)) + np.tile(np.arange(0, frame_num * inc, inc),
                                                                    (wlen, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    windown = np.hanning(wlen)
    d = np.zeros(frame_num)
    time = np.arange(0, frame_num) * (inc * 1.0 / framerate)
    for i in range(0, frame_num):
        a = frames[i:i + 1]
        b = a[0] * windown
        c = np.square(b)
        d[i] = np.sum(c)
    d = d * 1.0 / (max(abs(d)))

    plt.figure()
    plt.plot(time, d, c="g")
    plt.xlabel("time", font)
    plt.ylabel("energy", font)
    # plt.grid()  # 生成网格
    plt.savefig(target_path + file_name + '_energy.png', dpi=600)
    plt.show()


def frequency(file_path, file_name):
    f = wave.open(file_path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    fft_signal = np.fft.fft(wave_data)  # 语音信号FFT变换
    fft_signal = abs(fft_signal)  # 取变换结果的模
    plt.figure(figsize=(10, 4))
    time = np.arange(0, nframes) * framerate / nframes
    plt.plot(time, fft_signal, c="g")
    plt.grid()
    plt.savefig('D:/workspace/PycharmProject/result/parallel/test6/VCTK6/frequency/' + file_name + '.png', dpi=600)
    plt.show()


def spectrum(file_path, file_name, target_path):
    prng = RandomState()
    # read audio file
    # 以 NumPy 数组的形式从声音文件中提供音频数据
    samples, sr = librosa.load(file_path, sr=16000)
    if samples.shape[0] % 256 == 0:
        samples = np.concatenate((samples, np.array([1e-06])), axis=0)
    # Remove drifting noise
    y = signal.filtfilt(b, a, samples)
    # Ddd a little random noise for model robustness
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

    # compute spectrogram, (1 + n_fft//2, T), (513, ),幅度谱
    D = pySTFT(wav).T
    # Convert to mel and normalize,(n_mels, T)，梅尔谱
    D_mel = np.dot(D, mel_basis)
    # to decibel,转换成分贝
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100
    # S = S[20:148, :]

    # x = list(range(0, 131, 20))
    # y = list(range(0, 80, 10))

    fig, axs = plt.subplots(1, 1)
    # axs.set_title('Spectrogram (db)', font)
    # axs.set_ylabel('frequncy', font)
    # axs.set_xlabel('times', font)
    im = axs.imshow(S.T, origin='lower', aspect='auto')
    # plt.tick_params(labelsize=20)  # 刻度字体大小20
    # fig.colorbar(im, ax=axs)
    # fig.colorbar(im)
    # plt.xticks(x)
    # plt.yticks(y)
    plt.tick_params(labelsize=16)  # 刻度字体大小13
    # plt.savefig('D:/workspace/PycharmProject/result/parallel/test6/VCTK6/spectrum/' + file_name + '.png')
    plt.savefig(target_path + file_name + '_mel_1.png')
    plt.show(block=False)
    # # plt.imshow(S.T, origin='lower', aspect='auto')
    # # plt.title("spectrogram")
    # # plt.xlabel("times")
    # # plt.ylabel("frequency")
    # # plt.show()

    return S


def displayWaveform(file_path, file_name, target_path):  # 显示语音时域波形
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    samples, sr = librosa.load(file_path, sr=16000)

    time = np.arange(0, len(samples)) * (1.0 / sr)

    plt.plot(time, samples)
    plt.title("语音信号时域波形")
    plt.xlabel("时长（秒）")
    plt.ylabel("振幅")
    plt.savefig(target_path + file_name + '_wave.png')
    plt.show()


def pitch_contours(file_path, file_name, target_path):
    # pitch contours
    samples, sr = librosa.load(file_path, sr=16000)
    # samples = librosa.effects.trim(samples, 15, frame_length=32, hop_length=16)[0]
    samples = librosa.util.normalize(samples)
    confidence_threshold = 0.7
    d = pitch_calc(samples, sr=16000, confidence_threshold=confidence_threshold, gaussian_smoothing_sigma=1.0)
    # d = d[20:148]

    # x = list(range(0, 131, 20))
    y = list(range(0, 301, 50))

    plt.figure()
    plt.plot(d, c="g")
    # plt.xlabel("times", font)
    # plt.xticks(x)
    plt.yticks(y)
    plt.tick_params(labelsize=16)  # 刻度字体大小13
    # plt.tick_params(labelsize=20)  # 刻度字体大小20
    # plt.ylabel("energy")
    # plt.grid()  # 生成网格
    plt.savefig(target_path + file_name + '_pitch.png')
    # plt.savefig('D:/workspace/PycharmProject/result/parallel/test6/VCTK6/pitch/' + file_name + '-1.png')
    plt.show()

    # d_yin = compute_yin(samples, sr=16000, w_len=1024, w_step=256, harmo_thresh=1 - confidence_threshold)[0]
    # plt.figure()
    # plt.plot(d_yin, c="g")
    # plt.xlabel("times")
    # plt.yticks(y)
    # plt.savefig(target_path + file_name + '_pitch_1.png')
    # plt.show()
    return d


def joinMelAndPitch():
    dir_path = r"D:/workspace/result/parallel/test40/VCTK40/001_p226_English_p251_Indian_test_d_40/"
    # name_list = ['001_p226_p251_source.wav']     # confidence_threshold = 0.72
    # name_list = ['001_p226_p251_U.wav']     # confidence_threshold = 0.72
    # name_list = ['001_p226_p251_A.wav']     # confidence_threshold = 0.72
    # name_list = ['001_p226_p251_F.wav']   # confidence_threshold = 0.755
    name_list = ['001_p226_p251_R.wav']    # 0.7
    # name_list = ['001_p226_p251_target.wav']
    for name in name_list:
        file_name = name.split('.')[0]
        file_path = dir_path + name
        samples, sr = librosa.load(file_path, sr=16000)
        samples = librosa.effects.trim(samples)[0]
        # 音高轮廓图
        samples = librosa.util.normalize(samples)

        if samples.shape[0] % 256 == 0:
            samples = np.concatenate((samples, np.array([1e-06])), axis=0)

        confidence_threshold = 0.72
        pitch = pitch_calc(samples, sr=16000, confidence_threshold=confidence_threshold, gaussian_smoothing_sigma=1.0)
        # 声谱图

        prng = RandomState()

        # Remove drifting noise
        y = signal.filtfilt(b, a, samples)
        # Ddd a little random noise for model robustness
        wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06
        # compute spectrogram, (1 + n_fft//2, T), (513, ),幅度谱
        D = pySTFT(wav).T
        # Convert to mel and normalize,(n_mels, T)，梅尔谱
        D_mel = np.dot(D, mel_basis)
        # to decibel,转换成分贝
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        mel = (D_db + 100) / 100

        plt.figure(figsize=(5, 2), dpi=150)
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.subplot(1, 2, 1)
        plt.imshow(mel.T, origin='lower', aspect='auto')
        # ax1.imshow(mel.T, origin='lower', aspect='auto')

        plt.subplot(1, 2, 2)
        # plt.imshow(pitch, c='g')
        plt.plot(pitch, color='steelblue')
        # ax2.imshow(pitch, c="g")
        plt.tight_layout()
        y = list(range(0, 251, 50))
        plt.yticks(y)
        plt.savefig('D:/workspace//result/parallel/test40/VCTK40/' + file_name + '.png')
        plt.show()


def logF0(file_path, file_name):
    hop_length = 256
    x, sr = librosa.load(file_path, sr=16000)
    x = x.T
    if x.shape[0] % hop_length == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    # Remove drifting noise
    y = signal.filtfilt(b, a, x)
    prng = RandomState(225)
    # Ddd a little random noise for models robustness
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06
    f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768, sr, hop_length, min=100, max=600, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    plt.figure()
    plt.plot(f0_norm, c="g")
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.grid()     # 生成网格
    plt.savefig('D:/workspace/PycharmProject/result/parallel/test6/VCTK6/pitch/' + file_name + '_3.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    # file_path = r"../../result/parallel/test6/VCTK6/001_p226_English_p231_English_test_6/001_p226_p231_source.wav"
    # name = '003_p231_p225_source'
    # logF0(file_path, name)
    # dir_path = r"../../result/parallel/test6/VCTK6/001_p226_English_p231_English_test_6/"

    # joinMelAndPitch()
    # # 0.7
    dir_path = r"../dataset/VCTK/dataset/wav16/p226/"
    name_list = ['p226_001_mic1.wav']
    target_path = r"../results/figures/"
    #
    for name in name_list:
        file_name = name.split('.')[0]
        file_path = dir_path + name
    #     # 短时能量图
    #     short_time_energy(file_path, file_name, target_path)
        # 音高轮廓图
        pitch_contours(file_path, file_name, target_path)
        # 声谱图
        spectrum(file_path, file_name, target_path)
    #     # 语音时域波形
    #     # displayWaveform(file_path, file_name, target_path)

