import librosa
import numpy as np
import pyworld
import pysptk


def world_encode_data(wav, fs, frame_period=5.0, num_mcep=36):
    # f0s = list()
    # timeaxes = list()
    # sps = list()
    # aps = list()
    # mcs = list()

    # for wav in wavs:
    f0, timeaxis, sp, ap, mc = world_decompose(wav=wav, fs=fs, frame_period=frame_period, num_mcep=num_mcep)
    # f0s.append(f0)
    # timeaxes.append(timeaxis)
    # sps.append(sp)
    # aps.append(ap)
    # mcs.append(mc)

    return f0, timeaxis, sp, ap, mc


def world_decompose(wav, fs, frame_period=5.0, num_mcep=36):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(sp, order=num_mcep - 1, alpha=alpha)

    return f0, timeaxis, sp, ap, mc
