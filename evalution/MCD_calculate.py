import csv
import json
import os

from glob import glob
import librosa
from evalution.world_processing import *
import scipy.spatial

from fastdtw import fastdtw

def load_eval_spec(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        samples = list(reader)
    return samples


def get_feature(wav, fs=16000):
    f0, timeaxis, sp, ap, mc = world_encode_data(wav, fs)
    return f0, mc


def evaluate_mcd_wav(file_path):
    """
    file_path2: conversion file dir
    file_path1: source file dir
    """

    MCD_array = []
    samples = load_eval_spec(file_path)

    for sample in samples:
        src_data, _ = librosa.load(sample["syn"], sr=16000)
        trg_data, _ = librosa.load(sample["ref"], sr=16000)

        src_f0, src_mcc = get_feature(src_data)
        trg_f0, trg_mcc = get_feature(trg_data)

        # non-silence parts
        trg_idx = np.where(trg_f0 > 0)[0]
        trg_mcc = trg_mcc[trg_idx, :24]

        src_idx = np.where(src_f0 > 0)[0]
        src_mcc = src_mcc[src_idx, :24]

        # DTW
        _, path = fastdtw(src_mcc, trg_mcc, dist=scipy.spatial.distance.euclidean)
        twf = np.array(path).T
        cvt_mcc_dtw = src_mcc[twf[0]]
        trg_mcc_dtw = trg_mcc[twf[1]]

        # MCD
        diff2sum = np.sum((cvt_mcc_dtw - trg_mcc_dtw) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        # logging.info('{} {}'.format(basename, mcd))
        print('utterance {} mcd: {}'.format(sample["syn"].split("/")[-1], mcd))
        MCD_array.append(mcd)

    return MCD_array


if __name__ == '__main__':

    # file_path = "eval_data_pairs_p225_p226.csv"
    file_path = "eval_data_pairs.csv"
    MCD_arr = evaluate_mcd_wav(file_path)

    mcd_value = np.mean(np.array(MCD_arr))

    print('MCD value between two speaker: ', mcd_value)
