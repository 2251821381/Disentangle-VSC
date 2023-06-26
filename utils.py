import copy
import torch
import numpy as np
from scipy import signal
from librosa.filters import mel
from scipy.signal import get_window
from math import ceil


def quantize_f0_numpy(x, num_bins=256):
    # x is logf0
    assert x.ndim == 1
    # 保证所有的元素都在【0,1】范围内，小于0的则取0
    x = x.astype(float).copy()
    uv = (x <= 0)
    x[uv] = 0.0
    assert (x >= 0).all() and (x <= 1).all()
    x = np.round(x * (num_bins - 1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins + 1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc, x.astype(np.int64)


def quantize_f0_torch(x, num_bins=256):
    # x is logf0
    B = x.size(0)
    x = x.view(-1).clone()
    uv = (x <= 0)
    x[uv] = 0
    assert (x >= 0).all() and (x <= 1).all()
    x = torch.round(x * (num_bins - 1))
    x = x + 1
    x[uv] = 0
    enc = torch.zeros((x.size(0), num_bins + 1), device=x.device)
    enc[torch.arange(x.size(0)), x.long()] = 1
    return enc.view(B, -1, num_bins + 1), x.view(B, -1).long()


def get_mask_from_lengths(lengths, max_len):
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids >= lengths.unsqueeze(1)).bool()
    return mask


def pad_seq_to_2(x, len_out=128):
    """填充x，使得其长度为len_out"""
    # 计算需要补齐的长度
    len_pad = (len_out - x.shape[1])
    assert len_pad >= 0
    # 返回填充后的数据，以及填充用的长度
    return np.pad(x, ((0, 0), (0, len_pad), (0, 0)), 'constant'), len_pad


def pad_sequences(sequences, max_len_pad):
    # 通道维度
    channel_dim = sequences[0].size()[-1]
    # 输出数据维度
    out_dims = (len(sequences), max_len_pad, channel_dim)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, :] = tensor[:max_len_pad]

    return out_tensor


def pad_seq(x, base=32, constant_values=0):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant', constant_values=constant_values), len_pad


def make_onehot(label, n_classes):
    speaker_vector = np.zeros(n_classes)
    speaker_vector[label] = 1
    return speaker_vector.astype(dtype=np.float32)