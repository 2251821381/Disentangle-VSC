import os
import torch
import pickle
import numpy as np

from functools import partial
from numpy.random import uniform
from multiprocessing import Process, Manager

from torch.utils import data
from torch.utils.data.sampler import Sampler


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, feat_dir, mode, train_file, segment_size):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.feat_dir = feat_dir
        self.mode = mode
        self.step = 20
        self.split = 0
        self.train_file = train_file
        self.segment_size = segment_size

        # 读取训练文件train.pkl
        # metaname = os.path.join(self.root_dir, "train.pkl")
        # meta = pickle.load(open(metaname, "rb"))
        meta = pickle.load(open(self.train_file, "rb"))

        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta) * [None])  # <-- can be shared between processes.
        # 多线程，每个线程处理二十个pkl文件
        processes = []
        for i in range(0, len(meta), self.step):
            # 通过创建一个 Process 对象创建新的进程
            # 进程使用成员函数 load_data ，输入为部分训练数据、数据集地址与编号偏移量
            p = Process(target=self.load_data,
                        args=(meta[i:i + self.step], dataset, i, mode))
            p.start()
            processes.append(p)
        # 遍历进程池，等待至进程终止
        for p in processes:
            p.join()

        # very important to do dataset = list(dataset)
        if mode == 'train':
            # 将得到的数据集保存到成员列表变量
            self.train_dataset = list(dataset)
            # 计算数据集长度，并保存至成员变量num_tokens
            self.num_tokens = len(self.train_dataset)
        elif mode == 'test':
            self.test_dataset = list(dataset)
            self.num_tokens = len(self.test_dataset)
        else:
            raise ValueError

        print('Finished loading {} dataset...'.format(mode))

    def load_data(self, submeta, dataset, idx_offset, mode):
        """
        submeta: 数据集子集，每个线程处理step个数据，例如第一个线程处理【0,19】个pkl文件
        dataset: 数据集对象
        idx_offset:编号偏移量
        """
        # 遍历子集数据集submeta，k为编号，sbmt为当前说话人信息（包括说话人id，说话人编码，口音编码，pkl文件路径os.path.join(speaker, fileName)）
        for k, sbmt in enumerate(submeta):
            uttrs = len(sbmt) * [None]
            # fill in speaker id and embedding
            # uttrs[0] = sbmt[0]
            # uttrs[1] = sbmt[1]
            #
            # # 遍历该用户下的音频文件
            # sp_tmp = np.load(os.path.join(self.root_dir, sbmt[2]))
            # f0_tmp = np.load(os.path.join(self.feat_dir, sbmt[2]))
            #
            # if self.mode == 'train':
            #     sp_tmp = sp_tmp[self.split:, :]
            #     f0_tmp = f0_tmp[self.split:]
            # elif self.mode == 'test':
            #     sp_tmp = sp_tmp[:self.split, :]
            #     f0_tmp = f0_tmp[:self.split]
            # else:
            #     raise ValueError
            # uttrs[2] = (sp_tmp, f0_tmp)
            for j, tmp in enumerate(sbmt):
                if j < 3:
                    # fill in speaker id and embedding
                    uttrs[j] = tmp
                else:
                    # fill in data
                    sp_tmp = np.load(os.path.join(self.root_dir, tmp))
                    f0_tmp = np.load(os.path.join(self.feat_dir, tmp))

                    if self.mode == 'train':
                        sp_tmp = sp_tmp[self.split:, :]
                        f0_tmp = f0_tmp[self.split:]
                    elif self.mode == 'test':
                        sp_tmp = sp_tmp[:self.split, :]
                        f0_tmp = f0_tmp[:self.split]
                    else:
                        raise ValueError
                    uttrs[j] = (sp_tmp, f0_tmp)
            # 将组合的说话人信息保存至偏移后的正确位置
            dataset[idx_offset + k] = uttrs

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        list_uttrs = dataset[index]
        # speaker id
        spk_id_org = list_uttrs[0]
        # speaker embedding
        emb_org = list_uttrs[1]
        # speaker accent id
        accent_org = list_uttrs[2]

        # pick random uttr with random crop
        a = np.random.randint(3, len(list_uttrs))
        # 取出该人目录下第a段语音信息
        tmp = list_uttrs[a]
        # mel_spectrogram and normed_f0
        melsp, f0_org = tmp
        if len(melsp) > self.segment_size:
            max_start = len(melsp) - self.segment_size
            left = np.random.randint(0, max_start)
        else:
            left = 0
        melsp = melsp[left:left + self.segment_size, :]
        f0_org = f0_org[left:left + self.segment_size]
        # 使得a的数据都在[0, 1]区间内，即小于0的数字改成0，大于1的改成1
        melsp = np.clip(melsp, 0, 1)
        # 用0填充a，使得第一维大小为segment_size=128

        melsp = np.pad(melsp, ((0, self.segment_size - len(melsp)), (0, 0)), 'constant')
        f0_org = np.pad(f0_org[:, np.newaxis], ((0, self.segment_size - len(f0_org)), (0, 0)), 'constant')

        len_org = len(melsp)

        return melsp, emb_org, accent_org, f0_org, len_org


    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens


class MyCollator(object):
    def __init__(self, hparams):
        self.min_len_seq = hparams.min_len_seq
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad

    def __call__(self, batch):
        # batch[i] is a tuple of __getitem__ outputs
        new_batch = []
        for token in batch:
            # Random resampling operation, be regarded as an information bottleneck on rhythm
            aa, b, c = token
            # 随机选取截取长度，源文件1.5s ~ 3s, 返回 ndarray of ints
            # the first step is to divide the input into segments of random lengths
            len_crop = np.random.randint(self.min_len_seq, self.max_len_seq + 1, size=2)  # 1.5s ~ 3s
            if len_crop[0] >= len(aa):
                left = [0, 0]
            else:
                # 随机选取开始时间
                left = np.random.randint(0, len(aa) - len_crop[0], size=2)
            # pdb.set_trace()

            # the second step is to randomly stretch or squeeze each segment along the time dimension
            # a的第一维是[left[0], left[0] + len_crop[0]], 第二维是[left[0], :], 截取mel，使得时间在1.5s ~ 3s内
            a = aa[left[0]:left[0] + len_crop[0], :]
            # c的第一维是[left[0], left[0] + len_crop[0]], 截取normed_f0
            c = c[left[0]:left[0] + len_crop[0]]

            # 使得a的数据都在[0, 1]区间内，即小于0的数字改成0，大于1的改成1
            a = np.clip(a, 0, 1)

            # 用0填充a，使得第一维大小为max_len_pad=192
            a_pad = np.pad(a, ((0, self.max_len_pad - a.shape[0]), (0, 0)), 'constant')
            c_pad = np.pad(c[:, np.newaxis], ((0, self.max_len_pad - c.shape[0]), (0, 0)), 'constant',
                           constant_values=-1e10)

            new_batch.append((a_pad, b, c_pad, len_crop[0]))

        batch = new_batch

        a, b, c, d = zip(*batch)
        melsp = torch.from_numpy(np.stack(a, axis=0))
        spk_emb = torch.from_numpy(np.stack(b, axis=0))
        pitch = torch.from_numpy(np.stack(c, axis=0))
        len_org = torch.from_numpy(np.stack(d, axis=0))

        return melsp, spk_emb, pitch, len_org


class MultiSampler(Sampler):
    """Samples elements more than once in a single pass through the data.
    """

    def __init__(self, num_samples, n_repeats, shuffle=False):
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(self):
        self.sample_idx_array = torch.arange(self.num_samples, dtype=torch.int64).repeat(self.n_repeats)
        if self.shuffle:
            self.sample_idx_array = self.sample_idx_array[torch.randperm(len(self.sample_idx_array))]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.sample_idx_array)


def get_loader(hparams):
    """Build and return a data loader."""
    dataset = Utterances(hparams.root_dir, hparams.feat_dir, hparams.mode, hparams.train_file, hparams.segment_size)

    # 将一个batch的数据和标签进行合并操作，即对数据进行截取操作，使得音频时间在1.5s ~ 3s间
    # my_collator = MyCollator(hparams)

    # 定义从数据集中提取样本的策略，即生成index的方式，可以顺序也可以乱序
    sampler = MultiSampler(len(dataset), hparams.samplier, shuffle=hparams.shuffle)

    # 不管传给DataLoader的num_workers等于几，Dataset的构造函数都只会被创建一次，即不同的worker是使用同一个Dataset；
    # 但是worker_init_fn会被调用num_workers次，用于初始化每个worker自己独有的数据，避免了和其他worker使用公用的数据，进而加快数据加载速度。
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=hparams.batch_size,
                                  sampler=sampler,
                                  num_workers=hparams.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn)
    # collate_fn=my_collator)
    return data_loader
