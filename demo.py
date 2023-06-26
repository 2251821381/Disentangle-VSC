# demo conversion
import math

import torch
import pickle
import numpy as np
from hparams import hparams
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator as Generator

import torch
import soundfile
import pickle
import os
from synthesis import build_model
from synthesis import wavegen
import matplotlib.pyplot as plt
import threading

import torch.multiprocessing as mp


def loadData(G, demo_path, conditions, device):
    global x_identic_val

    metadata = pickle.load(open(demo_path, "rb"))

    # ----------------------------------the source speaker-------------------------------------------------
    # [speaker_Name, one-hot(speaker embedding), accent_org, [mel-spec, normd-F0, length, utterance_name]]
    sbmt_i = metadata[0]
    # 源说话人的嵌入, source speaker embedding
    emb_org = torch.from_numpy(sbmt_i[1]).to(device)
    accent_org = torch.from_numpy(sbmt_i[2]).to(device)
    # 源说话人的(mel-spectrogram)梅尔频谱图x-org, 归一化量化的log-F0 f0_org, len_org, 说话人ID uid_org
    x_org, f0_org, len_org, uid_org = sbmt_i[4]
    len_org = torch.tensor([len_org]).to(device)
    # np.newaxis的作用就是给数据添加纬度
    uttr_org_pad = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    # 填充后的f0_org_pad,表示的是音高

    f0_org_quantized = quantize_f0_numpy(f0_org)[0]
    f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
    # ndarray to tensor，归一化量化后的log - F0
    f0_org_onehot = torch.from_numpy(f0_org_onehot).to(device)

    # ------------------the target speaker---------------------------------------------------------------
    sbmt_j = metadata[1]
    # 目标说话人的嵌入, the target speaker embedding
    emb_trg = torch.from_numpy(sbmt_j[1]).to(device)
    accent_trg = torch.from_numpy(sbmt_j[2]).to(device)
    x_trg, f0_trg, len_trg, uid_trg = sbmt_j[4]
    len_trg = torch.tensor([len_trg]).to(device)
    # 目标说话人的mel-spectrogram
    uttr_trg_pad = torch.from_numpy(x_trg[np.newaxis, :, :]).to(device)
    # log-F0

    f0_trg_quantized = quantize_f0_numpy(f0_trg)[0]
    f0_trg_onehot = f0_trg_quantized[np.newaxis, :, :]
    # 目标说话人的音高编码
    f0_trg_onehot = torch.from_numpy(f0_trg_onehot).to(device)
    # with torch.no_grad():
    #     # 重建音高轮廓，解开音高轮廓中的音高和节奏信息，从而可以只进行节奏转换
    #     f0_pred = P(uttr_org_pad, f0_trg_onehot)[0]
    #     f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
    #     f0_con_onehot = torch.zeros((1, len_trg, 257), device=device)
    #     f0_con_onehot[0, torch.arange(len_trg), f0_pred_quantized] = 1

    # 'R':Rhythm, 'F':Pitch, 'U':Timbre, 'A': Accent
    # conditions = ['RAU']
    # conditions = ['RFAU', 'source', 'target',
    #               'R', 'U', 'F', 'A', 'FAU',
    #               'RF', 'RU', 'RA', 'FU', 'FA', 'AU',
    #               'RFU', 'RFA', 'RAU']
    # conditions = ['AFRU', 'source', 'target', 'A', 'AF', 'AR', 'AU', 'AFR', 'AFU', 'ARU',
    #               'F', 'FR', 'FU', 'FRU', 'R', 'RU', 'U']
    map = {
        # content, rhythm, pitch, accent, timbre
        'AFRU': (uttr_org_pad, uttr_trg_pad, f0_trg_onehot, f0_trg_onehot, emb_trg),
        'source': (uttr_org_pad, uttr_org_pad, f0_org_onehot, f0_org_onehot, emb_org),
        'target': (uttr_trg_pad, uttr_trg_pad, f0_trg_onehot, f0_trg_onehot, emb_trg),
        'A': (uttr_org_pad, uttr_org_pad, f0_org_onehot, f0_trg_onehot, emb_org),
        'AF': (uttr_org_pad, uttr_org_pad, f0_trg_onehot, f0_trg_onehot, emb_org),
        'AFR': (uttr_org_pad, uttr_trg_pad, f0_trg_onehot, f0_trg_onehot, emb_org),
        'AFU': (uttr_org_pad, uttr_org_pad, f0_trg_onehot, f0_trg_onehot, emb_trg),
        'AR': (uttr_org_pad, uttr_trg_pad, f0_org_onehot, f0_trg_onehot, emb_org),
        'ARU': (uttr_org_pad, uttr_trg_pad, f0_org_onehot, f0_trg_onehot, emb_trg),
        'AU': (uttr_org_pad, uttr_org_pad, f0_org_onehot, f0_trg_onehot, emb_trg),
        'F': (uttr_org_pad, uttr_org_pad, f0_trg_onehot, f0_org_onehot, emb_org),
        'FR': (uttr_org_pad, uttr_trg_pad, f0_trg_onehot, f0_org_onehot, emb_org),
        'FRU': (uttr_org_pad, uttr_trg_pad, f0_trg_onehot, f0_org_onehot, emb_trg),
        'FU': (uttr_org_pad, uttr_org_pad, f0_trg_onehot, f0_org_onehot, emb_trg),
        'R': (uttr_org_pad, uttr_trg_pad, f0_org_onehot, f0_org_onehot, emb_org),
        'RU': (uttr_org_pad, uttr_trg_pad, f0_org_onehot, f0_org_onehot, emb_trg),
        'U': (uttr_org_pad, uttr_org_pad, f0_org_onehot, f0_org_onehot, emb_trg)
    }
    spect_vc = []
    for condition in conditions:
        content, rhythm, pitch, accent, timbre = map.get(condition)
        _, _, _, _, x_identic_val = G(content, rhythm, pitch, accent, timbre)
        uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
        spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
    fileName = ('{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0]))
    # drawPlot(spect_vc, fileName)
    return spect_vc


def drawPlot(spect_vc, fileName):
    melsp_out_RFAU = (spect_vc[0])[1].T
    melsp_source = (spect_vc[1])[1].T
    melsp_target = (spect_vc[2])[1].T
    melsp_out_R = (spect_vc[3])[1].T
    melsp_out_U = (spect_vc[4])[1].T
    melsp_out_F = (spect_vc[5])[1].T
    melsp_out_A = (spect_vc[6])[1].T
    melsp_out_FAU = (spect_vc[7])[1].T

    min_value = np.min(np.hstack(
        [melsp_out_RFAU, melsp_source, melsp_target,
         melsp_out_R, melsp_out_U, melsp_out_F, melsp_out_A, melsp_out_FAU]))
    max_value = np.max(np.hstack(
        [melsp_out_RFAU, melsp_source, melsp_target,
         melsp_out_R, melsp_out_U, melsp_out_F, melsp_out_A, melsp_out_FAU]))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, sharex=True)
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, sharex=True)
    ax1.imshow(melsp_source, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    ax1.set_title('source')
    ax2.imshow(melsp_target, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    ax2.set_title('target')
    ax3.imshow(melsp_out_RFAU, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    ax3.set_title('RFAU')
    ax4.imshow(melsp_out_FAU, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    ax4.set_title('FAU')
    ax5.imshow(melsp_out_R, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    ax5.set_title('R')
    ax6.imshow(melsp_out_U, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    ax6.set_title('U')
    ax7.imshow(melsp_out_F, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    ax7.set_title('F')
    ax8.imshow(melsp_out_A, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    ax8.set_title('A')

    # 调整子图间的上下距离，左右距离用wspace
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.savefig(f'results/VCTK122/figure/' + fileName + '_122.png', dpi=600)
    plt.close(fig)
    print('Generate the figure finish')


def synthesis(spect_vc, results_path, device):
    """
    spectrogram to waveform
    @param spect_vc:
    @param results_path:
    @return:
    """
    print('device:', device, ',spect length:', len(spect_vc))

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    model = build_model().to(device)
    checkpoint = torch.load("assets/checkpoint_step001000000_ema.pth", map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])

    # 将mel-spectrogram转换成wave file
    for spect in spect_vc:
        name = spect[0]
        c = spect[1]
        print(mp.current_process(), name)
        waveform = wavegen(device, model, c=c)
        soundfile.write(results_path + name + '.wav', waveform, samplerate=16000)
        print(results_path + name + '.wav has been generated successfully!')


def gen_pairs(G, device):
    meta_path = 'dataset/VCTK/test/pairs/p226_p227/'
    dirName, _, subdirList = next(os.walk(meta_path))
    result_path = 'results/VCTK122/' + 'p226_p227/'
    conditions = ['AFRU']

    for demo in sorted(subdirList):
        print("Process the file " + demo)
        spect_vc = loadData(G, os.path.join(meta_path, demo), conditions, device)
        print("Data loading is completed!")
        print("Start synthesizing wave files:")
        synthesis(spect_vc, result_path, device)

def gen_list(G, device):
    demo_path_list = [
        'dataset/VCTK/test/parallel/002_p300_American_002_p304_NorthernIrish_test_x.pkl',   # parallel,F2M
        'dataset/VCTK/test/parallel/003_p300_American_003_p304_NorthernIrish_test_x.pkl',   # parallel,F2M
        'dataset/VCTK/test/parallel/004_p300_American_004_p304_NorthernIrish_test_x.pkl',   # parallel,F2M
        'dataset/VCTK/test/parallel/005_p300_American_005_p304_NorthernIrish_test_x.pkl',   # parallel,F2M
        'dataset/VCTK/test/parallel/006_p300_American_006_p304_NorthernIrish_test_x.pkl',   # parallel,F2M
        'dataset/VCTK/test/parallel/007_p300_American_007_p304_NorthernIrish_test_x.pkl',   # parallel,F2M
        'dataset/VCTK/test/parallel/008_p300_American_008_p304_NorthernIrish_test_x.pkl',   # parallel,F2M
        'dataset/VCTK/test/parallel/009_p300_American_009_p304_NorthernIrish_test_x.pkl',   # parallel,F2M
        'dataset/VCTK/test/parallel/010_p300_American_010_p304_NorthernIrish_test_x.pkl',   # parallel,F2M
        # 'dataset/VCTK/test/parallel/001_p226_English_p251_Indian_test_x.pkl',   # parallel,M2M
        # 'dataset/VCTK/test/parallel/001_p302_Canadian_001_p315_American_test_x.pkl',   # parallel,M2M
        # 'dataset/VCTK/test/parallel/003_p227_English_p237_Scottish_test_x.pkl',
        # 'dataset/VCTK/test/parallel/001_p304_NorthernIrish_001_p314_SouthAfrican_test_x.pkl',  # parallel,M2F
        # 'dataset/VCTK/test/parallel/003_p226_English_p250_English_test_x.pkl',
        # 'dataset/VCTK/test/parallel/001_p231_English_p232_English_test_x.pkl',  # parallel,F2M
        # 'dataset/VCTK/test/parallel/001_p314_SouthAfrican_001_p316_Canadian_test_x.pkl',
        # 'dataset/VCTK/test/parallel/003_p231_English_p245_Irish_test_x.pkl',
        # 'dataset/VCTK/test/parallel/001_p231_English_p248_Indian_test_x.pkl',   # parallel,F2F
        # 'dataset/VCTK/test/parallel/001_p307_Canadian_001_p313_Irish_test_x.pkl',   # parallel,F2F
        # 'dataset/VCTK/test/parallel/003_p228_English_p229_English_test_x.pkl',
        # 'dataset/VCTK/test/non-parallel/002_p226_English_003_p251_Indian_test_x.pkl',   # non-parallel,M2M
        # 'dataset/VCTK/test/non-parallel/002_p226_English_003_p231_English_test_x.pkl',  # non-parallel,M2F
        # 'dataset/VCTK/test/non-parallel/002_p231_English_003_p232_English_test_x.pkl',  # non=parallel,F2M
        # 'dataset/VCTK/test/non-parallel/002_p231_English_003_p248_Indian_test_x.pkl'
    ]

    conditions = ['AFRU', 'source', 'target',
                 'A', 'AF', 'AFR', 'AFU', 'AR', 'ARU', 'AU',
                 'F', 'FR', 'FRU', 'FU', 'R', 'RU', 'U']
    mp.set_start_method('spawn', force=True)

    for demo_path in demo_path_list:
        demo_folder = demo_path.split('/')[-1].split('.')[0]
        print("Process the file " + demo_folder)
        result_path = 'results/VCTK122/' + demo_folder + '_122/'
        if not os.path.exists(os.path.join('results/VCTK122', 'figure')):
            os.makedirs(os.path.join('results/VCTK122', 'figure'))

        spect_vc = loadData(G, demo_path, conditions, device)
        print("Data loading is completed!")
        print("Start synthesizing wave files in VCTK122:")

        totle = len(spect_vc)
        print(len(spect_vc))
        jobsNum = 6
        threadNum = math.ceil(totle / jobsNum)
        print(['jobsNum:', jobsNum], ['threadNum:', threadNum])
        # GPU多线程
        processes = []
        for rank in range(threadNum):
            p = mp.Process(target=synthesis,
                           args=(spect_vc[rank * jobsNum:rank * jobsNum + jobsNum], result_path, device,),
                           name='Process' + str(rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()



if __name__ == '__main__':
    model_path = 'run/VCTK122/models/200000-G.ckpt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 创建模型，并加载训练的模型文件
    G = Generator(hparams).eval().to(device)
    g_checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    G.load_state_dict(g_checkpoint['models'])

    gen_list(G, device)
    # gen_pairs(G, device)

