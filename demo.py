# demo conversion
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


def loadData(G, demo_path, device):
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


    conditions = ['RFAU', 'source', 'target',
                  'R', 'U', 'F', 'A', 'FAU',
                  'RF', 'RU', 'RA', 'FU', 'FA', 'AU',
                  'RFU', 'RFA', 'RAU']

    spect_vc = []

    for condition in conditions:
        if condition == 'source':
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_org_pad, f0_org_onehot, emb_org, emb_org, uttr_org_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'target':
            _, _, _, _, x_identic_val = G(uttr_trg_pad, uttr_trg_pad, f0_trg_onehot, emb_trg, emb_trg, uttr_trg_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_trg, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'R':
            # rhythm
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_trg_pad, f0_org_onehot, emb_org, emb_org, uttr_org_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'U':
            # speaker id
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_org_pad, f0_org_onehot, emb_org, emb_trg, uttr_org_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'F':
            # speaker pitch
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_org_pad, f0_trg_onehot, emb_org, emb_org, uttr_org_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'A':
            # speaker accent
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_org_pad, f0_org_onehot, emb_org, emb_org, uttr_trg_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'RF':
            # rhythm, pitch
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_trg_pad, f0_trg_onehot, emb_org, emb_org, uttr_org_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'RU':
            # rhythm, speaker id
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_trg_pad, f0_org_onehot, emb_org, emb_trg, uttr_org_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'RA':
            # rhythm, accent
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_trg_pad, f0_org_onehot, emb_org, emb_org, uttr_trg_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'FU':
            # pitch, speaker id
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_org_pad, f0_trg_onehot, emb_org, emb_trg, uttr_org_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'FA':
            # pitch, accent
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_org_pad, f0_trg_onehot, emb_org, emb_org, uttr_trg_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'AU':
            # accent, speaker id
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_org_pad, f0_org_onehot, emb_org, emb_trg, uttr_trg_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'RFU':
            # rhythm, pitch, speaker id
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_trg_pad, f0_trg_onehot, emb_org, emb_trg, uttr_org_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'RFA':
            # rhythm, pitch, accent
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_trg_pad, f0_trg_onehot, emb_org, emb_org, uttr_trg_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'RAU':
            # rhythm, accent, speaker id
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_trg_pad, f0_org_onehot, emb_org, emb_trg, uttr_trg_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'FAU':
            # pitch, accent, speaker id
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_org_pad, f0_trg_onehot, emb_org, emb_trg, uttr_trg_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
        if condition == 'RFAU':
            # rhythm, pitch, speaker id, accent
            _, _, _, _, x_identic_val = G(uttr_org_pad, uttr_trg_pad, f0_trg_onehot, emb_org, emb_trg, uttr_trg_pad)
            uttr_trg = x_identic_val.cpu().detach().numpy().squeeze(0)
            spect_vc.append(('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], sbmt_j[0], condition), uttr_trg))
    fileName = ('{}_{}_{}_{}'.format(uid_org, sbmt_i[0], uid_trg, sbmt_j[0]))
    drawPlot(spect_vc, fileName)
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
    plt.savefig(f'results/VCTK47/figure/' + fileName + '_47.png', dpi=600)
    plt.close(fig)
    print('Generate the figure finish')


def synthesis(spect_vc, results_path, device):
    """
    spectrogram to waveform
    @param spect_vc:
    @param results_path:
    @return:
    """
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    model = build_model().to(device)
    checkpoint = torch.load("assets/checkpoint_step001000000_ema.pth", map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])

    # 将mel-spectrogram转换成wave file
    for spect in spect_vc:
        name = spect[0]
        c = spect[1]
        print(name)
        waveform = wavegen(device, model, c=c)
        soundfile.write(results_path + name + '.wav', waveform, samplerate=16000)


if __name__ == "__main__":
    demo_path_list = []
    # 1.parallel data,
    #   a.M2M
    parallel_m2m = 'dataset/VCTK/test/parallel/001_p226_English_p251_Indian_test_d.pkl'
    demo_path_list.append(parallel_m2m)
    #   b.M2F
    parallel_m2f = 'dataset/VCTK/test/parallel/001_p226_English_p231_English_test_d.pkl'
    demo_path_list.append(parallel_m2f)
    #   c.F2M
    parallel_f2m = 'dataset/VCTK/test/parallel/001_p231_English_p232_English_test_d.pkl'
    demo_path_list.append(parallel_f2m)
    #   d.F2F
    parallel_f2f = 'dataset/VCTK/test/parallel/001_p231_English_p248_Indian_test_d.pkl'
    demo_path_list.append(parallel_f2f)

    model_path = 'run/VCTK47/models/100000-G.ckpt'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 创建模型，并加载训练的模型文件
    G = Generator(hparams).eval().to(device)
    g_checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    G.load_state_dict(g_checkpoint['models'])

    for demo_path in demo_path_list:
        demo_folder = demo_path.split('/')[-1].split('.')[0]
        print("Process the file " + demo_folder)
        result_path = 'results/VCTK47/' + demo_folder + '_47/'
        if not os.path.exists(os.path.join('results/VCTK47', 'figure')):
            os.makedirs(os.path.join('results/VCTK47', 'figure'))

        spect_vc = loadData(G, demo_path, device)
        print("Data loading is completed!")
        print("Start synthesizing wave files in VCTK47:")
        synthesis(spect_vc, result_path, device)
        print("\n")
