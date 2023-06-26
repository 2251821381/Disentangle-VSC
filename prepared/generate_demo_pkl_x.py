import csv
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
from kaldiio import ReadHelper

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('../assets/3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
segment_size = 128
num_uttrs = 10
len_crop = 128

mel_dir = '../dataset/VCTK/test/spmel'
f0_dir = '../dataset/VCTK/test/raptf0'



ACCENTS_TOTAL = ['English', 'Scottish', 'NorthernIrish', 'Irish', 'Indian', 'Welsh', 'American', 'Canadian',
                 'SouthAfrican', 'Australian', 'NewZealand', 'British']


def make_spect_f0(speaker, fileName):
    """读取mel和f0"""
    print(os.path.join(speaker, fileName))
    mel = np.load(os.path.join(mel_dir, speaker, fileName))
    # 读取f0
    f0 = np.load(os.path.join(f0_dir, speaker, fileName)).T
    print(mel.shape)

    # 使得每个片段都是hparams.freq=8的倍数，encoder里面upsampling时不出错
    a, b = divmod(len(mel), 8)
    if b != 0:
        left = ((a + 1) * 8 - len(mel)) // 2
        right = (a + 1) * 8 - len(mel) - left
        mel = np.pad(mel, ((left, right), (0, 0)), 'constant', constant_values=(0, 0))
        f0 = np.pad(f0, (left, right), 'constant', constant_values=(0, 0))
    # mel = mel[:148, :]
    # f0 = f0[:128]

    return mel, f0


def get_speaker_id(speaker):
    # get speaker id and accents
    spk_info_txt = '../dataset/VCTK/speaker-info.txt'
    f = open(spk_info_txt, 'r')
    for i, line in enumerate(f):
        if i == 0:
            continue
        else:
            # [ID, AGE, GENDER, ACCENTS, REGION, COMMENTS]
            tmp = line.split()
            if tmp[0] == speaker:
                a_index = 15
                if tmp[3] in ACCENTS_TOTAL:
                    a_index = ACCENTS_TOTAL.index(tmp[3])
                print('speaker: %s, accent: %s, a_index: %s' % (tmp[0], tmp[3], a_index))
                return a_index, tmp[3]


def make_test_metadata(subdirList):
    """获取基本数据"""
    speakers = []
    for speaker in sorted(subdirList):
        if speaker not in [source_speaker.split('_')[0], target_speaker.split('_')[0]]:
            continue
        print(f'Processing speaker : %s' % speaker)

        # speaker_name
        utterances = [speaker]
        _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))

        # make speaker embedding
        reader = ReadHelper('ark:cat ../dataset/spk_xvector.ark |')
        embs = np.zeros((512,), dtype=np.float32)
        for speaker_id, xvector in reader:
            # (speaker_id, xvector)
            if speaker_id == speaker:
                embs = xvector
                break
        embs = embs[np.newaxis, :]
        utterances.append(embs)
        print('emb.shape: ', embs.shape)

        # speaker accent id
        a_index, accent = get_speaker_id(speaker)
        accent_id = np.zeros((16,), dtype=np.float32)
        accent_id[a_index] = 1
        accent_id = accent_id[np.newaxis, :]
        utterances.append(accent_id)
        print('accent_id.shape: %s', accent_id.shape)
        utterances.append(accent)

        _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
        for fileName in sorted(fileList):
            if fileName not in [source_speaker, target_speaker]:
                continue
            mel, f0_norm = make_spect_f0(speaker=speaker, fileName=fileName)
            print(mel.shape)
            print(f0_norm.shape)
            length = mel.shape[0]
            utterance_name = fileName.split('.')[0].split('_')[1]
            # (mel, f0_norm, length, utterance_name)
            utterances.append((mel, f0_norm, length, utterance_name))
        speakers.append(utterances)
    # target的序号要比source小的时候用来颠倒顺序
    # speakers[0], speakers[1] = speakers[1], speakers[0]
    print('first:' + speakers[0][0])
    len1 = speakers[0][4][2]
    len2 = speakers[1][4][2]
    if (len1 > len2):
        speaker, spkid, accent_id, accent, (mel, f0_norm, length, utterance_name) = speakers.pop(1)
        mel, f0_norm = pad_mel(len1, length, mel, f0_norm)
        length = len1
        speakers.append((speaker, spkid, accent_id, accent, (mel, f0_norm, length, utterance_name)))

    elif len1 < len2:
        speakers[0], speakers[1] = speakers[1], speakers[0]
        speaker, spkid, accent_id, accent, (mel, f0_norm, length, utterance_name) = speakers.pop(1)
        mel, f0_norm = pad_mel(len2, length, mel, f0_norm)
        length = len2
        speakers.append((speaker, spkid, accent_id, accent, (mel, f0_norm, length, utterance_name)))
        speakers[0], speakers[1] = speakers[1], speakers[0]
        # max_start = length - len1
        # # 随机选取开始时间
        # left = np.random.randint(0, max_start)
        # mel = mel[left:left + len1, :]
        # f0_norm = f0_norm[left:left + len1]
    print('first: %s, len: %d' % (speakers[0][0], speakers[0][4][2]))
    print('second: %s, len: %d' % (speakers[1][0], speakers[1][4][2]))
    return speakers


def pad_mel(len1, length, mel, f0_norm):
    # # 重复
    # mel_tmp = mel
    # f0_tmp = f0_norm
    # while len1 >= len(mel):
    #     mel = np.concatenate([mel, mel_tmp], 0)
    #     f0_norm = np.concatenate([f0_norm, f0_tmp], 0)
    # mel = mel[:len1, :]
    # f0_norm = f0_norm[:len1]
    left = (len1 - length) // 2
    right = len1 - length - left
    # 前后用0填充
    mel = np.pad(mel, ((left, right), (0, 0)), 'constant', constant_values=(0, 0))
    f0_norm = np.pad(f0_norm, (left, right), 'constant', constant_values=(0, 0))
    return mel, f0_norm


if __name__ == '__main__':
    # test
    # source_speaker = 'p225_001_mic1.npy'
    # target_speaker = 'p226_001_mic1.npy'

    # -------------------------parallel data----------------------------------------------
    # ---------------1.M2M-----------------------
    # source_speaker = 'p226_001_mic1.npy'
    # target_speaker = 'p251_001_mic1.npy'
    # source_speaker = 'p227_003_mic1.npy'
    # target_speaker = 'p237_003_mic1.npy'
    source_speaker = 'p226_010_mic1.npy'
    target_speaker = 'p227_010_mic1.npy'
    # ---------------2.M2F-----------------------
    # source_speaker = 'p226_001_mic1.npy'
    # target_speaker = 'p231_001_mic1.npy'
    # source_speaker = 'p226_003_mic1.npy'
    # target_speaker = 'p250_003_mic1.npy'
    # source_speaker = 'p226_002_mic1.npy'
    # target_speaker = 'p231_003_mic1.npy'
    # ---------------3.F2M-----------------------
    # source_speaker = 'p231_001_mic1.npy'
    # target_speaker = 'p232_001_mic1.npy'
    # source_speaker = 'p231_003_mic1.npy'
    # target_speaker = 'p245_003_mic1.npy'
    # ---------------4.F2F-----------------------
    # source_speaker = 'p231_001_mic1.npy'
    # target_speaker = 'p248_001_mic1.npy'
    # source_speaker = 'p228_003_mic1.npy'
    # target_speaker = 'p229_003_mic1.npy'

    dirName, subdirList, _ = next(os.walk(mel_dir))
    speakers = make_test_metadata(subdirList)
    # demo_name = '{}_{}_{}_test_d'.format(source_speaker.split('_')[1], source_speaker.split('_')[0],
    #                                    target_speaker.split('_')[0])
    demo_name = '{}_{}_{}_{}_{}_{}_test_x'.format(source_speaker.split('_')[1], source_speaker.split('_')[0], speakers[0][3],
                                             target_speaker.split('_')[1], target_speaker.split('_')[0], speakers[1][3])
    print(f'generate {demo_name}.pkl')
    with open(os.path.join('../dataset/VCTK/test/parallel', demo_name + '.pkl'), 'wb') as handle:
    # with open(os.path.join('../dataset/VCTK/test', demo_name + '.pkl'), 'wb') as handle:
        pickle.dump(speakers, handle)
