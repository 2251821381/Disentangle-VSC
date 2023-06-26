import os
import pickle
import numpy as np
from kaldiio import ReadHelper

mel_dir = 'VCTK/dataset/spmel'
f0_dir = 'VCTK/dataset/raptf0'

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
    # make speaker embedding
    reader = ReadHelper('ark:cat ../dataset/spk_xvector.ark |')
    embs = np.zeros((512,), dtype=np.float32)
    for speaker_id, xvector in reader:
        # (speaker_id, xvector)
        if speaker_id == speaker:
            embs = xvector
            break
    embs = embs[np.newaxis, :]
    return embs


def get_accent(speaker):
    spk_info_txt = 'VCTK/speaker-info.txt'
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


def make_test_metadata(mel_dir, src_spk, trg_spk):
    """获取基本数据"""
    global length
    src_spkid = get_speaker_id(src_spk)
    trg_spkid = get_speaker_id(trg_spk)

    # source speaker accent id
    src_index, src_accent = get_accent(src_spk)
    src_accent_id = np.zeros((16,), dtype=np.float32)
    src_accent_id[src_index] = 1
    src_accent_id = src_accent_id[np.newaxis, :]

    # target speaker accent id
    trg_index, trg_accent = get_accent(src_spk)
    trg_accent_id = np.zeros((16,), dtype=np.float32)
    trg_accent_id[trg_index] = 1
    trg_accent_id = trg_accent_id[np.newaxis, :]

    src_path = os.path.join(mel_dir, src_spk)
    trg_path = os.path.join(mel_dir, trg_spk)
    dirName, _, subdirList = next(os.walk(src_path))
    i = 0
    for src_name in sorted(subdirList):
        if i >= 10:
            break
        i = i + 1
        speakers = []
        uttr_id = src_name.split('.')[0].split('_')[1]
        suffix = src_name[4:]
        trg_name = trg_spk + suffix
        if not os.path.exists(os.path.join(mel_dir, src_spk, src_name)) or not os.path.exists(
            os.path.join(mel_dir, trg_spk, trg_name)):
            i = i - 1
            continue
        print("source: " + src_name + ", target:" + trg_name)
        src_mel, src_f0 = make_spect_f0(src_spk, src_name)
        trg_mel, trg_f0 = make_spect_f0(trg_spk, trg_name)

        if src_mel.shape[0] > trg_mel.shape[0]:
            trg_mel, trg_f0 = pad_mel(src_mel.shape[0], trg_mel.shape[0], trg_mel, trg_f0)
        elif src_mel.shape[0] < trg_mel.shape[0]:
            src_mel, src_f0 = pad_mel(trg_mel.shape[0], src_mel.shape[0], src_mel, src_f0)

        length = max(src_mel.shape[0], trg_mel.shape[0])

        src = [src_spk, src_spkid, src_accent_id, src_accent, [src_mel, src_f0, length, uttr_id]]
        trg = [trg_spk, trg_spkid, trg_accent_id, trg_accent, [trg_mel, trg_f0, length, uttr_id]]
        print(f'source_speaker.shape: {src_mel.shape}, target_speaker.shape: {trg_mel.shape}')

        speakers.append(src)
        speakers.append(trg)

        # demo_name = '{}_{}_{}'.format(uttr_id, src_spk, trg_spk)
        demo_name = '{}_{}_{}_{}_{}_{}_test_x'.format(uttr_id, src_spk, src_accent,
                                                      uttr_id, trg_spk, trg_accent)

        dir_name = 'VCTK/test/pairs/' + src_spk + '_' + trg_spk + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(os.path.join(dir_name, demo_name + '.pkl'), 'wb') as handle:
            pickle.dump(speakers, handle)


def getSpeechInfo(speaker, fileName):
    mel, f0_norm = make_spect_f0(speaker, fileName)

    return mel, f0_norm, mel.shape[0]


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
    source_speaker = 'p300'
    target_speaker = 'p304'

    make_test_metadata(mel_dir, source_speaker, target_speaker)
