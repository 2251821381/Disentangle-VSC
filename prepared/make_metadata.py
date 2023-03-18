import os
import pickle
import numpy as np

dataset_dir = '../dataset/VCTK/dataset'
rootDir = '../dataset/VCTK/dataset/spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

ACCENTS_TOTAL = ['English', 'Scottish', 'NorthernIrish', 'Irish', 'Indian', 'Welsh', 'American', 'Canadian',
                 'SouthAfrican', 'Australian', 'NewZealand', 'British']


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

                return i, a_index


speakers = []
i = 0
for speaker in sorted(subdirList):
    if i == 26:
        break
    print('Processing speaker: %s' % speaker)
    i = i + 1
    utterances = [speaker]
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))

    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion
    spkid = np.zeros((82,), dtype=np.float32)
    # gender, index = spk2gen[speaker]
    # index = int(index)
    spk_index, a_index = get_speaker_id(speaker)
    # print(f'gender: %s, index: %s' % (gender, index))
    spkid[spk_index] = 1

    utterances.append(spkid)

    accent_id = np.zeros((16,), dtype=np.float32)
    accent_id[a_index] = 1
    utterances.append(accent_id)

    # create file list
    j = 0
    for fileName in sorted(fileList):
        if j == 20:
            break
        utterances.append(os.path.join(speaker, fileName))
    speakers.append(utterances)

with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)
