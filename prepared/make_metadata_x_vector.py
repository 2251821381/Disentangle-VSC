"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
from kaldiio import ReadHelper

num_uttrs = 10
len_crop = 128

ACCENTS_TOTAL = ['English', 'Scottish', 'NorthernIrish', 'Irish', 'Indian', 'Welsh', 'American', 'Canadian',
                 'SouthAfrican', 'Australian', 'NewZealand', 'British']


def get_speaker_accent_id(speaker):
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
                return a_index


# Directory containing mel-spectrograms
rootDir = '../dataset/VCTK/dataset/spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

# reader = ReadHelper('ark:cat spk_xvector.ark |')

speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
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
    utterances.append(embs)

    # speaker accent id
    a_index = get_speaker_accent_id(speaker)
    accent_id = np.zeros((16,), dtype=np.float32)
    accent_id[a_index] = 1
    utterances.append(accent_id)

    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker + "/", fileName))
    speakers.append(utterances)

with open(os.path.join(rootDir, 'train_x.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)
