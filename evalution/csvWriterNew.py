import csv
import os

# csv header
fieldnames = ['id', 'ref', 'syn']

dirName = '../results/VCTK121/pairs/'
source_path = '../dataset/VCTK/dataset/wav16/'


def writecsv():
    with open('eval_data_pairs.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        dirList = os.listdir(dirName)
        i = 0
        for subdir in dirList:
            _, _, fileList = next(os.walk(os.path.join(dirName, subdir)))
            for file in fileList:
                uttr_id, src_spk, trg_spk, _ = file.split('_')
                # converted speech vs target speech
                syn = os.path.join(dirName, subdir, file)
                ref = os.path.join(source_path, trg_spk, trg_spk + '_' + uttr_id + '_mic1.wav')
                writer.writerow({'id': i,
                                'ref': ref,
                                'syn': syn})
                i = i + 1


def load_eval_spec(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        samples = list(reader)
    return samples


if __name__ == "__main__":
    writecsv()
    # load_eval_spec('eval_data_pairs.csv')
