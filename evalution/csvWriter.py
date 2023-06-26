import csv

# csv header
fieldnames = ['id', 'ref', 'syn']
# ref：参考语音，
# syn：合成语音

# csv data
rows = [
    # {
    #     'id': 1,
    #     'ref': '../dataset/VCTK/test/wav16/p304/p304_001_mic1.wav',
    #     'syn': '../results/VCTK121/001_p300_American_001_p304_NorthernIrish_test_x_121/001_p300_p304_AFRU.wav'
    #  },
    {
        'id': 2,
        'ref': '../dataset/VCTK/test/wav16/p304/p304_002_mic1.wav',
        'syn': '../results/VCTK121/002_p300_American_002_p304_NorthernIrish_test_x_121/002_p300_p304_AFRU.wav'
     },
    {
        'id': 3,
        'ref': '../dataset/VCTK/test/wav16/p304/p304_003_mic1.wav',
        'syn': '../results/VCTK121/003_p300_American_003_p304_NorthernIrish_test_x_121/003_p300_p304_AFRU.wav'
     },
    {
        'id': 4,
        'ref': '../dataset/VCTK/test/wav16/p304/p304_004_mic1.wav',
        'syn': '../results/VCTK121/004_p300_American_004_p304_NorthernIrish_test_x_121/004_p300_p304_AFRU.wav'
     },
    {
        'id': 5,
        'ref': '../dataset/VCTK/test/wav16/p304/p304_005_mic1.wav',
        'syn': '../results/VCTK121/005_p300_American_005_p304_NorthernIrish_test_x_121/005_p300_p304_AFRU.wav'
     },
    {
        'id': 6,
        'ref': '../dataset/VCTK/test/wav16/p304/p304_006_mic1.wav',
        'syn': '../results/VCTK121/006_p300_American_006_p304_NorthernIrish_test_x_121/006_p300_p304_AFRU.wav'
     },
    {
        'id': 7,
        'ref': '../dataset/VCTK/test/wav16/p304/p304_007_mic1.wav',
        'syn': '../results/VCTK121/007_p300_American_007_p304_NorthernIrish_test_x_121/007_p300_p304_AFRU.wav'
     },
    {
        'id': 8,
        'ref': '../dataset/VCTK/test/wav16/p304/p304_008_mic1.wav',
        'syn': '../results/VCTK121/008_p300_American_008_p304_NorthernIrish_test_x_121/008_p300_p304_AFRU.wav'
     },
    {
        'id': 9,
        'ref': '../dataset/VCTK/test/wav16/p304/p304_009_mic1.wav',
        'syn': '../results/VCTK121/009_p300_American_009_p304_NorthernIrish_test_x_121/009_p300_p304_AFRU.wav'
     },
    {
        'id': 10,
        'ref': '../dataset/VCTK/test/wav16/p304/p304_010_mic1.wav',
        'syn': '../results/VCTK121/010_p300_American_010_p304_NorthernIrish_test_x_121/010_p300_p304_AFRU.wav'
     },


]

with open('eval_data_p300_p304.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)
