import os

# 将wav语音由48kHz的采样频率改成16kHz的采样频率，单声道

# absolute_path = r'D:\workspace\projects\voiceConversion\prepared\data_aishell\dataset'
absolute_path = r'G:\VCTK-Corpus-0.92'
source_path = r"wav48_silence_trimmed"
target_path = r"wav16"

dirName, subdirList, _ = next(os.walk(os.path.join(absolute_path, source_path)))
print('Found directory: %s' % dirName)
for speaker in sorted(subdirList):
    print('Processing speaker : %s' % speaker)

    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))

    for fileName in sorted(fileList):
        if not os.path.exists(os.path.join(absolute_path, target_path, speaker)):
            os.makedirs(os.path.join(absolute_path, target_path, speaker))
        flac = os.path.join(absolute_path, source_path, speaker, fileName)
        wav_name = (fileName.split('.')[0]) + '.wav'
        wav = os.path.join(absolute_path, target_path, speaker, wav_name)
        command = 'D://Tools//ffmpeg-master-latest-win64-gpl//bin//ffmpeg -i ' + flac + ' -ac 1 -ar 16000 ' + wav
        print(command)

        os.system(command)
