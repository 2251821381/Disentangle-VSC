import os
import soundfile as sf

flac_directory = 'G://VCTK-Corpus-0.92//wav48_silence_trimmed'
wav_directory = 'G://VCTK-Corpus-0.92//wav16'

# 创建保存 WAV 文件的目录
os.makedirs(wav_directory, exist_ok=True)

# 遍历 FLAC 文件
dirName, subdirList, _ = next(os.walk(flac_directory))
print('Found directory: %s' % dirName)
for speaker in sorted(subdirList):
    print('Processing speaker : %s' % speaker)
    if not os.path.exists(os.path.join(wav_directory, speaker)):
        os.makedirs(os.path.join(wav_directory, speaker))

    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
    for file in sorted(fileList):
        flac = os.path.join(flac_directory, speaker, file)
        wav_name = (file.split('.')[0]) + '.wav'
        wav = os.path.join(wav_directory, speaker, wav_name)
        # 转换为 WAV 文件，采样频率为 16000 Hz
        data, samplerate = sf.read(flac)
        sf.write(wav, data, samplerate=16000)
        print(f"Converted {flac} to {wav}")

