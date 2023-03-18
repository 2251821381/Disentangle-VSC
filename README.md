# Parallel Voice Conversion based on Feature Disentanglement Using Variational Auto-Encoder(FDVAE-VC)
This repository provides a Pytorch implementation of FDVAE-VC, which can separate the speech into content, timbre, rhythm, pitch, and accent.

# Dependencies
- Python 3.7
- Numpy
- Scipy
- Pytorch
- librosa
- pysptk
- soundfile
- matplotlib
- wavenet_vocoder `pip install wavenet_vocoder==0.1.11

# Prepared
Extract the mel-spectrogram and F0 from the wav files.

    python prepared/make_spect_f0.py

Then generate the dataset file 'train.pkl' for training model.

    python prepared/make_metadata.py
    
# Train model
Run the main training script

    python main.py
    
# Converted
First generate a test pkl file, which contains the information of the source speaker and the target speaker.

    [source_speaker_Name, one-hot(source speaker embedding), accent_org, [source_mel-spec, source_normd-F0, source_length, source_utterance_name]]

    [target_speaker_Name, one-hot(target speaker embedding), accent_org, [target_mel-spec, target_normd-F0, target_length, target_utterance_name]]

Then run the demo.py file for conversion.

    python demo.py

And the wavenet_vocoder can be downloaded [here](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view)
