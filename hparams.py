from tfcompat.hparam import HParams

# NOTE: If you want full control for models architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = HParams(
    dim_pre=512,
    conv_dim=1024,

    # SpeakerEncoder
    n_conv_blocks=6,
    dim_speaker=512,
    dim_out=128,
    dropout_rate=0,

    # ContentEncoder
    dim_enc=512,    # Conv Dim
    dim_neck=8,     # BLSTM Dim
    freq=4,         # downsample Factor
    dim_c=16,
    c_dropout=0.1,

    # RhythmEncoder
    dim_enc_2=128,  # Conv Dim
    dim_neck_2=2,  # BLSTM Dim
    freq_2=4,  # downsample Factor
    dim_r=4,

    # PitchEncoder
    dim_enc_3=256,  # Conv Dim
    dim_neck_3=32,  # BLSTM Dim
    freq_3=8,  # downsample Factor
    dim_f=64,

    # AccentEncoder
    dim_enc_4=256,
    dim_neck_4=1,
    freq_4=8,
    dim_a=2,

    dim_freq=80,  # in_channels, content, rhythm
    # speaker embedding: d-vector
    dim_spk_emb=256,  # dimension of the speaker embedding
    # speaker embedding : one-hot
    # dim_spk_emb=82,  # dimension of the speaker embedding
    dim_accent_emb=16,  # dimension of the speaker accent embedding
    dim_f0=257,  # in_channels, f0

    dim_dec=512,
    len_raw=128,
    chs_grp=16,  # GNorm时分组的参数

    # Convenient models builder
    builder="wavenet",

    hop_size=256,
    log_scale_min=float(-32.23619130191664),
    out_channels=10 * 3,
    layers=24,
    stacks=4,
    residual_channels=512,
    gate_channels=512,  # split into 2 groups internally for gated activation
    skip_out_channels=256,
    cin_channels=80,
    gin_channels=-1,  # i.e., speaker embedding dim
    weight_normalization=True,
    n_speakers=-1,
    dropout=1 - 0.95,
    kernel_size=3,
    upsample_conditional_features=True,
    upsample_scales=[4, 4, 4, 4],
    freq_axis_kernel_size=3,
    legacy=True,

    # interp
    # random resampling
    # the input signal is dividded into segments, whose length is randomly uniformly drawn from 19 frames to 32 frames
    min_len_seg=19,
    max_len_seg=32,
    min_len_seq=64,
    max_len_seq=128,
    max_len_pad=192,

    # data loader
    root_dir='dataset/VCTK/dataset/spmel',
    feat_dir='dataset/VCTK/dataset/raptf0',
    # speaker embedding : one-hot
    validate_dir='dataset/VCTK/test/001_p225_p226_test.pkl',
    train_file='dataset/VCTK/dataset/spmel/train.pkl',

    batch_size=32,
    mode='train',
    shuffle=True,
    num_workers=0,
    samplier=8,

    # generate mel-spectrogram
    sample_rate=16000,
    n_fft=1024,
    n_mels=80,
    fmin=90,
    fmax=7600,
    hop_length=256,

    segment_size=128,
    model_dim=2,
    head_count=1,
    dim_per_head=2,
    mi_weight=0.01,

    n_heads=2,

)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in values]
    return 'Hyperparameters:\n' + '\n'.join(hp)

