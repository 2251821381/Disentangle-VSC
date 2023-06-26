import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import ceil


class LinearNorm(torch.nn.Module):
    """线性规范层"""

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    """卷积规范层
        in_channels:输入数据通道数
        out_channels:输出数据通道数
        kernel_size:卷积核的尺寸，卷积核的大小为(k,),第二个纬度是由in_channels来决定的，所以实际上卷积大小为kernel_size * in_channels
        stride:卷积步幅大小，默认为1
        padding:输入的每一条边补充0的层数
        dilation:卷积核元素之间的间距，默认为1
        bias:偏差学习标志，默认为True
        w_init_gain:非线性函数名，默认为linear
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        # 若 padding 不存在，则计算一个合适的值，使处于边界的数据也能卷积
        if padding is None:
            # 若 kernel_size 不为奇数则警告
            assert (kernel_size % 2 == 1)
            # 计算合适的 padding 以满足处在边界的数据能被卷积层覆盖
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        # 初始化weight，Glorot初始化，使用均匀分布填充weight
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


def randomSampling(x, S, training):
    if not training:
        return x
    device = x.device
    # print(a.shape)  # torch.Size([8, 128, 256])
    B, N, C = x.shape
    # S = 64
    index = torch.LongTensor(random.sample(range(N), S)).cuda()
    # print(index)
    b = torch.index_select(x, 1, index)
    # print(b.shape)  # torch.Size([32, 64, 64])
    return b


def reparameterize(mean, logvar):
    """
    Will a single z be enough ti compute the expectation for the loss
    @param mean: (Tensor) Mean of the latent Gaussian
    @param logvar: (Tensor) Standard deviation of the latent Gaussian
    @return:
    """
    # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
    std = torch.exp(0.5 * logvar)
    # eps = torch.randn_like(logvar)
    eps = torch.exp(logvar / 2)
    z = mean + eps * std
    return z


class AccentEncoder(nn.Module):
    """Accent Encoder
    """

    def __init__(self, hparams):
        super().__init__()

        self.dim_f0 = hparams.dim_f0
        self.dim_freq = hparams.dim_freq  # 输入数据维数
        self.dim_enc = hparams.dim_enc_4
        self.hidden_dim = hparams.dim_neck_4
        self.dim_a = hparams.dim_a
        self.chs_grp = hparams.chs_grp
        self.freq_4 = hparams.freq_4
        self.dropout = hparams.c_dropout

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_f0 if i == 0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.dim_enc),
                nn.ReLU()
                # nn.GroupNorm(self.dim_enc // self.chs_grp, self.dim_enc)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # self.dropout = nn.Dropout(self.dropout)

        self.a_lstm = nn.LSTM(self.dim_enc, self.hidden_dim, 1, batch_first=True, bidirectional=True)
        # self.interp = InterpLnr(hparams)

    def forward(self, mel):
        """
        :param mel: Speech S, mel-spectrogram # (8, 80, 128)
        :return:
        """
        batch_size, seq, feature = mel.shape
        len_org = torch.tensor(seq)
        mel = mel.transpose(1, 2)
        for conv in self.convolutions:
            mel = conv(mel)
            # mel = mel.transpose(1, 2)
            # # mel = randomSampling(mel, seq, self.training)
            # mel = self.interp(mel, len_org.expand(batch_size))
            # mel = mel.transpose(1, 2)
            # mel = self.dropout(mel)
        mel = mel.transpose(1, 2)

        self.a_lstm.flatten_parameters()
        lstm_out, _ = self.a_lstm(mel)
        # downsampling operation, to reduce the temporal dimension, producing the hidden representations
        out_forward = lstm_out[:, :, :self.hidden_dim]
        out_backward = lstm_out[:, :, self.hidden_dim:]
        # 创建采样编码
        # 降采样，假设降采样因子为k，我们使用基于零的帧索引，对于双向的正向输出，采样t = kn+k-1,对于反向输出，采样t=kn
        # ensure the frames at both ends are covered by at least one forward code and one backward code
        lstm_out = torch.cat((out_forward[:, self.freq_4 - 1::self.freq_4, :],
                              out_backward[:, ::self.freq_4, :]), dim=-1)
        return lstm_out


class RhythmEncoder(nn.Module):
    """Rhythm Encoder
    """

    def __init__(self, hparams):
        super().__init__()

        self.dim_freq = hparams.dim_freq  # 输入数据维数
        self.dim_enc = hparams.dim_enc_2
        self.hidden_dim = hparams.dim_neck_2
        self.dim_r = hparams.dim_r
        self.chs_grp = hparams.chs_grp
        self.freq_2 = hparams.freq_2

        convolutions = []
        for i in range(1):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_freq if i == 0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.dim_enc)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.r_lstm = nn.LSTM(self.dim_enc, self.hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, mel):
        """
        :param mel: Speech S, mel-spectrogram # (8, 80, 128)
        :return:
        """
        batch_size, seq, feature = mel.shape
        mel = mel.transpose(1, 2)
        for conv in self.convolutions:
            mel = F.relu(conv(mel))
        mel = mel.transpose(1, 2)

        self.r_lstm.flatten_parameters()
        lstm_out, _ = self.r_lstm(mel)
        # downsampling operation, to reduce the temporal dimension, producing the hidden representations
        out_forward = lstm_out[:, :, :self.hidden_dim]
        out_backward = lstm_out[:, :, self.hidden_dim:]
        # 创建采样编码
        # 降采样，假设降采样因子为k，我们使用基于零的帧索引，对于双向的正向输出，采样t = kn+k-1,对于反向输出，采样t=kn
        # ensure the frames at both ends are covered by at least one forward code and one backward code
        lstm_out = torch.cat((out_forward[:, self.freq_2 - 1::self.freq_2, :],
                              out_backward[:, ::self.freq_2, :]), dim=-1)
        return lstm_out


class Attention(nn.Module):
    def __init__(self, num_heads, dim):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim
        self.input_dim = dim
        self.query = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.key = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.out = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # (batch_size, num_heads, seq_len, hidden_dim/num_heads)
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # (batch_size, num_heads, seq_len, hidden_dim/num_heads)
        k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # (batch_size, num_heads, seq_len, hidden_dim/num_heads)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim / self.num_heads) ** 0.5
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # attn_output shape: (batch_size, seq_len, hidden_dim)
        output = self.out(attn_output)
        # output shape: (batch_size, seq_len, hidden_dim)
        # output = output.mean(dim=1) # average pooling along the time dimension
        # # output shape: (batch_size, hidden_dim)
        return output


class F0Encoder(nn.Module):
    """F0 encoder
    """

    def __init__(self, hparams):
        super().__init__()

        self.dim_f0 = hparams.dim_f0  # 输入数据维数
        self.dim_enc = hparams.dim_enc_3
        self.hidden_dim = hparams.dim_neck_3
        self.dim_f = hparams.dim_f
        self.chs_grp = hparams.chs_grp
        self.freq_3 = hparams.freq_3

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_f0 if i == 0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.dim_enc),
                nn.ReLU()
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.f_lstm = nn.LSTM(self.dim_enc, self.hidden_dim, 1, batch_first=True, bidirectional=True)
        self.interp = InterpLnr(hparams)

    def forward(self, x):
        """
        @param x: log-F0, shape:(batch_size, seq_length, 257)
        @return: pitch code
        """
        batch_size, seq, feature = x.shape
        len_org = torch.tensor(seq)
        x = x.transpose(1, 2)  # bz x 257 x 128
        for conv in self.convolutions:
            x = conv(x)
            x = x.transpose(1, 2)
            x = self.interp(x, len_org.expand(batch_size))
            x = x.transpose(1, 2)
        x = x.transpose(1, 2)  # bz x 128 x256

        self.f_lstm.flatten_parameters()
        lstm_out, _ = self.f_lstm(x)  # bz x 128 x 64
        # downsampling
        out_forward = lstm_out[:, :, :self.hidden_dim]
        out_backward = lstm_out[:, :, self.hidden_dim:]
        lstm_out = torch.cat((out_forward[:, self.freq_3 - 1::self.freq_3, :],
                              out_backward[:, ::self.freq_3, :]), dim=-1)
        return lstm_out


class ContentEncoder(nn.Module):
    """Content encoder
    """

    def __init__(self, hparams):
        super().__init__()

        self.dim_freq = hparams.dim_freq
        self.dim_enc = hparams.dim_enc
        self.hidden_dim = hparams.dim_neck
        self.dim_c = hparams.dim_c
        self.chs_grp = hparams.chs_grp
        self.freq = hparams.freq
        self.dropout = hparams.c_dropout

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_freq if i == 0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.dim_enc)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.c_lstm = nn.LSTM(self.dim_enc, self.hidden_dim, 2, batch_first=True, bidirectional=True)

        self.mean_layer = LinearNorm(self.hidden_dim * 2, self.dim_c)
        self.std_layer = LinearNorm(self.hidden_dim * 2, self.dim_c)

        self.interp = InterpLnr(hparams)

    def forward(self, mel):
        """
        @param c_org: speaker embedding, (bz, 82)
        @param mel:: speech S, # (bz, 128, 80)
        :return:
        """
        batch_size, seq, feature = mel.shape
        len_org = torch.tensor(seq)
        mel = mel.transpose(1, 2)  # (bz, 128, 80) ---> (bz, 80, 128)
        for conv in self.convolutions:
            mel = F.relu(conv(mel))
            mel = mel.transpose(1, 2)
            mel = self.interp(mel, len_org.expand(batch_size))
            mel = mel.transpose(1, 2)
        mel = mel.transpose(1, 2)  # bz x 16 x 512

        self.c_lstm.flatten_parameters()
        lstm_out, _ = self.c_lstm(mel)
        # downsampling operation ,to reduce the temporal dimension, producing the hidden representations,潜在层表示
        out_forward = lstm_out[:, :, :self.hidden_dim]
        out_backward = lstm_out[:, :, self.hidden_dim:]
        # lstm_out = torch.cat((out_forward, out_backward), dim=-1)
        lstm_out = torch.cat((out_forward[:, self.freq - 1::self.freq, :],
                              out_backward[:, ::self.freq, :]), dim=-1)
        lstm_out = lstm_out.repeat_interleave(self.freq, dim=1)
        mu = self.mean_layer(lstm_out)
        logvar = self.std_layer(lstm_out)

        return mu, logvar, reparameterize(mu, logvar)


class Decoder(nn.Module):
    """Decoder module
    """

    def __init__(self, hparams):
        super().__init__()
        self.c_dim = hparams.dim_c
        self.r_dim = hparams.dim_r
        self.f_dim = hparams.dim_f
        self.a_dim = hparams.dim_a
        self.dim_out = hparams.dim_out

        self.dim_spk_emb = hparams.dim_spk_emb
        self.dim_freq = hparams.dim_freq
        self.dim_accent_emb = hparams.dim_accent_emb
        self.dim_pre = hparams.dim_pre
        self.conv_dim = hparams.conv_dim

        # self.latent_dim = self.c_dim + self.r_dim + self.f_dim + self.dim_spk_emb + self.dim_accent_emb
        self.latent_dim = self.c_dim + self.r_dim + self.f_dim + self.a_dim + self.dim_spk_emb

        # three bidirectional-LSTM layers
        self.lstm1 = nn.LSTM(self.latent_dim, self.dim_pre, 1, batch_first=True)

        # there convolutional layers with 512 channels
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_pre,
                         self.dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm2 = nn.LSTM(self.dim_pre, self.conv_dim, 2, batch_first=True)
        self.attention = Attention(hparams.num_heads, self.conv_dim)
        self.linear_projection = LinearNorm(self.conv_dim, self.dim_freq)

    def forward(self, x):
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)

        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)

        outputs = self.attention(outputs)

        decoder_output = self.linear_projection(outputs)

        return decoder_output


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x


class Generator(nn.Module):
    """SpeechSplit models"""

    def __init__(self, hparams):
        super().__init__()

        # self.dim_freq = hparams.dim_freq
        # self.conv_dim = hparams.conv_dim
        # self.dim_pre = hparams.dim_pre

        self.encoder_c = ContentEncoder(hparams)
        self.encoder_r = RhythmEncoder(hparams)
        self.encoder_f = F0Encoder(hparams)
        self.encoder_a = AccentEncoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet()

        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3
        self.freq_4 = hparams.freq_4

    def forward(self, c_mel, r_mel, f0, accent_f0, spk_emb):
        batch_size, sequence_length, feature_size = c_mel.shape
        # [batch size, sequence len, feature size]
        # mean, logvar, content = self.encoder_c(mel_org)
        mu, logvar, content = self.encoder_c(c_mel)
        rhythm = self.encoder_r(r_mel)
        pitch = self.encoder_f(f0)
        accent = self.encoder_a(accent_f0)

        # content = content.repeat_interleave(self.freq, dim=1)
        rhythm = rhythm.repeat_interleave(self.freq_2, dim=1)
        pitch = pitch.repeat_interleave(self.freq_3, dim=1)
        accent = accent.repeat_interleave(self.freq_4, dim=1)

        decoder_input = torch.cat((content, rhythm, pitch, accent,
                                   spk_emb.unsqueeze(1).expand(-1, sequence_length, -1)),
                                  dim=-1)
        mel_outputs = self.decoder(decoder_input)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        return mu, logvar, content, mel_outputs, mel_outputs_postnet


class InterpLnr(nn.Module):
    """Random resampling"""

    def __init__(self, hparams):
        super().__init__()
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad

        self.min_len_seg = hparams.min_len_seg
        self.max_len_seg = hparams.max_len_seg

        # self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def pad_sequences(self, sequences, seq_length):
        # 输入通道维度
        channel_dim = sequences[0].size()[-1]
        # 输出数据维度
        out_dims = (len(sequences), seq_length, channel_dim)
        # 输出的tensor初始化
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:seq_length]

        return out_tensor

    def forward(self, x, len_seq):

        if not self.training:
            return x

        device = x.device
        batch_size, seq_length, feature = x.shape
        max_num_seg = seq_length // self.min_len_seg + 1
        # indices of each sub segment,
        # shape = (batch_size * self.max_num_seg, self.max_len_seg * 2),
        # each row have the same sequences,[0,1,2,...,self.max_len_seg*2]
        indices = torch.arange(self.max_len_seg * 2, device=device) \
            .unsqueeze(0).expand(batch_size * max_num_seg, -1)
        # scales of each sub segment
        # 从区间[0, 1)返回一个用均匀分布的随机数填充的张量
        # [0, 1) + 0.5 ----> [0.5, 1.5),shape=[batch_size * self.max_num_seg]
        scales = torch.rand(batch_size * max_num_seg,
                            device=device) + 0.5

        idx_scaled = indices / scales.unsqueeze(-1)
        idx_scaled_fl = torch.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl

        # 首先，将输入信号分成段，其长度为19帧到32帧
        len_seg = torch.randint(low=self.min_len_seg,
                                high=self.max_len_seg,
                                size=(batch_size * max_num_seg, 1),
                                device=device)

        # end point of each segment
        idx_mask = idx_scaled_fl < (len_seg - 1)

        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)
        # offset starts from the 2nd segment
        offset = F.pad(offset[:, :-1], (1, 0), value=0).view(-1, 1)

        idx_scaled_org = idx_scaled_fl + offset

        len_seq_rp = torch.repeat_interleave(len_seq, max_num_seg).to(device)
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)

        idx_mask_final = idx_mask & idx_mask_org

        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)

        index_1 = torch.repeat_interleave(torch.arange(batch_size,
                                                       device=device), counts)

        index_2_fl = idx_scaled_org[idx_mask_final].long()
        index_2_cl = index_2_fl + 1

        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl

        sequences = torch.split(y, counts.tolist(), dim=0)
        # 每个片段使用线性插值进行重采样，重采样因子随机从0.5到1.5
        seq_padded = self.pad_sequences(sequences, seq_length)

        return seq_padded


def get_network(hparams, **kwargs):
    gen = Generator(hparams)

    networks = {'net': gen}
    return networks
