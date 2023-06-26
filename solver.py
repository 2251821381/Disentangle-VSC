from model import Generator as Generator
# from model import InterpLnr

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle
from hparams import hparams
import torch.nn as nn

from utils import quantize_f0_torch, quantize_f0_numpy

# use demo data for simplicity
# make your own validation set as needed
# validation_pt = pickle.load(open('assets/demo.pkl', "rb"))
validation_pt = pickle.load(open(hparams.validate_dir, "rb"))


class Solver(object):
    """Solver for training"""

    def __init__(self, vcc_loader, config, hparams):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.hparams = hparams

        # Training configurations.
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.test_iters = config.test_iters
        self.lambda_kl = hparams.lambda_kl

        # Build the models and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        self.G = Generator(self.hparams)

        # self.Interp = InterpLnr(self.hparams)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')

        self.G.to(self.device)
        # self.Interp.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
        self.G.load_state_dict(g_checkpoint['models'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def loss_fn(self, original_seq, recon_seq, recon_seq_psnt, c_mean, c_logvar):
        """
        Loss function consists of 3 parts, the reconstruction term that is the MSE loss between the generated and the original images
        the KL divergence of f, and the sum over the KL divergence of each z_t, with the sum divided by batch_size

        Loss = {mse + KL of f + sum(KL of z_t)} / batch_size
        Prior of f is a spherical zero mean unit variance Gaussian and the prior of each z_t is a Gaussian whose mean and variance
        are given by the LSTM
        """
        batch_size, _, _ = original_seq.shape
        g_loss_id = F.mse_loss(original_seq, recon_seq)
        g_loss_id_psnt = F.mse_loss(original_seq, recon_seq_psnt)
        kld_c = -0.5 * torch.mean(1 + c_logvar - torch.pow(c_mean, 2) - torch.exp(c_logvar))
        return g_loss_id / batch_size, \
               g_loss_id_psnt / batch_size, \
               kld_c / batch_size

    # =====================================================================================================================

    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.g_optimizer, 'G_optimizer')
            # self.print_optimizer(self.p_optimizer, 'P_optimizer')

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        print('Current learning rates, g_lr: {}.'.format(g_lr))

        # Print logs in specified order
        keys = ['G/g_loss_id', 'G/g_loss_id_psnt', 'G/kld_c']
        # keys = ['G/g_loss']

        # Start training.
        print('device:' + str(self.device))
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real_org, emb_org, accent_org, f0_org, len_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_org, emb_org, accent_org, f0_org, len_org = next(data_iter)

            x_real_org = x_real_org.to(self.device)  # (8, 128, 80)
            emb_org = emb_org.to(self.device)  # (8, 256)
            accent_org = accent_org.to(self.device)  # (8, 256)
            len_org = len_org.to(self.device)  # (8, )
            f0_org = f0_org.to(self.device)  # (8, 128, 1)

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            self.G = self.G.train()

            # Identity mapping loss
            f0_org_intrp = quantize_f0_torch(f0_org[:, :, -1])[0]  # (16, 192, 257)------> (8, 128, 257)

            # x_identic = self.G(x_f0_intrp_org, x_real_org, emb_org)
            # c_mean, c_logvar, content, recon_mel = self.G(x_real_org, x_real_org, f0_org_intrp, emb_org, x_real_org)
            c_mean, c_logvar, content, recon_mel, recon_mel_psnt = \
                self.G(x_real_org, x_real_org, f0_org_intrp, f0_org_intrp, emb_org)
            g_loss_id, g_loss_id_psnt, kld_c = self.loss_fn(x_real_org, recon_mel, recon_mel_psnt, c_mean, c_logvar)

            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_kl * kld_c
            # Backward and optimize.
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {'G/g_loss_id': g_loss.item(),
                    'G/g_loss_id_psnt': g_loss_id_psnt.item(),
                    'G/kld_c': kld_c.item()}
            # loss = {'G/g_loss': g_loss.item()}

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, i + 1)

            # Save models checkpoints.
            if (i + 1) % self.model_save_step == 0:
                k = i + 1
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(k))
                torch.save({'models': self.G.state_dict(),
                            'optimizer': self.g_optimizer.state_dict()}, G_path)
                print('Saved models G_checkpoints into {}...'.format(self.model_save_dir))

            # Validation.
            if (i + 1) % self.sample_step == 0:
                self.G = self.G.eval()
                with torch.no_grad():
                    loss_val = []
                    # validation_pt: [speaker, emb_org_val, accent_org_val, accent,
                    #                         (mel_org, f0_org, len_org, uttr_id)]
                    for val_sub in validation_pt:
                        # speaker embedding
                        emb_org_val = torch.from_numpy(val_sub[1]).to(self.device)
                        # accent
                        accent_org_val = torch.from_numpy(val_sub[2]).to(self.device)
                        for k in range(4, 5):
                            # mel
                            x_real_pad = val_sub[k][0][np.newaxis, :, :]

                            # length
                            len_org_val = torch.tensor([val_sub[k][2]]).to(self.device)
                            # f0
                            f0_org = val_sub[k][1]
                            # f0_org = quantize_f0_torch(f0_org[:, :, -1])[0]

                            # f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device)
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device)

                            # 验证，并计算损失值
                            # x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                            # _, _, _, x_identic_val = self.G(x_real_pad, x_real_pad, f0_org_val, emb_org_val,
                            #                                 x_real_pad)
                            _, _, _, recon_mel, x_identic_psnt = \
                                self.G(x_real_pad, x_real_pad, f0_org_val, f0_org_val, emb_org_val)

                            g_loss_val = F.mse_loss(x_real_pad, x_identic_psnt, reduction='sum') + \
                                         F.mse_loss(x_real_pad, recon_mel, reduction='sum')
                            loss_val.append(g_loss_val.item())
                val_loss = np.mean(loss_val)
                print('Validation loss: {}'.format(val_loss))
                if self.use_tensorboard:
                    self.writer.add_scalar('Validation_loss', val_loss, i + 1)

            # plot test samples
            if (i + 1) % self.sample_step == 0:
                self.G = self.G.eval()
                with torch.no_grad():
                    # validation_pt: [speaker, emb_org_val, accent_org_val, accent,
                    #                         (mel_org, f0_org, len_org, uttr_id)]
                    for val_sub in validation_pt:
                        # speaker embedding
                        emb_org_val = torch.from_numpy(val_sub[1]).to(self.device)
                        # accent
                        accent_org_val = torch.from_numpy(val_sub[2]).to(self.device)
                        for k in range(4, 5):
                            # mel
                            x_real_pad = val_sub[k][0][np.newaxis, :, :]
                            # length
                            len_org_val = torch.tensor([val_sub[k][2]]).to(self.device)
                            # f0
                            f0_org = val_sub[k][1]
                            # f0_org = quantize_f0_torch(f0_org[:, :, -1])[0]

                            # f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device)
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device)
                            f0_trg = torch.zeros_like(f0_org_val)

                            # 合成的mel
                            _, _, _, _, x_identic_val = \
                                self.G(x_real_pad, x_real_pad, f0_org_val, f0_org_val, emb_org_val)
                            # 合成的mel且Pitch为空
                            _, _, _, _, x_identic_woF = \
                                self.G(x_real_pad, x_real_pad, torch.zeros_like(f0_org_val), f0_org_val, emb_org_val)
                            # 合成的mel且Rhythm为空
                            _, _, _, _, x_identic_woR = \
                                self.G(x_real_pad, torch.zeros_like(x_real_pad), f0_org_val, f0_org_val, emb_org_val)
                            # 合成的mel且Content为空
                            _, _, _, _, x_identic_woC = \
                                self.G(torch.zeros_like(x_real_pad), x_real_pad, f0_org_val, f0_org_val, emb_org_val)
                            # Remove Timbre
                            _, _, _, _, x_identic_woU = \
                                self.G(x_real_pad, x_real_pad, f0_org_val, f0_org_val, torch.zeros_like(emb_org_val))
                            # Remove Accent
                            _, _, _, _, x_identic_woA = \
                                self.G(x_real_pad, x_real_pad, f0_org_val, torch.zeros_like(f0_org_val), emb_org_val)

                            melsp_gd_pad = x_real_pad[0].cpu().numpy().T
                            melsp_out = x_identic_val[0].cpu().numpy().T
                            melsp_woF = x_identic_woF[0].cpu().numpy().T
                            melsp_woR = x_identic_woR[0].cpu().numpy().T
                            melsp_woC = x_identic_woC[0].cpu().numpy().T
                            melsp_woU = x_identic_woU[0].cpu().numpy().T
                            melsp_woA = x_identic_woA[0].cpu().numpy().T

                            min_value = np.min(np.hstack(
                                [melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC, melsp_woU, melsp_woA]))
                            max_value = np.max(np.hstack(
                                [melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC, melsp_woU, melsp_woA]))

                            # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, sharex=True,
                            #                                                                      sharey=True)
                            fig, ((ax1, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(3, 2, sharex=True,
                                                                                                 sharey=True)
                            ax1.imshow(melsp_gd_pad, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
                            ax1.set_title('source')
                            # ax2.imshow(melsp_out, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
                            # ax2.set_title('output')
                            ax3.imshow(melsp_woC, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
                            ax3.set_title('Remove Content')
                            ax4.imshow(melsp_woR, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
                            ax4.set_title('Remove Rhythm')
                            ax5.imshow(melsp_woF, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
                            ax5.set_title('Remove Pitch')
                            ax6.imshow(melsp_woU, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
                            ax6.set_title('Remove Timbre')
                            ax7.imshow(melsp_woA, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
                            ax7.set_title('Remove Accent')
                            # fig.colorbar(ax1)
                            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
                            plt.savefig(f'{self.sample_dir}/{i + 1}_{val_sub[0]}_{k}_{self.test_iters}.png', dpi=600)
                            plt.close(fig)
