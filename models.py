#============================================================
#
#  Main script for training and testing. 
#  If you use this motion model, please our work: 
#  "Predictive online 3D target tracking with population-based 
#  generative networks for image-guided radiotherapy" 
#  Accepted at IPCAI 2021
#  
#  (The implementation of VoxelMorph and the code for the 
#  associated losses was taken from: 
#  https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/)
#
#  github id: lisetvr
#  MedICAL Lab
#============================================================

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions import Normal, Independent
from pytorch_prototyping import *

from convlstm import *
from convgru import *
import utiles

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------
# --- Motion estimation ----------------------
class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding
                            layers
        """
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)

        return y

class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out

class Voxelmorph(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for
    the 2018 implementation of voxelmorph.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True):
        """
        Instantiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(Voxelmorph, self).__init__()

        dim = len(vol_size)
        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

    def forward(self, v_ref, v_curr):
        """
        Pass input x through forward once
            :param v_ref: moving image that we want to shift
            :param v_curr: fixed image that we want to shift to
        """
        x = torch.cat([v_ref, v_curr], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        return flow

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, mode=self.mode)
# --------------------------------------------

# --- Motion modeling ------------------------

class Encoder(nn.Module):
    def __init__(self, nb_convs, in_channels, out_channels, custom_stride=[(2,2,2), (2,2,2), (2,2,2), (1,1,1)],
                 norm=nn.BatchNorm3d):

        super().__init__()
        assert nb_convs == len(out_channels)
        self.encoder = list()
        kernel_pad_tuples = [(3, 1), (3, 1), (3, 1), (3, 1)]
        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]
            # ---------------------------
            self.encoder += [nn.Conv3d(in_ch, out_channels[i], kernel_size=kernel_pad_tuples[i][0],
                                       padding=kernel_pad_tuples[i][1], stride=custom_stride[i])]
            if norm is not None:
                self.encoder += [norm(out_channels[i])]
            self.encoder += [nn.ReLU(True)]

        self.linear = nn.Linear(out_channels[-1] * 4 * 8 * 8, 4 * 8 * 8)
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, Vref):
        x = self.encoder(Vref).view(Vref.shape[0], -1)
        return self.linear(x)


def conv_bn_Lrelu(in_planes, out_planes, ksp_conv=(3,2,1), ksp_Tconv=(2,2,0)):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=ksp_conv[0], stride=ksp_conv[1], padding=ksp_conv[2]),
                         nn.BatchNorm3d(out_planes), nn.LeakyReLU(0.4),
                         nn.ConvTranspose3d(out_planes, out_planes, kernel_size=ksp_Tconv[0], stride=ksp_Tconv[1],
                                            padding=ksp_Tconv[2]))


class Decoder(nn.Module):
    def __init__(self):

        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(1, 64, 1, 1, 0), nn.BatchNorm3d(64), nn.LeakyReLU(0.2))

        self.sc1 = nn.Sequential(nn.Conv3d(32, 1, 1, 1, 0), nn.InstanceNorm3d(1), nn.LeakyReLU(0.2))
        self.sc2 = nn.Sequential(nn.Conv3d(48, 1, 1, 1, 0), nn.InstanceNorm3d(1), nn.LeakyReLU(0.2))
        self.sc3 = nn.Sequential(nn.Conv3d(96, 1, 1, 1, 0), nn.InstanceNorm3d(1), nn.LeakyReLU(0.2))

        self.c2 = conv_bn_Lrelu(64+1, 32, ksp_conv=(3, 1, 1), ksp_Tconv=(2, 2, 0))
        self.c1 = conv_bn_Lrelu(32+1, 16, ksp_conv=(3, 1, 1), ksp_Tconv=(2, 2, 0))
        self.c0 = conv_bn_Lrelu(16+1, 8, ksp_conv=(3, 1, 1), ksp_Tconv=(2, 2, 0))

        # Get the flow field
        self.flow = nn.Conv3d(8, 3, kernel_size=1, padding=0, stride=1)

    def forward(self, z, f_low, f_med, f_high):

        sc_low = self.sc3(f_low)
        sc_med = self.sc2(f_med)
        sc_high = self.sc1(f_high)

        z3 = self.conv1(z)
        z3 = torch.cat([z3, sc_low], dim=1)
        z2 = self.c2(z3)
        z2 = torch.cat([z2, sc_med], dim=1)
        z1 = self.c1(z2)
        z1 = torch.cat([z1, sc_high], dim=1)
        z0 = self.c0(z1)

        x = self.flow(z0)
        return x


class Temporal_pred_net(nn.Module):

    def __init__(self, nb_convs, in_channels, out_channels, sag_lin_input_dim=128*8*8, cor_lin_input_dim=128*8*8,
                 output_dim=256, norm=nn.BatchNorm2d, dropout=False, rnn='gru'):

        super().__init__()
        assert nb_convs == len(out_channels)

        self.encoder_sag = list()
        self.encoder_cor = list()
        self.dropout = dropout
        self.sag_lin_dim = sag_lin_input_dim
        self.cor_lin_dim = cor_lin_input_dim
        sag_custom_stride = [(2, 2), (2, 2), (2, 2), (1, 1)]
        cor_custom_stride = [(1, 2), (2, 2), (2, 2), (1, 1)]
        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]

            self.encoder_sag += [nn.Conv2d(in_ch, out_channels[i], kernel_size=3, padding=1, stride=sag_custom_stride[i])]
            self.encoder_cor += [nn.Conv2d(in_ch, out_channels[i], kernel_size=3, padding=1, stride=cor_custom_stride[i])]

            if norm is not None:
                self.encoder_sag += [norm(out_channels[i], affine=True)]
                self.encoder_cor += [norm(out_channels[i], affine=True)]
            self.encoder_sag += [nn.ReLU(True)]
            self.encoder_cor += [nn.ReLU(True)]

            self.encoder_sag += [nn.Conv2d(out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1)]
            self.encoder_cor += [nn.Conv2d(out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1)]

            if norm is not None:
                self.encoder_sag += [norm(out_channels[i], affine=True)]
                self.encoder_cor += [norm(out_channels[i], affine=True)]
            self.encoder_sag += [nn.ReLU(True)]
            self.encoder_cor += [nn.ReLU(True)]
            if self.dropout:
                self.encoder_sag += [nn.Dropout2d()] ####################
                self.encoder_cor += [nn.Dropout2d()] ####################
        self.encoder_sag = nn.Sequential(*self.encoder_sag)
        self.encoder_cor = nn.Sequential(*self.encoder_cor)

        if rnn == 'gru':
            self.sag_rnn_enc = ConvGRU(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
            self.cor_rnn_enc = ConvGRU(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
        else:
            self.sag_rnn_enc = ConvLSTM(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
            self.cor_rnn_enc = ConvLSTM(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.bn_sag = norm(out_channels[-1])
        self.bn_cor = norm(out_channels[-1])
        self.sag_enc = nn.Linear(self.sag_lin_dim, output_dim)
        self.cor_enc = nn.Linear(self.cor_lin_dim, output_dim)

    def forward(self, sag, cor):

        if sag is not None and cor is None:
            encsag1 = self.encoder_sag(sag[:, :, 0, :, :])
            encsag2 = self.encoder_sag(sag[:, :, 1, :, :])
            encsag3 = self.encoder_sag(sag[:, :, 2, :, :])
            encsag = torch.stack([encsag1, encsag2, encsag3], dim=2)
            encsag = encsag.permute(0, 2, 1, 3, 4)
            h_c = self.sag_rnn_enc(encsag)[1][0]
            encsag = self.bn_sag(h_c)
            encsag = encsag.view(encsag.shape[0], self.sag_lin_dim)
            encsag = self.sag_enc(encsag)
            if self.dropout:
                encsag = nn.Dropout()(encsag)
            return encsag

        elif cor is not None and sag is None:
            enccor1 = self.encoder_cor(cor[:, :, 0, :, :])
            enccor2 = self.encoder_cor(cor[:, :, 1, :, :])
            enccor3 = self.encoder_cor(cor[:, :, 2, :, :])
            enccor = torch.stack([enccor1, enccor2, enccor3], dim=2)
            enccor = enccor.permute(0, 2, 1, 3, 4)
            h_c = self.cor_rnn_enc(enccor)[1][0]
            enccor = self.bn_cor(h_c)
            enccor = enccor.view(enccor.shape[0], self.cor_lin_dim)
            enccor = self.cor_enc(enccor)
            if self.dropout:
                enccor = nn.Dropout()(enccor)
            return enccor

        else:
            raise NotImplementedError('At least one conditioning plane is required')


class RefNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, out_channels[0], 3, 2, 1), nn.BatchNorm3d(out_channels[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels[0], out_channels[1], 3, 2, 1), nn.BatchNorm3d(out_channels[1]), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(out_channels[1], out_channels[2], 3, 2, 1), nn.BatchNorm3d(out_channels[2]), nn.ReLU())

    def forward(self, Vref):
        high_res = self.conv1(Vref)
        med_res = self.conv2(high_res)
        low_res = self.conv3(med_res)
        return high_res, med_res, low_res


class AE(nn.Module):

    # Motion modeling via autoencoding

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.vol_enc = Encoder(nb_convs=3, in_channels=self.in_channels, out_channels=[64, 128, 64], norm=nn.BatchNorm3d)
        self.vref_encoder = RefNet(in_channels=1, out_channels=[32, 48, 96])
        self.vol_dec = Decoder()
        self.stn = SpatialTransformer([32, 64, 64])

    def forward(self, Vref, Vn, dvf):

        if self.in_channels == 1:
            enc_latent = self.vol_enc(Vn)    # Voxel intensity
        else:
            enc_latent = self.vol_enc(dvf)   # Motion data

        f_high, f_med, f_low = self.vref_encoder(Vref)  # Compressed skip connections from the reference volume
        reshaped_bottleneck = enc_latent.view(enc_latent.shape[0], 1, 4, 8, 8)
        generated_dvf = self.vol_dec(reshaped_bottleneck, f_low, f_med, f_high)

        return self.stn(Vref, dvf), generated_dvf, self.stn(Vref, generated_dvf), enc_latent


class TLNet_dvf_3S(nn.Module):

    def __init__(self, in_channels, train_step):
        super().__init__()
        self.in_channels = in_channels
        self.train_step = train_step
        self.AE = AE(in_channels)
        self.condi_encoder = Temporal_pred_net(nb_convs=4, in_channels=2, out_channels=[64, 128, 256, 128],
                                               sag_lin_input_dim=128 * 8 * 8, cor_lin_input_dim=128 * 8 * 8,
                                               output_dim=256, norm=nn.BatchNorm2d)
        self.fc_c1c2 = nn.Linear(128 * 2 * 8 * 8, 256)
        self.fc_c = nn.Linear(128 * 8 * 8, 256)
        self.stn = SpatialTransformer([32, 64, 64])
        self.euc_loss = nn.MSELoss()

    def forward(self, Vn, Vref, c1, c2, dvf):

        if self.in_channels == 1:
            X_in = Vn   # In this case, the input channel is the voxel intensity of the current volume (Vn)
        else:
            X_in = dvf  # The input channel is a DVF

        if Vn is not None:  # Training mode

            with torch.set_grad_enabled(self.train_step != '2'):  # True if the training step is not 2
                enc_latent = self.AE.vol_enc(X_in)

            if c1 is not None and c2 is None:
                features_sag = self.condi_encoder(sag=c1, cor=None)
                l2_loss = self.euc_loss(features_sag, enc_latent)
            else:
                features_cor = self.condi_encoder(sag=None, cor=c2)
                l2_loss = self.euc_loss(features_cor, enc_latent)

            with torch.set_grad_enabled(self.train_step != '2'):  # True if the training step is not 2
                f_high, f_med, f_low = self.AE.vref_encoder(Vref)
                reshaped_bottleneck = enc_latent.view(enc_latent.shape[0], 1, 4, 8, 8)
                generated_dvf = self.AE.vol_dec(reshaped_bottleneck, f_low, f_med, f_high)

            return generated_dvf, self.stn(Vref, generated_dvf), l2_loss

        else:  # Inference mode

            if c1 is not None and c2 is None:
                features_sag = self.condi_encoder(sag=c1, cor=None)
                reshaped_bottleneck = features_sag.view(features_sag.shape[0], 1, 4, 8, 8)
                f_high, f_med, f_low = self.AE.vref_encoder(Vref)
                generated_dvf = self.AE.vol_dec(reshaped_bottleneck, f_low, f_med, f_high)
            else:
                features_cor = self.condi_encoder(sag=None, cor=c2)
                reshaped_bottleneck = features_cor.view(features_cor.shape[0], 1, 4, 8, 8)
                f_high, f_med, f_low = self.AE.vref_encoder(Vref)
                generated_dvf = self.AE.vol_dec(reshaped_bottleneck, f_low, f_med, f_high)

            return generated_dvf, self.stn(Vref, generated_dvf)


if __name__ == "__main__":

    # --- Inference with random tensors ---
    device = torch.device("cuda:0")
    vref = torch.randn([2, 1, 32, 64, 64]).to(device)
    vt = torch.randn([2, 1, 32, 64, 64]).to(device)
    sag_images = torch.randn([2, 2, 3, 64, 64]).to(device)
    cor_images = torch.randn([2, 2, 32, 64]).to(device)
    dvf = torch.randn([2, 3, 32, 64, 64]).to(device)
    model = TLNet_dvf_3S(3, train_step=2).to(device)
    output = model(vt, vref, sag_images, None, dvf)
