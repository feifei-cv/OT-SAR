from torch import nn
import torch.nn.functional as F
import torch
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


def kp2gaussian(kp, spatial_size):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)
    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)
    mean_sub = (coordinate_grid - mean)

    var = kp['covar']
    inv_var = matrix_inverse(var)
    shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
    inv_var = inv_var.view(*shape)
    under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))
    under_exp = under_exp.squeeze(-1).squeeze(-1)
    out = torch.exp(-0.5 * under_exp)

    # import matplotlib.pyplot as plt
    # import numpy as np
    # Y = np.arange(0, 64, 1)
    # X = np.arange(0, 64, 1)
    # X, Y = np.meshgrid(X, Y)
    # # fig = plt.figure()
    # # fig, axes = plt.subplots(2, 2)
    # # ax = plt.axes(projection="3d")
    # ax = plt.subplot(121, projection='3d')
    # ax.plot_surface(X, Y, out.squeeze().detach().cpu().numpy()[0,0,...], cmap='gist_rainbow')  # cmap='hot', cmap='gist_rainbow'
    # ax.axis('off')
    #
    # ax1 = plt.subplot(122, projection='3d')
    # ax1.plot_surface(X, Y, out.squeeze().detach().cpu().numpy()[0,1, ...], cmap='gist_rainbow',)  # cmap='hot', cmap='gist_rainbow'
    # ax1.axis('off')
    # plt.tight_layout()
    # # plt.imshow(out.squeeze().detach().cpu().numpy()[0,0,...])
    # plt.show()

    return out


def smallest_singular(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    s1 = a ** 2 + b ** 2 + c ** 2 + d ** 2
    s2 = (a ** 2 + b ** 2 - c ** 2 - d ** 2) ** 2
    s2 = torch.sqrt(s2 + 4 * (a * c + b * d) ** 2)

    norm = torch.sqrt((s1 - s2) / 2)
    return norm


def matrix_inverse(batch_of_matrix, eps=0):
    if eps != 0:
        init_shape = batch_of_matrix.shape
        a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
        b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
        c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
        d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

        det = a * d - b * c
        out = torch.cat([d, -b, -c, a], dim=-1)
        eps = torch.tensor(eps).type(out.type())
        out /= det.max(eps)

        return out.view(init_shape)
    else:
        b_mat = batch_of_matrix
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv =  torch.linalg.solve(eye, b_mat)
        # b_inv, _ = torch.gesv(eye, b_mat)
        return b_inv


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)


    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SAR(nn.Module):
    def __init__(self, feature_dim=256, mid_fim=128, norm_G='spadespectralinstance'):
        super().__init__()
        self.layers = nn.Sequential(
                    SPADEResnetBlock(feature_dim, feature_dim, norm_G, mid_fim),
                    SPADEResnetBlock(feature_dim, feature_dim, norm_G, mid_fim),
                    SPADEResnetBlock(feature_dim, feature_dim, norm_G, mid_fim),
                    SPADEResnetBlock(feature_dim, feature_dim, norm_G, mid_fim))
        self.mapping = nn.Sequential(
            nn.Conv2d(20+3, feature_dim, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(feature_dim, mid_fim, kernel_size=3, padding=1, dilation=1),
        )
        self.fc_warp = nn.Conv2d(feature_dim, mid_fim, kernel_size=1, padding=0)
        self.weight = nn.Sequential(
            nn.Conv2d(feature_dim, 1, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid(),
        )
        self.ratio = 0.001

    def forward(self, f_warp, f_source):
        identity_feat = self.mapping(f_source) ##mapping source feature to identity feature
        x = f_warp
        for layer in self.layers:
            x = layer(x, identity_feat)
        scale_feat = self.fc_warp(f_warp)
        weight = self.weight(torch.cat((identity_feat, scale_feat), dim=1))

        ####
        loss_con = self.ratio*torch.abs(scale_feat - identity_feat).mean()
        x = weight * x + f_warp
        return x, loss_con


class FeatureExtractor(nn.Module):
    """
    Extracting image feature.
    """

    def __init__(self, block_expansion, in_features, out_features, max_features, num_blocks):
        super(FeatureExtractor, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_filters = (out_features if i != num_blocks - 1 else 0) + min(max_features,
                                                                            block_expansion * (2 ** (i + 1)))
            out_filters = out_features
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = out_features

    def forward(self, x):

        down_outs = [x]
        for i in range(len(self.down_blocks)):
            down_outs.append(self.down_blocks[i](down_outs[-1]))
        out = down_outs.pop()

        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
            skip = down_outs.pop()
            if i != len(self.up_blocks) - 1:
                out = torch.cat([out, skip], dim=1)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad1d(padding)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.in1 = nn.InstanceNorm1d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad1d(padding)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.in2 = nn.InstanceNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        out = self.prelu(out)
        return out


class SPADEDecoder(nn.Module):
    def __init__(self, upscale=1, max_features=256, block_expansion=64, out_channels=64, num_down_blocks=2):
        for i in range(num_down_blocks):
            input_channels = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels  # 256

        self.fc = nn.Conv2d(input_channels, input_channels, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(input_channels, input_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(input_channels, input_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(input_channels, input_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(input_channels, input_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(input_channels, input_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(input_channels, input_channels, norm_G, label_num_channels)
        self.up_0 = SPADEResnetBlock(input_channels, input_channels//2, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input_channels//2, out_channels, norm_G, label_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2)
            )

    def forward(self, feature):
        seg = feature  # Bx256x64x64
        x = self.fc(feature)  # Bx512x64x64
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        x = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256

        x = self.conv_img(F.leaky_relu(x, 2e-1))  # Bx64x256x256 -> Bx3xHxW
        x = torch.sigmoid(x)  # Bx3xHxW

        return x


class SPADEDecoder1(nn.Module):
    def __init__(self, upscale=1, max_features=256, block_expansion=64, out_channels=64, num_down_blocks=2):
        for i in range(num_down_blocks):
            input_channels = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels  # 256

        self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input_channels, out_channels, norm_G, label_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2)
            )

    def forward(self, feature):
        seg = feature  # Bx256x64x64
        x = self.fc(feature)  # Bx512x64x64
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        x = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256

        x = self.conv_img(F.leaky_relu(x, 2e-1))  # Bx64x256x256 -> Bx3xHxW
        x = torch.sigmoid(x)  # Bx3xHxW

        return x
