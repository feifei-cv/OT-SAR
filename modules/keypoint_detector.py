from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d,DownBlock2d


class KPDetector(nn.Module):
    """
    Detecting a keypoints.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks,
                 temperature, scale_factor=1, pad=3):

        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)
        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp,
                            kernel_size=(7, 7), padding=pad)

        # self.layers = nn.Sequential(
        #     nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=32, kernel_size=(7, 7), padding=pad),
        #     DownBlock2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
        #     DownBlock2d(64, 96, kernel_size=(3, 3), padding=(1, 1)),
        #     DownBlock2d(96, 128, kernel_size=(3, 3), padding=(1, 1)),
        #     nn.Conv2d(in_channels=128, out_channels=num_kp, kernel_size=(7, 7), padding=pad),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Sigmoid(),
        # )


        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        #covar
        mean_sub = grid - value.unsqueeze(-2).unsqueeze(-2)
        covar = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        covar = covar * heatmap.unsqueeze(-1)
        covar = covar.sum(dim=(2, 3))
        kp['covar'] = covar
        # kp['covar'] = 0.01
        return kp

    def kp2gaussian(self, kp, spatial_size):
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

        # #### case1
        var = kp['covar']
        # var = 0.01
        if type(var) == float:
            out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / var)
        else:
            shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
            inv_var = torch.inverse(var).view(*shape)
            under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))
            under_exp = under_exp.squeeze(-1).squeeze(-1)
            out = torch.exp(-0.5 * under_exp)
        return out

    def forward(self, x):

        if self.scale_factor != 1:
            x = self.down(x)

        #### feature map and heatmap
        feature_map = self.predictor(x)
        ###
        # confidence = self.layers(feature_map)
        ####
        prediction = self.kp(feature_map)
        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        ### mean and var
        out = self.gaussian2kp(heatmap)

        ### gauss-like representation
        gauss_heatmap_point = self.kp2gaussian(out, final_shape[2:])
        out['gauss_heatmap_point'] = gauss_heatmap_point
        # out['confidence'] = confidence

        return out



