import torch
from torch import nn
import torch.nn.functional as F
import sys
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, AntiAliasInterpolation2d
from modules.util import FeatureExtractor, SAR ##SPADEDecoder


class OptimGenerator(nn.Module):

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks, num_bottleneck_blocks):
        super(OptimGenerator, self).__init__()

        self.temperature = 0.05

        ##### 1、Feature Adapter Module
        self.down = AntiAliasInterpolation2d(num_channels, scale=0.25)
        self.predictor_source = FeatureExtractor(block_expansion=block_expansion, in_features=num_channels + num_kp,
                                                 out_features=256, max_features=max_features, num_blocks=5)
        self.predictor_driving = FeatureExtractor(block_expansion=block_expansion, in_features=num_kp, out_features=256,
                                                  max_features=max_features,  num_blocks=5)

        #### 2、Encoder: same as FOMM
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        ###3、Decoder: same as FOMM
        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        ###4、prediction
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

        ####5、Better decoder
        # self.decoder = SPADEDecoder()

        ####6，SAR module
        self.sar = SAR()


    def forward(self, source_image, driving_image, kp_driving, kp_source):

        ### 1、source image encode
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        output_dict = {}

        ### 2、Extract feature
        source_inputs = torch.concat([self.down(source_image), kp_source['gauss_heatmap_point']], dim=1)
        driving_inputs = kp_driving['gauss_heatmap_point']

        ### source
        source_feats = self.predictor_source(source_inputs)
        # source_feat = F.unfold(source_feats, kernel_size=3, padding=1)  ###
        # source_feat = source_feat - source_feat.mean(dim=1, keepdim=True) ###
        source_feats = source_feats - source_feats.mean(dim=1, keepdim=True)
        source_feat = source_feats.view(source_feats.shape[0], source_feats.shape[1], -1)
        source_feat_norm = torch.norm(source_feat, 2, 1, keepdim=True) + sys.float_info.epsilon
        source_feat = torch.div(source_feat, source_feat_norm)

        #### driving
        driving_feats = self.predictor_driving(driving_inputs)
        # driving_feat = F.unfold(driving_feats, kernel_size=3, padding=1)
        # driving_feat = driving_feat - driving_feat.mean(dim=1, keepdim=True)
        driving_feats = driving_feats - driving_feats.mean(dim=1, keepdim=True)
        driving_feat = driving_feats.view(driving_feats.shape[0], driving_feats.shape[1], -1)
        driving_feat_norm = torch.norm(driving_feat, 2, 1, keepdim=True) + sys.float_info.epsilon
        driving_feat = torch.div(driving_feat, driving_feat_norm)

        #### 3、Optimal Transport: Correlation matrix C (cosine similarity),Then Sinkhorn algorithm iter
        C_Matrix = torch.matmul(driving_feat.permute(0, 2, 1), source_feat)
        K = torch.exp(-(1.0 - C_Matrix) / self.temperature)

        #### Sinkhorn algorithm iter, adap from github.com.
        power = 1
        a = (torch.ones((K.shape[0], K.shape[1], 1), device=driving_feat.device, dtype=driving_feat.dtype)/K.shape[1])
        prob1 = (torch.ones((K.shape[0], K.shape[1], 1), device=driving_feat.device, dtype=driving_feat.dtype)/K.shape[1])
        prob2 = (torch.ones((K.shape[0], K.shape[2], 1), device=source_feat.device, dtype=source_feat.dtype)/K.shape[2])
        for _ in range(5): ### 10
            # Update b
            KTa = torch.bmm(K.transpose(1, 2), a)
            b = torch.pow(prob2 / (KTa + 1e-8), power)
            # Update a
            Kb = torch.bmm(K, b)
            a = torch.pow(prob1 / (Kb + 1e-8), power)
        ## Optimal matching matrix
        T_m = torch.mul(torch.mul(a, K), b.transpose(1, 2))
        T_m_norm = T_m / torch.sum(T_m, dim=2, keepdim=True)

        #### 4、Implicit warping
        final_shape = out.shape
        source_out_warp = torch.matmul(T_m_norm, out.view(final_shape[0], final_shape[1], -1).permute(0, 2, 1))
        source_out_warp = source_out_warp.permute(0, 2, 1)
        out_source_warp = source_out_warp.view(final_shape)

        ### 5、Refinement f_warp
        out_source_warp, loss_con = self.sar(out_source_warp, source_inputs)

        #### 6、Decoder
        outs = self.bottleneck(out_source_warp)
        for i in range(len(self.up_blocks)):
            outs = self.up_blocks[i](outs)
        outs = self.final(outs)
        outs = F.sigmoid(outs)

        # outs = self.decoder(out_source_warp)
        output_dict["prediction"] = outs
        output_dict["loss_con"] = loss_con.view(1,)
        return output_dict
