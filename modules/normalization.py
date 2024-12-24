import torch
import torch.nn as nn
from modules.util import calc_mean_std

class ElaIN(nn.Module):
    def __init__(self, norm_nc, addition_nc): ## 128, 256
        super().__init__()
        
        self.mlp_same = nn.Conv1d(addition_nc, norm_nc, 1)
        self.mlp_gamma = nn.Conv1d(norm_nc, norm_nc, 1)
        self.mlp_beta = nn.Conv1d(norm_nc, norm_nc, 1)

        self.mlp_weight = nn.Conv1d(2*norm_nc, norm_nc, 1)

    def forward(self, x, addition): #1*128*4096, 1*256*4096

        # feature dim align
        addition = self.mlp_same(addition)  #1*256*4096

        # get gamma and beta
        addition_gamma = self.mlp_gamma(addition)
        addition_beta = self.mlp_beta(addition)

        # calculate the mean of identity features and warped features in dim=2
        id_avg = torch.mean(addition, 2 ,keepdim=True) #1*128*1
        x_avg = torch.mean(x, 2, keepdim=True)
        
        # get the adaptive weight
        weight_cat = torch.cat((id_avg, x_avg), 1)
        weight = self.mlp_weight(weight_cat)
        
        # calculate the final modulation parameters
        x_mean, x_std = calc_mean_std(x) ## warp 特征
        gamma = addition_gamma * weight + x_std * (1-weight)
        beta = addition_beta * weight + x_mean * (1-weight)
            
        # normalization and denormalization    
        x = (x - x_mean) / x_std
        out = x * (1 + gamma) + beta

        return out