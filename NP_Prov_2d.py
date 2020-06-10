import random

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from utils import channel_last
import warnings


warnings.filterwarnings("ignore")

class Conv2dResBlock(nn.Module):
    def __init__(self, in_channel, out_channel=128, kernel_size = 5):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, 1, (kernel_size-1)//2, groups=in_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size, 1, (kernel_size-1)//2, groups=in_channel),
            nn.ReLU()
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output

class NPPROV2d(nn.Module):
    def __init__(self, channel=1, kernel_size = 9):
        super().__init__()

        self.conv_theta = nn.Conv2d(channel, 128, kernel_size, 1, (kernel_size-1)//2)
        self.encoder = nn.Conv2d(channel, 128, kernel_size, 1, (kernel_size -1)//2)
        self.decoder = nn.Conv2d(128, channel, kernel_size, 1, (kernel_size - 1) // 2)
        self.conv_theta_var = nn.Conv2d(channel, 128, kernel_size, 1, (kernel_size -1)//2)
        res_kernel_size = 3 if kernel_size!=9 else 5
        self.cnn = nn.Sequential(
            nn.Conv2d(128 + 128 +128, 128, 1, 1, 0),
            Conv2dResBlock(128, 128, res_kernel_size),
            Conv2dResBlock(128, 128, res_kernel_size),
            Conv2dResBlock(128, 128, res_kernel_size),
            Conv2dResBlock(128, 128, res_kernel_size)
        )
        self.mu_layer = nn.Conv2d(128, channel, 1, 1, 0)
        self.sigma_layer = nn.Conv2d(128*2, channel, 1, 1, 0)
        self.pos = nn.Softplus()
        self.mse_loss = nn.MSELoss()
        self.channel = channel

        self.mr = [0.5, 0.7, 0.9]

    def forward(self, I, context_mask):
        M_c = context_mask.unsqueeze(1).repeat(1, self.channel, 1, 1)
        signal = I * M_c
        density = M_c

        # self correlation
        h_self = self.encoder(density)
        density_recover = self.decoder(h_self)
        mse_loss = self.mse_loss(density_recover, density)

        # cross correlation
        # self.conv_theta.abs_constraint()
        density_prime = self.conv_theta(density)
        signal_prime = self.conv_theta(signal)
        # signal_prime = signal_prime.div(density_prime + 1e-8)
        # # self.conv_theta.abs_unconstraint()
        # print(signal_prime.size(), density_prime.size())
        h_cross = torch.cat([signal_prime, density_prime], 1)
        # print(h_self.size())
        h = torch.cat([h_self, h_cross], 1)
        f = self.cnn(h)
        mean = self.mu_layer(f)

        h_cross_var = torch.cat([density_prime, density_prime], 1)
        h_var = torch.cat([h_self, h_cross_var], 1)
        f_var = self.cnn(h_var)
        M_t = M_c.new_ones(M_c.size())
        f_t = self.conv_theta_var(M_t)
        # print(f.size(), f_t.size())
        f = torch.cat([f_var, f_t], dim=1)
        pre_std = self.sigma_layer(f)
        std = self.pos(pre_std)

        mean, std = channel_last(mean), channel_last(std)
        return MultivariateNormal(mean, scale_tril=std.diag_embed()), mse_loss, std

    def complete(self, I, M_c=None, missing_rate=None):
        if M_c is None:
            if missing_rate is None:
                missing_rate = random.choice(self.mr)
            M_c = I.new_empty(I.size(0), 1, I.size(2), I.size(3)).bernoulli_(p=1 - missing_rate).repeat(1, self.channel, 1, 1)

        signal = I * M_c
        density = M_c
        # self correlation
        h_c = self.conv_theta1(density)

        # cross correlation
        # self.conv_theta.abs_constraint()
        density_prime = self.conv_theta(density) * h_c
        signal_prime = self.conv_theta(signal) * h_c
        # signal_prime = signal_prime.div(density_prime + 1e-8)
        # # self.conv_theta.abs_unconstraint()
        h = torch.cat([signal_prime, density_prime], 1)
        f = self.cnn(h)
        mean = self.mu_layer(f)

        I_ones = I.new_ones(I.size())
        signal = I_ones * M_c
        signal_prime = self.conv_theta(signal) * h_c
        h = torch.cat([signal_prime, density_prime], 1)
        f = self.cnn(h)
        M_t = M_c.new_ones(M_c.size())
        f_t = self.conv_theta2(M_t)
        # print(f.size(), f_t.size())
        f = torch.cat([f, f_t], dim=1)
        pre_std = self.sigma_layer(f)
        std = self.pos(pre_std)

        return M_c, mean, std, MultivariateNormal(channel_last(mean), scale_tril=channel_last(std).diag_embed())


