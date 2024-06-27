import torch
import torch.nn as nn


class InstanceNorm2dV3(nn.Module):
    def __init__(self, in_channels:int, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False,
                 device=None, dtype=None):
        super(InstanceNorm2dV3, self).__init__()
        if track_running_stats:
          self.running_mean = nn.Parameter(torch.zeros(in_channels), requires_grad=False)  # shape: (c,)
          self.running_var = nn.Parameter(torch.ones(in_channels), requires_grad=False)  # shape: (c,)
        else:
          self.running_mean = None
          self.running_var = None
        self.in_channels = in_channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype
        if affine:
            # γ and 𝛽 are learnable parameter vectors of size C (where C is the input size)
            self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1), requires_grad=True)  # shape: (1, c, 1, 1)
            self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)  # shape: (1, c, 1, 1)
        else:
          self.gamma = 1.0
          self.beta = 0.0

    def forward(self, input):
        bs, c, h, w = input.shape
        shape = bs, c, h, w
        if self.training:
            curr_mean = input.mean(dim=(2, 3), keepdim=False).mean(0)  # shape: (c,)
            curr_var = input.var(dim=(2, 3), keepdim=False).mean(0)  # shape: (c,)
            if self.track_running_stats:
                self.running_mean = nn.Parameter(curr_mean * self.momentum + self.running_mean * (1.0 - self.momentum))  # shape: (c,)
                self.running_var = nn.Parameter(curr_var * self.momentum + self.running_var * (1.0 - self.momentum))  # shape: (c,)
            output = (input - curr_mean.view(1, -1, 1, 1)) / torch.sqrt(curr_var.view(1, -1, 1, 1) + self.eps)  # shape: (bs, c, h, w)
        else:
            output = (input - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)  # shape: (bs, c, h, w)
        if self.affine:
            output = self.gamma * output + self.beta
        return output
