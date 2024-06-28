import torch
import torch.nn as nn


def replace_in_by_bn(model:nn.Module):
    i = 0
    for m in model:
        if instance(m, nn.InstanceNorm2d):
            bn = nn.BatchNorm()
            bn.running_mean = m.running_mean
            bn.running_var = m.running_var
            model[i] = bn
        i += 1
