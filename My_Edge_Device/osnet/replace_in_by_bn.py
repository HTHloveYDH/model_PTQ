import torch
import torch.nn as nn


def replace_in_by_bn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.InstanceNorm2d):
            bn = nn.BatchNorm2d(mm.num_features)
            bn.running_mean = mm.running_mean
            bn.running_var = mm.running_var
            setattr(model, name, bn)
        elif len(list(module.children())) > 0:
            replace_in_by_bn(module)
