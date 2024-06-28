import torch
import torch.nn as nn


def replace_in_by_bn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.InstanceNorm2d):
            bn = nn.BatchNorm2d(
                module.num_features, eps=module.eps, momentum=module.momentum, 
                affine=module.affine, track_running_stats=module.track_running_stats
            )
            bn.running_mean = module.running_mean
            bn.running_var = module.running_var
            bn.weight = module.weight
            bn.bias = module.bias
            setattr(model, name, bn)
        elif len(list(module.children())) > 0:
            replace_in_by_bn(module)
