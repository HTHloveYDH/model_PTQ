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


if __name__ == '__main__':
    # prove that in the eval mode, nn.BatchNorm2d and nn.InstanceNorm2d are equivalent
    BN = nn.BatchNorm2d(3)
    IN = nn.InstanceNorm2d(3, affine=True, track_running_stats=True)
    print(BN.running_mean, IN.running_mean)
    print(BN.running_var, IN.running_var)
    print(BN.weight, IN.weight)
    print(BN.bias, IN.bias)
    torch_input = torch.randn(1, 3, 3, 3)
    BN.eval()
    IN.eval()
    print(BN(torch_input) - IN(torch_input))
