import torch
import torch.nn as nn

from InstanceNorm2dV2 import InstanceNorm2dV2
    

if __name__ == '__main__':
    torch_input = torch.randn(3, 4, 2, 2)
    # my version
    INV2 = InstanceNorm2dV2(4, affine=False, track_running_stats=True)
    print('initial running mean of InstanceNorm2dV2: ', INV2.running_mean)
    print('initial running var of InstanceNorm2dV2: ', INV2.running_var)
    INV2.train()
    y_v2 = INV2(torch_input)
    # pytorch version
    IN = nn.InstanceNorm2d(4, affine=False, track_running_stats=True)
    print('initial running mean of InstanceNorm2d: ', IN.running_mean)
    print('initial running var of InstanceNorm2d: ', IN.running_var)
    IN.train()
    y = IN(torch_input)
    print('updated running mean of InstanceNorm2dV2: ', INV2.running_mean, 'updated running mean of InstanceNorm2d: ', IN.running_mean)
    print('updated running var of InstanceNorm2dV2: ', INV2.running_var, 'updated running var of InstanceNorm2d: ', IN.running_var)
    print(abs(y - y_v2))
