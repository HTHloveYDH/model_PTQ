import torch
import torch.nn as nn

from InstanceNorm2dV3 import InstanceNorm2dV3
    

if __name__ == '__main__':
    # dummy input
    torch_input = torch.randn(3, 4, 2, 2)
    
    ### dummy training stage
    # my version
    INV3 = InstanceNorm2dV3(4, affine=False, track_running_stats=True)
    print('initial running mean of InstanceNorm2dV3: ', INV3.running_mean)
    print('initial running var of InstanceNorm2dV3: ', INV3.running_var)
    INV3.train()
    y_v3 = INV3(torch_input)
    # pytorch version
    IN = nn.InstanceNorm2d(4, affine=False, track_running_stats=True)
    print('initial running mean of InstanceNorm2d: ', IN.running_mean)
    print('initial running var of InstanceNorm2d: ', IN.running_var)
    IN.train()
    y = IN(torch_input)
    print('updated running mean of InstanceNorm2dV3: ', INV3.running_mean, 'updated running mean of InstanceNorm2d: ', IN.running_mean)
    print('updated running var of InstanceNorm2dV3: ', INV3.running_var, 'updated running var of InstanceNorm2d: ', IN.running_var)
    
    ### dummy eval stage
    INV3.eval()
    y_v3_eval = INV3(torch_input)
    IN.eval()
    y_eval = IN(torch_input)
    print(abs(y_eval - y_v3_eval))
