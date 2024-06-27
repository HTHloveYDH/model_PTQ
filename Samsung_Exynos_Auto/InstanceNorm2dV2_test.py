import torch
import torch.nn as nn

from InstanceNorm2dV2 import InstanceNorm2dV2
    

if __name__ == '__main__':
    # dummy input
    torch_input = torch.randn(3, 4, 2, 2)
    
    ### dummy training stage
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
    
    ### dummy eval stage
    INV2.eval()
    y_v2_eval = INV2(torch_input)
    IN.eval()
    y_eval = IN(torch_input)
    print(abs(y_eval - y_v2_eval))

    ### test InstanceNorm2dV2 on GPU
    test_model = TestModel()
    # print(test_model.parameters())
    for param in test_model.parameters():
        print(param.device)
    test_model.cuda()
    for param in test_model.parameters():
        print(param.device)
    print(test_model(torch_input.cuda()).shape)
