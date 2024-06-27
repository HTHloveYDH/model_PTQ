import torch
import torch.nn as nn

from InstanceNorm2dV3 import InstanceNorm2dV3


class TestGPUModel(nn.Module):
    def __init__(self):
        super(m3, self).__init__()
        self.IN = InstanceNorm2dV3(4, affine=True, track_running_stats=True)
    def forward(self, x):
        return self.IN(x)


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

    ### test InstanceNorm2dV3 on GPU
    test_gpu_model = TestGPUModel()
    # print(test_gpu_model.parameters())
    for param in test_gpu_model.parameters():
        print(param.device)
    test_gpu_model.cuda()
    for param in test_gpu_model.parameters():
        print(param.device)
    print(test_gpu_model(torch_input.cuda()).shape)
