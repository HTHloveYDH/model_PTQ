import torch
import torch.nn as nn

from BatchNorm2dV3 import BatchNorm2dV3


class TestGPUModel(nn.Module):
    def __init__(self):
        super(m3, self).__init__()
        self.IN = BatchNorm2dV3(4, affine=True, track_running_stats=True)
    def forward(self, x):
        return self.IN(x)


if __name__ == '__main__':
    # dummy input
    torch_input = torch.randn(3, 4, 2, 2)
    
    ### dummy training stage
    # my version
    BNV3 = BatchNorm2dV3(4, affine=False, track_running_stats=True)
    print('initial running mean of BatchNorm2dV3: ', BNV3.running_mean)
    print('initial running var of BatchNorm2dV3: ', BNV3.running_var)
    BNV3.train()
    y_v3 = BNV3(torch_input)
    # pytorch version
    BN = nn.BatchNorm2d(4, affine=False, track_running_stats=True)
    print('initial running mean of BatchNorm2d: ', BN.running_mean)
    print('initial running var of BatchNorm2d: ', BN.running_var)
    BN.train()
    y = BN(torch_input)
    print('updated running mean of BatchNorm2dV3: ', BNV3.running_mean, 'updated running mean of BatchNorm2d: ', BN.running_mean)
    print('updated running var of BatchNorm2dV3: ', BNV3.running_var, 'updated running var of BatchNorm2d: ', BN.running_var)
    
    ### dummy eval stage
    BNV3.eval()
    y_v3_eval = BNV3(torch_input)
    BN.eval()
    y_eval = BN(torch_input)
    print(abs(y_eval - y_v3_eval))

    ### test BatchNorm2dV3 on GPU
    test_gpu_model = TestGPUModel()
    # print(test_gpu_model.parameters())
    for param in test_gpu_model.parameters():
        print(param.device)
    test_gpu_model.cuda()
    for param in test_gpu_model.parameters():
        print(param.device)
    print(test_gpu_model(torch_input.cuda()).shape)
