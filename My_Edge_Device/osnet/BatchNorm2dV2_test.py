import torch
import torch.nn as nn

from BatchNorm2dV2 import BatchNorm2dV2


class TestGPUModel(nn.Module):
    def __init__(self, in_channels:int):
        super(TestGPUModel, self).__init__()
        self.bn = BatchNorm2dV2(in_channels, affine=True, track_running_stats=True)
    def forward(self, x):
        return self.bn(x)


if __name__ == '__main__':
    # dummy input
    torch_input = torch.randn(3, 4, 2, 2)
    
    ### dummy training stage
    # my version
    BNV2 = BatchNorm2dV2(4, affine=False, track_running_stats=True)
    print('initial running mean of BatchNorm2dV2: ', BNV2.running_mean)
    print('initial running var of BatchNorm2dV2: ', BNV2.running_var)
    BNV2.train()
    y_v2 = BNV2(torch_input)
    # pytorch version
    BN = nn.BatchNorm2d(4, affine=False, track_running_stats=True)
    print('initial running mean of BatchNorm2d: ', BN.running_mean)
    print('initial running var of BatchNorm2d: ', BN.running_var)
    BN.train()
    y = BN(torch_input)
    print('updated running mean of BatchNorm2dV2: ', BNV2.running_mean, 'updated running mean of BatchNorm2d: ', BN.running_mean)
    print('updated running var of BatchNorm2dV2: ', BNV2.running_var, 'updated running var of BatchNorm2d: ', BN.running_var)
    
    ### dummy eval stage
    BNV2.eval()
    y_v2_eval = BNV2(torch_input)
    BN.eval()
    y_eval = BN(torch_input)
    print(abs(y_eval - y_v2_eval))

    ### test BatchNorm2dV2 on GPU
    test_gpu_model = TestGPUModel()
    # print(test_gpu_model.parameters())
    for param in test_gpu_model.parameters():
        print(param.device)
    test_gpu_model.cuda()
    for param in test_gpu_model.parameters():
        print(param.device)
    print(test_gpu_model(torch_input.cuda()).shape)
