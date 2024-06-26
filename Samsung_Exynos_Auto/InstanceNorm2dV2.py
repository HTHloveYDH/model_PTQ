import torch
import torch.nn as nn


class InstanceNorm2dV2(nn.Module):
    def __init__(self, in_channels:int, momentum=0.1, eps=1e-5, affine=False, track_running_stats=False,
                 device=None, dtype=None):
        super(InstanceNorm2dV2, self).__init__()
        self.run_mean = torch.zeros(1, in_channels, 1, 1, requires_grad=False)
        self.run_var = torch.zeros(1, in_channels, 1, 1, requires_grad=False)
        self.in_channels = in_channels
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype
        if affine:
            # γ and 𝛽 are learnable parameter vectors of size C (where C is the input size)
            self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1), requires_grad=True)
            self.beta = nn.Parameter(torch.ones(1, in_channels, 1, 1), requires_grad=True)

    def forward(self, input):
        # bs, c, h, w = input.shape
        if self.track_running_stats and self.training:
            curr_mean = input.mean(dim=(2, 3), keepdim=True)  # (bs, c, 1, 1)
            curr_var = input.var(dim=(2, 3), keepdim=True)  # (bs, c, 1, 1)
            self.run_mean = curr_mean * self.momentum + self.run_mean * (1.0 - self.momentum)
            self.run_var = curr_var * self.momentum + self.run_var * (1.0 - self.momentum)
        output = (input - self.run_mean) / torch.sqrt(self.run_var + self.eps)
        if self.affine:
            output = self.gamma * output + self.beta
        return output
    

if __name__ == '__main__':
    # train
    torch_input = torch.randn(1, 3, 640, 640)
    IN = InstanceNorm2dV2(3, affine=True, track_running_stats=True)
    IN.train()
    print(IN.training, IN.track_running_stats)
    torch.onnx.export(
        IN,               # model being run
        torch_input,                         # model input (or a tuple for multiple inputs)
        "export_in.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
    )
    # eval
    torch_input = torch.randn(1, 3, 640, 640)
    IN = InstanceNorm2dV2(3, affine=True, track_running_stats=True)
    IN.eval()
    print(IN.training)
    torch_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        IN,               # model being run
        torch_input,                         # model input (or a tuple for multiple inputs)
        "export_in.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
    )