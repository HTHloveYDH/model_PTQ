import torch
import torch.nn as nn


class BatchNorm2dV3(nn.Module):
    def __init__(self, in_channels:int, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None):
        super(BatchNorm2dV3, self).__init__()
        self.running_mean = nn.Parameter(torch.zeros(in_channels), requires_grad=False)  # shape: (c,)
        self.running_var = nn.Parameter(torch.ones(in_channels), requires_grad=False)  # shape: (c,)
        self.in_channels = in_channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype
        if affine:
            # γ and 𝛽 are learnable parameter vectors of size C (where C is the input size)
            self.weight = nn.Parameter(torch.ones(in_channels), requires_grad=True)  # shape: (c,)
            self.bias = nn.Parameter(torch.zeros(in_channels), requires_grad=True)  # shape: (c,)
        else:
          self.weight = 1.0
          self.bias = 0.0

    def forward(self, input):
        bs, c, h, w = input.shape
        shape = bs, c, h, w
        if self.training:
            curr_mean = input.mean(dim=(0, 2, 3), keepdim=False)  # shape: (c,)
            curr_var = input.var(dim=(0, 2, 3), keepdim=False)  # shape: (c,)
            if self.track_running_stats:
                self.running_mean = nn.Parameter(curr_mean * self.momentum + self.running_mean * (1.0 - self.momentum))  # shape: (c,)
                self.running_var = nn.Parameter(curr_var * self.momentum + self.running_var * (1.0 - self.momentum))  # shape: (c,)
            output = (input - curr_mean.view(1, -1, 1, 1).expand_as(input)) / torch.sqrt(curr_var.view(1, -1, 1, 1).expand_as(input) + self.eps)  # shape: (bs, c, h, w)
        else:
            output = (input - self.running_mean.view(1, -1, 1, 1).expand_as(input)) / torch.sqrt(self.running_var.view(1, -1, 1, 1).expand_as(input) + self.eps)  # shape: (bs, c, h, w)
        if self.affine:
            output = self.weight.view(1, -1, 1, 1).expand_as(input) * output + self.bias.view(1, -1, 1, 1).expand_as(input)
        return output


if __name__ == '__main__':
    # generate onnx
    # train mode
    torch_input = torch.randn(2, 3, 640, 640)
    BN = BatchNorm2dV3(3, affine=False, track_running_stats=True)
    BN.train()
    print(BN.training, BN.track_running_stats)
    IN(torch_input)
    torch.onnx.export(
        BN,               # model being run
        torch_input,                         # model input (or a tuple for multiple inputs)
        "export_in_train.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
    )
    # eval mode
    torch_input = torch.randn(2, 3, 640, 640)
    BN = BatchNorm2dV3(3, affine=False, track_running_stats=True)
    BN.eval()
    print(BN.training, BN.track_running_stats)
    torch.onnx.export(
        BN,               # model being run
        torch_input,                         # model input (or a tuple for multiple inputs)
        "export_in_eval.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
    )
