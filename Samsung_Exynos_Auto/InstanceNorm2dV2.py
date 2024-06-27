import torch
import torch.nn as nn


class InstanceNorm2dV2(nn.Module):
    def __init__(self, in_channels:int, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False,
                 device=None, dtype=None):
        super(InstanceNorm2dV2, self).__init__()
        if track_running_stats:
          self.running_mean = torch.zeros(in_channels, requires_grad=False)  # shape: (c,)
          self.running_var = torch.ones(in_channels, requires_grad=False)  # shape: (c,)
        else:
          self.run_mean = None
          self.run_var = None
        self.in_channels = in_channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype
        if affine:
            # γ and 𝛽 are learnable parameter vectors of size C (where C is the input size)
            self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1), requires_grad=True)  # shape: (1, c, 1, 1)
            self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)  # shape: (1, c, 1, 1)
        else:
          self.gamma = 1.0
          self.beta = 0.0

    def forward(self, input):
        bs, c, h, w = input.shape
        shape = bs, c, h, w
        if self.training:
            curr_mean = input.mean(dim=(2, 3), keepdim=False).mean(0)  # shape: (c,)
            curr_var = input.var(dim=(2, 3), keepdim=False).mean(0)  # shape: (c,)
            if self.track_running_stats:
                self.running_mean = curr_mean * self.momentum + self.running_mean * (1.0 - self.momentum)  # shape: (c,)
                self.running_var = curr_var * self.momentum + self.running_var * (1.0 - self.momentum)  # shape: (c,)
            output = (input - curr_mean.view(1, -1, 1, 1)) / torch.sqrt(curr_var.view(1, -1, 1, 1) + self.eps)  # shape: (bs, c, h, w)
        else:
            output = (input - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)  # shape: (bs, c, h, w)
        if self.affine:
            output = self.gamma * output + self.beta
        return output
    

if __name__ == '__main__':
    # generate onnx
    # train mode
    torch_input = torch.randn(2, 3, 640, 640)
    IN = InstanceNorm2dV2(3, affine=False, track_running_stats=True)
    IN.train()
    print(IN.training, IN.track_running_stats)
    IN(torch_input)
    torch.onnx.export(
        IN,               # model being run
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
    IN = InstanceNorm2dV2(3, affine=False, track_running_stats=True)
    IN.eval()
    print(IN.training, IN.track_running_stats)
    torch.onnx.export(
        IN,               # model being run
        torch_input,                         # model input (or a tuple for multiple inputs)
        "export_in_eval.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
    )
