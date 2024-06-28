# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class FocusV2(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super(FocusV2, self).__init__()
        weights = torch.zeros(in_channels * 4, in_channels, 2, 2)
        for i in range(in_channels):
            weights[i][i][0][0] = 1.
            weights[i+in_channels*1][i][1][0] = 1.
            weights[i+in_channels*2][i][0][1] = 1.
            weights[i+in_channels*3][i][1][1] = 1.
        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, 2, 2, padding=0, bias=False)
        self.conv1.weight = nn.Parameter(weights, requires_grad=False)
        self.conv2 = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 640, 640)
    focus = FocusV2(3, 48, 3)
    y = focus(x)
    print(y.shape)
    # convert to .onnx via torch.onnx.dynamo_export
    torch_input = torch.randn(1, 3, 640, 640)
    onnx_program = torch.onnx.dynamo_export(focus, torch_input)
    onnx_program.save("dynamo_export_model.onnx")
    # convert to .onnx via torch.onnx.export
    torch.onnx.export(
        focus,               # model being run
        torch_input,                         # model input (or a tuple for multiple inputs)
        "export_model.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
    )
