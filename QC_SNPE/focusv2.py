# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class FocusV2(nn.Module):
    def __init__(self):
        super(FocusV2, self).__init__()
        weights = torch.zeros(12, 3, 2, 2)
        for i in range(3):
            weights[i][i][0][0] = 1.
            weights[i+3][i][1][0] = 1.
            weights[i+6][i][0][1] = 1.
            weights[i+9][i][1][1] = 1.
        self.conv1 = nn.Conv2d(3, 12, 2, 2, padding=0, bias=False)
        self.conv1.weight = nn.Parameter(weights, requires_grad=False)
        self.conv2 = nn.Conv2d(12, 48, 3, 1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 640, 640)
    focus = FocusV2()
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