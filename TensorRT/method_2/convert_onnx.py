import onnx
from onnxsim import simplify

import torch
import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

from fp32_train import Net

def convert(quantization_path, onnx_path):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    quant_modules.initialize()
    model = Net()
    
    # load the calibrated model
    state_dict = torch.load(quantization_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.cuda()
    
    dummy_input = torch.randn(1, 3, 32, 32, device='cuda')
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=13, input_names = [ "actual_input_1" ], output_names = [ "output1" ])
    onnx_model = onnx.load(onnx_path)
    optimized_model, check = simplify(onnx_model)
    onnx.save(optimized_model, onnx_path)

if __name__ == "__main__":
    quanization_path = "../output/quant_resnet18-calibrated.pth"
    onnx_path = "../output/quant_resnet18.onnx"

    convert(quanization_path, onnx_path)
