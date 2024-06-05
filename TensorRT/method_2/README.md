# Quantization_pytorch_tensorrt
Here is an example of how to use TensorRT python_quantization toolkit to complete PTQ.


### 0.Requirement
``` bash
# torch >= 1.9.1
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
pip install tensorrt
```

### 1.Quantization Step

#### Step 1: Train FP32 model 
``` bash
python quantization_code/fp32_train.py
```

#### Step 2: Get PTQ model
``` bash
python quantization_code/ptq.py
```

#### Step 3: Convert .pth to .onnx
``` bash
python quantization_code/convert_onnx.py
```

#### Step 4: Convert .onnx to .trt by 'trtexec' on Jetson
``` bash
trtexec --onnx=model_ptq.onnx --int8 --saveEngine=model_ptq.engine --verbose
```
