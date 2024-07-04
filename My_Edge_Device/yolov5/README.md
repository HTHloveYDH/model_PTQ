## for yolov5 option1# (.onnx)
### step 1 clone official repo
git clone https://github.com/ultralytics/yolov5.git
### step 2 modify some code in official repo
[option 1#]: run yolov5 in shape: (1, 25200, 85)
Sustitute class 'Detect' in https://github.com/ultralytics/yolov5/blob/master/models/yolo.py by class 'Detect' in model_PTQ/My_Edge_Device/yolov5/onnx/Detect_shape_1_25200_85.py.

[option 2#]: run yolov5 in shape: (3, 85, 8400)
Sustitute class 'Detect' in https://github.com/ultralytics/yolov5/blob/master/models/yolo.py by class 'Detect' in model_PTQ/My_Edge_Device/yolov5/onnx/Detect_shape_3_85_8400.py.

[option 3#]: run yolov5 in shape: (3, 85, 8400) and replace split operations by convolution
Sustitute class 'Detect' in https://github.com/ultralytics/yolov5/blob/master/models/yolo.py by class 'Detect' in model_PTQ/My_Edge_Device/yolov5/onnx/Detect_split_by_conv_shape_3_85_8400.py.

### step 3 export to .onnx model
```bash
cd path/to/yolov5
```
```bash
python3 ./export.py --weights path/to/yolov5s.pt --include onnx --opset 13
```
## for yolov5 option2# (.tflite)
### step 1 clone official repo
git clone https://github.com/ultralytics/yolov5.git
### step 2 modify some code in official repo
Sustitute class 'TFDetect' in https://github.com/ultralytics/yolov5/blob/master/models/tf.py by class 'TFDetect' in model_PTQ/My_Edge_Device/yolov5/tflite/TFDetect.py.
### step 3 export to .tflite model
```bash
cp path/to/model_PTQ/My_Edge_Device/yolov5/tflite/TFDetect.py path/to/yolov5
```
```bash
cd path/to/yolov5
```
```bash
pyhton ./export_to_tflite.py
```
