# PTQ Steps for QC_SNPE

## for yolox
### step 1 clone official repo
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
### step 2 modify some code in official repo
  1. Add 'class FocusV2' defined in model_PTQ/QC_SNPE/yolox/FocusV2.py to https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/network_blocks.py.
  2. Change this line "from .network_blocks import BaseConv, CSPLayer, DWConv, FocusV2, ResLayer, SPPBottleneck" in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py to "from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, FocusV2, ResLayer, SPPBottleneck".
  3. Change this line "self.stem = Focus(3, base_channels, ksize=3, act=act)" in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py to "self.stem = FocusV2(3, base_channels, ksize=3, act=act)".
  4. (optional) change member function 'decode_outputs' of 'class YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to function 'decode_outputs' of 'class YOLOXHead' in model_PTQ/QC_SNPE/yolox/yolo_head.py

## for yolov5
### step 1 clone official repo
git clone https://github.com/ultralytics/yolov5.git
### step 2 modify some code in official repo
Sustitute class 'Detect' in https://github.com/ultralytics/yolov5/blob/master/models/yolo.py by model_PTQ/QC_SNPE/yolov5/Detect.py.
