## for yolox without loading pretrained weights
### step 1 clone official repo
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
### step 2 modify some code in official repo
1. Add 'class FocusV2' defined in model_PTQ/My_Edge_Device/yolox/FocusV2.py to https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/network_blocks.py.
2. Change this line "from .network_blocks import BaseConv, CSPLayer, DWConv, FocusV2, ResLayer, SPPBottleneck" in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py to "from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, FocusV2, ResLayer, SPPBottleneck".
3. Change this line "self.stem = Focus(3, base_channels, ksize=3, act=act)" in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py to "self.stem = FocusV2(3, base_channels, ksize=3, act=act)".
4. [option 1#: run yolox in shape: (1, 8400, 85)] Modify class 'YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head_shape_1_8400_85.py.
6. [option 2#: run yolox in shape: (1, 85, 8400)] Modify class 'YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head_shape_1_85_8400.py.
7. [option 2#: run yolox in shape: (1, 85, 8400)] Modify class 'YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head_split_by_conv_shape_1_85_8400.py.

## for yolox with loading pretrained weights (without FoucusV2 module) for testing and fintuning
### step 1 clone official repo
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
### step 2 modify some code in official repo
1. Add 'class FocusV2' defined in model_PTQ/My_Edge_Device/yolox/FocusV2.py to https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/network_blocks.py.
### step 3 copy to the root directory of YOLOX
```bash
cp path/to/load_pretrained_weights_and_replace_Focus_with_FocusV2_in_Pytorch.py path/to/YOLOX
cd /path/to/YOLOX
```
### step 4 run and new .pt file for yolox with Focus module replaced by FocusV2 module will be available.
```bash python ./load_pretrained_weights_and_replace_Focus_with_FocusV2_in_Pytorch.py
```
### step 5 modify some code in official repo
1. Change this line "from .network_blocks import BaseConv, CSPLayer, DWConv, FocusV2, ResLayer, SPPBottleneck" in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py to "from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, FocusV2, ResLayer, SPPBottleneck".
2. Change this line "self.stem = Focus(3, base_channels, ksize=3, act=act)" in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py to "self.stem = FocusV2(3, base_channels, ksize=3, act=act)".
3. [option 1#: run yolox in shape: (1, 8400, 85)] Modify class 'YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head_shape_1_8400_85.py.
4. [option 2#: run yolox in shape: (1, 85, 8400)] Modify class 'YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head_shape_1_85_8400.py.
5. [option 2#: run yolox in shape: (1, 85, 8400)] Modify class 'YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head_split_by_conv_shape_1_85_8400.py.
