## for yolox
### step 1 clone official repo
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
### step 2 modify some code in official repo
1. Add 'class FocusV2' defined in model_PTQ/My_Edge_Device/yolox/FocusV2.py to https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/network_blocks.py.
2. Change this line "from .network_blocks import BaseConv, CSPLayer, DWConv, FocusV2, ResLayer, SPPBottleneck" in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py to "from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, FocusV2, ResLayer, SPPBottleneck".
3. Change this line "self.stem = Focus(3, base_channels, ksize=3, act=act)" in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py to "self.stem = FocusV2(3, base_channels, ksize=3, act=act)".
4. [option 1#] run yolox in shape: (1, 8400, 85)

Modify class 'YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head_shape_1_8400_85.py.
6. [option 2#] run yolox in shape: (1, 85, 8400)
   
Modify class 'YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head_shape_1_85_8400.py.
7. [option 2#] run yolox in shape: (1, 85, 8400)

Modify class 'YOLOXHead' in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head_split_by_conv_shape_1_85_8400.py.
