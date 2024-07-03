## for deploying yolox in shape: (1, 85, 8400) instead of (1, 8400, 85)
### step 1 clone official repo
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
### step 2 modify some code in official repo
  1. Modify https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py according to model_PTQ/My_Edge_Device/yolox/yolo_head.py.