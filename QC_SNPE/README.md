# PTQ Steps for QC_SNPE

## for yolox
### step 1 clone official repo
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
### step 2 modify some code in official repo
  add 'class FocusV2' in model_PTQ/QC_SNPE/FocusV2.py to https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/network_blocks.py

## for yolov5
### step 1 clone official repo
git clone https://github.com/ultralytics/yolov5.git
### step 2 modify some code in official repo
sustitute https://github.com/ultralytics/yolov5/blob/master/models/yolo.py by model_PTQ/QC_SNPE/yolo.py
