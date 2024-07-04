## for yolov5
### step 1 clone official repo
git clone https://github.com/ultralytics/yolov5.git
### step 2 modify some code in official repo
option 1#
Sustitute class 'Detect' in https://github.com/ultralytics/yolov5/blob/master/models/yolo.py by model_PTQ/QC_SNPE/yolov5/DetectV1.py.

option 2#
Sustitute class 'Detect' in https://github.com/ultralytics/yolov5/blob/master/models/yolo.py by model_PTQ/QC_SNPE/yolov5/DetectV2.py.
