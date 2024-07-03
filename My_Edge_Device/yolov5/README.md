## for yolov5 option1#
### step 1 clone official repo
git clone https://github.com/ultralytics/yolov5.git
### step 2 modify some code in official repo
Sustitute class 'Detect' in https://github.com/ultralytics/yolov5/blob/master/models/yolo.py by model_PTQ/My_Edge_Device/yolov5/Detect.py.

## for yolov5 option2#
### step 1 clone official repo
git clone https://github.com/ultralytics/yolov5.git
### step 2 modify some code in official repo
Sustitute class 'TFDetect' in https://github.com/ultralytics/yolov5/blob/master/models/tf.py by model_PTQ/My_Edge_Device/yolov5/TFDetect.py.
