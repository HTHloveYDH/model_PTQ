## for yolov5
### step 1 clone official repo
git clone https://github.com/ultralytics/yolov5.git
### step 2 modify some code in official repo
Sustitute class 'Detect' in https://github.com/ultralytics/yolov5/blob/master/models/yolo.py by class 'Detect' in model_PTQ/QC_SNPE/yolov5/Detect.py.
# step 3 export to .onnx model
```bash
cd path/to/yolov5
```
```bash
python3 ./export.py --weights path/to/yolov5s.pt --include onnx --opset 13
```
