# PTQ Steps for Nvidia Jetson

### Step 1 Perform following steps on either X86 or Jetson. 
  You can do Step 3 -- Step 5 on PC platform or Jetson platform, so let's assume we do Step 3 -- Step 5 on PC platform and Step 6 -- Step 7 on Jetson (Because we should not always change software environment on Jetson);

### Step 2 Configure environment according to 'requirements.txt'.

### Step 3 Add your own image preprocessing method in 'image_batcher.py' (you can named your own method 'V3') by adding an extra 'elif branch' like following.
elif self.preprocessor == "V3":

### Step 4 Do not forget to modify command line arguments '–calib_preprocessor' in 'build_engine.py' like the following example.
parser.add_argument(
    "--calib_preprocessor",
    default="V2",
    choices=["V1", "V1MS", "V2", "V3"],
    help="Set the calibration image preprocessor to use, either 'V2', 'V1' or 'V1MS', default: V2",
)

### Step 5 Run the following code to generate trt model file 'engine.trt' and calibration file 'calibration.cache'.
``` bash
python3 build_engine.py --onnx /path/to/model.onnx --engine /path/to/engine.trt --precision int8 --calib_input path/to/calibration/images --calib_cache /path/to/calibration.cache -- calib_preprocessor V3
```

### Step 6 (optional) After you have done with final Step 5, if the final result is not as good as you expected, please check the INT8 Calibrator. By default, IInt8EntroyCalibrator is used, but you can change to IInt8MinMaxCalibrator just by changing one line in 'image_batcher.py' as following:
class EngineCalibrator(trt.IInt8EntropyCalibrator2) → class EngineCalibrator(trt.IInt8MinMaxCalibrator)

### Step 7 (optional) Then redo Step 3 -- Step 5.

### Step 8 Keep the generated calibration cache file and copy to your Jetson platform.

### Step 9 Run the following command to regenerate your trt model file 'engine.trt' that can inference on Jetson platform normally.
``` bash
trtexec --onnx=path/to/your/model.onnx --saveEngin=path/to/your/engine.trt --calib=path/to/your/calibration.cache --int8 --verbose
```