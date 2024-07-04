# PTQ Steps for QC_SNPE

### step 0: 
```bash
cd path/to/your/QC_Snapdragon_Neural_Processing_Engine_SDK
``` 

### step 1: create conda environment
```bash
conda create --name 8295_3.6 --file 6155_env.yaml
```

### step 2: prepare software environment
```bash
path/to/env_configuration.sh
``` 

### step 3: 
```bash
cd ./snpe-x.x.x.x
``` 

### step 4: convert to .dlc (without PTQ)
#### for onnx
snpe-onnx-to-dlc --input_network models/my_model/my_model.onnx --output_path path/to/my_model.dlc

#### for tflite
snpe-tflite-to-dlc --input_network models/my_model/my_model.tflite --input_dim input "1,224,224,3" --output_path path/to/my_model.dlc

#### for tensorflow frozen_graph.pb
##### A trained TensorFlow model consists of either:
###### 1. A frozen TensorFlow model (pb file) OR
###### 2. A pair of checkpoint and graph meta files
###### 3. A SavedModel directory (Tensorflow 2.x)
snpe-tensorflow-to-dlc --input_network models/my_model/my_frozen_graph_model.pb --input_dim input "1,224,224,3" --out_node "output_node_name" --output_path path/to/my_frozen_graph_model.dlc

#### for Pytorch .pt
snpe-pytorch-to-dlc --input_network models/my_model/my_model.pt --input_dim input "1,3,224,224" --output_path path/to/my_model.dlc

### step 5: convert to .dlc (with PTQ)
add "--input_list path/to/image_file_list.txt" option to the commands in step 4
or run following command:
snpe-dlc-quantize --input_dlc path/to/my_model.dlc --input_list path/to/image_file_list.txt --output_dlc path/to/my_q_model.dlc