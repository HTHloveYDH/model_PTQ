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
```bash
snpe-onnx-to-dlc --input_network models/my_model/my_model.onnx --output_path path/to/my_model.dlc
```

#### for tflite
```bash
snpe-tflite-to-dlc --input_network models/my_model/my_model.tflite --input_dim input "1,224,224,3" --output_path path/to/my_model.dlc
```

#### for tensorflow frozen_graph.pb
##### A trained TensorFlow model consists of either:
###### 1. A frozen TensorFlow model (pb file) OR
###### 2. A pair of checkpoint and graph meta files
###### 3. A SavedModel directory (Tensorflow 2.x)
```bash
snpe-tensorflow-to-dlc --input_network models/my_model/my_frozen_graph_model.pb --input_dim input "1,224,224,3" --out_node "output_node_name" --output_path path/to/my_frozen_graph_model.dlc
```

#### for Pytorch .pt
```bash
snpe-pytorch-to-dlc --input_network models/my_model/my_model.pt --input_dim input "1,3,224,224" --output_path path/to/my_model.dlc
```

### step 5: convert to .dlc (with PTQ)
run following command:
```bash
snpe-dlc-quantize --input_dlc path/to/my_model.dlc --input_list path/to/image_file_list.txt --output_dlc path/to/my_q_model.dlc --optimizations cle --optimizations bc
```
[Note]: Each path in "path/to/image_file_list.txt" is expected to point to a binary file containing one trial input in the 'raw' format, ready to be consumed by SNPE without any further modifications

