import argparse

import tensorflow as tf

from models.yolo import Model
from models.tf import TFModel
from models.experimental import attempt_load


def export_to_tflite(weights:str, yaml_path='./yolov5s.yaml', save_tflite_model_path='./yolov5s-fp32.tflite', 
                     imgsz=640, ch=3, batch_size=1, nc=80, precision='fp32', device="cpu"):
    if weights:
        model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model
    else:
        model = Model('/content/yolov5s.yaml')
    model.model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.parameters():
        print(param.requires_grad)
    tfmodel = TFModel(yaml_path, nc=nc, model=model)
    im = tf.zeros((batch_size, imgsz, imgsz, ch))  # BHWC order for TensorFlow
    _ = tfmodel.predict(im)
    inputs = tf.keras.Input(shape=(imgsz, imgsz, ch), batch_size=batch_size)
    outputs = tfmodel.predict(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if precision == 'fp16':
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # TODO: support int8
    # if precision == 'int8':
    #     ...
    tflite_model = converter.convert()
    open(save_tflite_model_path, "wb").write(tflite_model)
