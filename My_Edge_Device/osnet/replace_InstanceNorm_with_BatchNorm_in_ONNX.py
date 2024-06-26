import os

import onnx
import numpy as np


def replace_in_with_bn(onnx_model_path:str):
    onnx_model = onnx.load(onnx_model_path)
    graph = onnx_model.graph
    in_node_index = 0
    additional_initializers = []
    for node_index, node in enumerate(graph.node):
        if 'InstanceNorm' in node.name:
            # print('current InstanceNorm Node: ', node)
            if len(node.input) == 5:
                bn_node = onnx.helper.make_node(
                    'BatchNormalization', inputs=list(node.input), outputs=list(node.output), name=f'replace_{in_node_index}_{node.name}'
                )
            elif len(node.input) == 3:
                inputs = list(node.input) + [f'replace_{in_node_index}_{node.name}_running_mean', f'replace_{in_node_index}_{node.name}_running_var']
                for initializer_index, initializer in enumerate(graph.initializer):  
                    if initializer.name == node.input[1]:
                        in_channels = initializer.dims[0]  # number of channel
                        running_mean = onnx.helper.make_tensor(
                            f'replace_{in_node_index}_{node.name}_running_mean', onnx.TensorProto.FLOAT, [in_channels], 
                            np.zeros(in_channels, dtype=float)
                        )
                        running_var = onnx.helper.make_tensor(
                            f'replace_{in_node_index}_{node.name}_running_var', onnx.TensorProto.FLOAT, [in_channels], 
                            np.ones(in_channels, dtype=float)
                        )
                additional_initializers += [running_mean, running_var]
                bn_node = onnx.helper.make_node(
                    'BatchNormalization', inputs=inputs, outputs=list(node.output), name=f'replace_{in_node_index}_{node.name}'
                )
                print('target BatchNorm Node: ', bn_node)
            else:
                raise ValueError(f'length of input of InstanceNorm is {len(node.input)}, which is not supported!')
            for attr in node.attribute:
                bn_node.attribute.append(attr)
            graph.node.remove(node)
            graph.node.insert(node_index, bn_node)
            in_node_index += 1
    for initializer in additional_initializers:
        graph.initializer.append(initializer)
    onnx.checker.check_model(onnx_model)
    save_onnx_model_dir = os.path.dirname(onnx_model_path)
    onnx.save(onnx_model, os.path.join(f'{save_onnx_model_dir}', 'modified.onnx'))
