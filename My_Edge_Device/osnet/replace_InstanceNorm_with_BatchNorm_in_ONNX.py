import os

import onnx
import numpy as np
from onnx import helper, numpy_helper


def replace_in_with_bn(onnx_model_path:str):
    onnx_model = onnx.load(onnx_model_path)
    graph = onnx_model.graph
    for node_index, node in enumerate(graph.node):
        if 'InstanceNorm' in node.op_type:
            print('current InstanceNorm Node: ', node)
            if len(node.input) == 5:
                bn_node = onnx.helper.make_node(
                    'BatchNormalization', inputs=list(node.input), outputs=list(node.output), 
                    name=f'replace_{node_index}_InstanceNorm_BatchNorm_1'
                )
            elif len(node.input) == 3:
                for initializer_index, initializer in enumerate(graph.initializer):
                    if initializer.name == node.input[1]:
                        in_channels = initializer.dims[0]  # number of channel                
                # add suffix in order to avoid same tensor name or node name
                suffix = f"_bn_{node_index}"
                input_tensor_name = node.input[0]
                # nodes for mean
                mean_node = helper.make_node(
                    'ReduceMean',
                    inputs=[input_tensor_name],
                    outputs=[input_tensor_name + '_mean' + suffix],
                    axes=[2, 3],  # 1. compute mean on H W 2. [batch must = 1]
                    keepdims=True,
                    name=f'replace_{node_index}_InstanceNorm_ReduceMean_1'
                )

                # nodes for std
                sub_node = helper.make_node(
                    'Sub',
                    inputs=[input_tensor_name, input_tensor_name + '_mean' + suffix],
                    outputs=[input_tensor_name + '_sub' + suffix],
                    name=f'replace_{node_index}_InstanceNorm_Sub_1'
                )

                square_node = helper.make_node(
                    'Mul',
                    inputs=[input_tensor_name + '_sub' + suffix, input_tensor_name + '_sub' + suffix],
                    outputs=[input_tensor_name + '_square' + suffix],
                    name=f'replace_{node_index}_InstanceNorm_Mul_1'
                )

                var_node = helper.make_node(
                    'ReduceMean',
                    inputs=[input_tensor_name + '_square' + suffix],
                    outputs=[input_tensor_name + '_var' + suffix],
                    axes=[2, 3],  # 1. compute mean on H W 2. [batch must = 1]
                    keepdims=True,
                    name=f'replace_{node_index}_InstanceNorm_ReduceMean_2'
                )

                running_mean_reshape = helper.make_node(
                    'Reshape',
                    inputs=[input_tensor_name + '_mean' + suffix, f'shape_{node_index}'],
                    outputs=[input_tensor_name + '_running_mean' + suffix],
                    name=f'replace_{node_index}_InstanceNorm_Reshape_2'
                )

                running_var_reshape = helper.make_node(
                    'Reshape',
                    inputs=[input_tensor_name + '_var' + suffix, f'shape_{node_index}'],
                    outputs=[input_tensor_name + '_running_var' + suffix],
                    name=f'replace_{node_index}_InstanceNorm_Reshape_1'
                )

                bn_ndoe_inputs = list(node.input) + [input_tensor_name + '_running_mean' + suffix, input_tensor_name + '_running_var' + suffix]
                bn_node = onnx.helper.make_node(
                    'BatchNormalization', inputs=bn_ndoe_inputs, outputs=list(node.output), 
                    name=f'replace_{node_index}_InstanceNorm_BatchNorm_1'
                )
                for attr in node.attribute:
                    bn_node.attribute.append(attr)
                
                # delete InstanceNormalization node
                graph.node.remove(node)
                # add new node to graph.node
                insert_index = node_index
                for n in [mean_node, sub_node, square_node, var_node, running_mean_reshape, running_var_reshape, bn_node]:
                    graph.node.insert(insert_index, n)
                    insert_index += 1

                # add tensor 'shape' to initializer for Reshape node
                shape_tensor = helper.make_tensor(
                    name=f'shape_{node_index}',
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[in_channels]
                )
                graph.initializer.append(shape_tensor)
                print(' [relace to]:')
                print('target BatchNorm Node: ', bn_node)
                print('=================================================')
            else:
                raise ValueError(f'length of input of InstanceNorm is {len(node.input)}, which is not supported!')
    onnx.checker.check_model(onnx_model)
    save_onnx_model_dir = os.path.dirname(onnx_model_path)
    onnx.save(onnx_model, os.path.join(f'{save_onnx_model_dir}', 'modified_model.onnx'))


if __name__ == '__main__':
    import onnxruntime as nxrun

    ori_onnx_odel_path = '/content/faceID_model_2_0.onnx'
    modified_onnx_odel_path = '/content/modified_model.onnx'
    dummy_input = np.random.randn(1, 1, 224, 224).astype(np.float32)
    replace_in_with_bn(ori_onnx_odel_path)
    sess1 = nxrun.InferenceSession(ori_onnx_odel_path)
    input_name = sess1.get_inputs()[0].name
    # input_shape = sess1.get_inputs()[0].shape
    # print(input_name, input_shape)
    infer1 = sess1.run(None, {input_name: dummy_input})
    
    sess2 = nxrun.InferenceSession(modified_onnx_odel_path)
    input_name = sess2.get_inputs()[0].name
    # input_shape = sess1.get_inputs()[0].shape
    # print(input_name, input_shape)
    infer2 = sess2.run(None, {input_name: dummy_input})
    print(len(infer1), infer1[0].shape)
    print(len(infer2), infer2[0].shape)
    print(np.argmax(infer1[0]), np.argmax(infer2[0]))
    print(infer1[0] - infer2[0])
