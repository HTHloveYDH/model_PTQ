import os

import onnx
import numpy as np
from onnx import helper, numpy_helper


def replace_InstanceNorm_with_base_ops_in_ONNX(onnx_model_path:str):
    # load onnx original model
    model = onnx.load(onnx_model_path)
    graph = model.graph
    # find InstanceNormalization node
    for node_index, node in enumerate(graph.node):
        if node.op_type == "InstanceNormalization":
            # get input and outputs from InstanceNormalization node
            input_name = node.input[0]
            scale_name = node.input[1]
            bias_name = node.input[2]
            output_name = node.output[0]
            print(output_name)
            scale = None
            bias = None
            for initializer in graph.initializer:
                if initializer.name == scale_name:
                    scale = numpy_helper.to_array(initializer)
                if initializer.name == bias_name:
                    bias = numpy_helper.to_array(initializer)

            if scale is None or bias is None:
                raise ValueError("Could not find scale or bias initializers.")

            # get the number of input channels
            num_channels = scale.shape[0]

            # add suffix in order to avoid same tensor name or node name
            suffix = f"_bn_{node_index}"

            # mean
            mean_node = helper.make_node(
                'ReduceMean',
                inputs=[input_name],
                outputs=[input_name + '_mean' + suffix],
                axes=[2, 3],  # compute on H W dim
                keepdims=True,
                name=f'replace_{node_index}_InstanceNorm_ReduceMean_1'
            )

            # variance
            sub_node = helper.make_node(
                'Sub',
                inputs=[input_name, input_name + '_mean' + suffix],
                outputs=[input_name + '_sub' + suffix],
                name=f'replace_{node_index}_InstanceNorm_Sub_1'
            )

            square_node = helper.make_node(
                'Mul',
                inputs=[input_name + '_sub' + suffix, input_name + '_sub' + suffix],
                outputs=[input_name + '_square' + suffix],
                name=f'replace_{node_index}_InstanceNorm_Mul_1'
            )

            var_node = helper.make_node(
                'ReduceMean',
                inputs=[input_name + '_square' + suffix],
                outputs=[input_name + '_var' + suffix],
                axes=[2, 3],
                keepdims=True,
                name=f'replace_{node_index}_InstanceNorm_ReduceMean_2'
            )

            # epsilon
            epsilon_node = helper.make_node(
                'Add',
                inputs=[input_name + '_var' + suffix, f'epsilon_{node_index}'],
                outputs=[input_name + '_var_epsilon' + suffix],
                name=f'replace_{node_index}_InstanceNorm_Add_1'
            )

            # std
            sqrt_node = helper.make_node(
                'Sqrt',
                inputs=[input_name + '_var_epsilon' + suffix],
                outputs=[input_name + '_std' + suffix],
                name=f'replace_{node_index}_InstanceNorm_Sqrt_1'
            )

            # normalize
            normalize_node = helper.make_node(
                'Div',
                inputs=[input_name + '_sub' + suffix, input_name + '_std' + suffix],
                outputs=[input_name + '_normalized' + suffix],
                name=f'replace_{node_index}_InstanceNorm_Div_1'
            )

            scale_reshape = helper.make_node(
                'Reshape',
                inputs=[scale_name, f'shape_{node_index}'],
                outputs=[scale_name + '_reshaped' + suffix],
                name=f'replace_{node_index}_InstanceNorm_Reshape_2'
            )

            bias_reshape = helper.make_node(
                'Reshape',
                inputs=[bias_name, f'shape_{node_index}'],
                outputs=[bias_name + '_reshaped' + suffix],
                name=f'replace_{node_index}_InstanceNorm_Reshape_1'
            )

            # affine
            scale_node = helper.make_node(
                'Mul',
                inputs=[input_name + '_normalized' + suffix, scale_name + '_reshaped' + suffix],
                outputs=[input_name + '_scaled' + suffix],
                name=f'replace_{node_index}_InstanceNorm_Mul_2'
            )

            bias_node = helper.make_node(
                'Add',
                inputs=[input_name + '_scaled' + suffix, bias_name + '_reshaped' + suffix],
                outputs=[output_name],
                name=f'replace_{node_index}_InstanceNorm_Add_2'
            )

            # delete InstanceNormalization node
            graph.node.remove(node)
            # add new nodes to graph.node
            insert_index = node_index
            for n in [mean_node, sub_node, square_node, var_node, epsilon_node, sqrt_node, normalize_node, scale_reshape, bias_reshape, scale_node, bias_node]:
                graph.node.insert(insert_index, n)
                insert_index += 1

            # add 'epsilon' initializer to graph.initializer for Add node
            epsilon_tensor = helper.make_tensor(
                name=f'epsilon_{node_index}',
                data_type=onnx.TensorProto.FLOAT,
                dims=[],
                vals=[1e-5]  # avoid ZeroDivision error
            )
            graph.initializer.append(epsilon_tensor)
            # add 'shape' initializer to graph.initializer for Reshape node
            shape_tensor = helper.make_tensor(
                name=f'shape_{node_index}',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=[1, num_channels, 1, 1]
            )
            graph.initializer.append(shape_tensor)
            break
    # save modified onnx model
    onnx.checker.check_model(model)
    save_onnx_model_dir = os.path.dirname(onnx_model_path)
    onnx.save(model, os.path.join(f'{save_onnx_model_dir}', 'modified_model.onnx'))
    print("Model has been modified and saved as 'modified_model.onnx'")


if __name__ == '__main__':
    import onnxruntime as nxrun

    ori_onnx_odel_path = '/content/faceID_model_2_0.onnx'
    modified_onnx_odel_path = '/content/modified_model.onnx'
    dummy_input = np.random.randn(1, 1, 224, 224).astype(np.float32)
    replace_InstanceNorm_with_base_ops_in_ONNX(ori_onnx_odel_path)
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
