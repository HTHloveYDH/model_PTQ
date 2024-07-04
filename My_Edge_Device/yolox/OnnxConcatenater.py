import onnx
from onnx import helper, numpy_helper, TensorProto


class OnnxConcatenater:
    def __init__(self, onnx_model_a_path:str, onnx_model_b_path:str):
        self.onnx_model_a_path = onnx_model_a_path
        self.onnx_model_b_path = onnx_model_b_path

    def merge_models(self, model_a, model_b, b_suffix=None):
        if b_suffix is not None:
            assert isinstance(b_suffix, str)
            assert b_suffix.startswith('_')
        if isinstance(model_a, str):
            model_a = OnnxConcatenater.load_onnx_model(model_a)
        if isinstance(model_b, str):
            model_b = OnnxConcatenater.load_onnx_model(model_b)
        # add suffix in case of potential conflict
        OnnxConcatenater.rename_nodes(model_b, b_suffix)
        OnnxConcatenater.rename_initializers(model_b, b_suffix)
        OnnxConcatenater.rename_value_info(model_b, b_suffix)
        OnnxConcatenater.rename_input(model_b, b_suffix)
        OnnxConcatenater.rename_output(model_b, b_suffix)
        # get output from model_a and input from model_b
        a_output_name = OnnxConcatenater.get_output_name(model_a)
        b_input_name = OnnxConcatenater.get_input_name(model_b)
        OnnxConcatenater.link(model_b, b_input_name, a_output_name)
        # merge graph of model_a and model_b
        merged_nodes = list(model_a.graph.node) + list(model_b.graph.node)
        merged_initializers = list(model_a.graph.initializer) + list(model_b.graph.initializer)
        merged_inputs = model_a.graph.input
        merged_outputs = model_b.graph.output
        merged_graph = helper.make_graph(
            nodes=merged_nodes,
            name="MergedGraph",
            inputs=merged_inputs,
            outputs=merged_outputs,
            initializer=merged_initializers
        )
        # create merged model
        merged_model = helper.make_model(merged_graph, producer_name='merged_model')
        return merged_model

    @staticmethod
    def load_onnx_model(path:str):
        return onnx.load(path)

    @staticmethod
    def get_output_name(model):
        return model.graph.output[0].name

    @staticmethod
    def get_input_name(model):
        return model.graph.input[0].name

    @staticmethod
    def rename_input(model, suffix:str):
        model.graph.input[0].name = model.graph.input[0].name + suffix

    @staticmethod
    def rename_output(model, suffix:str):
        model.graph.output[0].name = model.graph.output[0].name + suffix

    @staticmethod
    def rename_nodes(model, suffix:str):
        for node in model.graph.node:
            node.name = node.name + suffix
            for i in range(len(node.input)):
                node.input[i] = node.input[i] + suffix
            for i in range(len(node.output)):
                node.output[i] = node.output[i] + suffix

    @staticmethod
    def rename_initializers(model, suffix:str):
        for initializer in model.graph.initializer:
            initializer.name = initializer.name + suffix

    @staticmethod
    def rename_value_info(model, suffix:str):
        for value_info in model.graph.value_info:
            value_info.name = value_info.name + suffix
    
    @staticmethod
    def link(model_b, b_input_name:str, a_output_name:str):
        # [important] rename input name of nodes in model_b in order to link to output of mode_a
        for node in model_b.graph.node:
            for i in range(len(node.input)):
                if node.input[i] == b_input_name:
                    node.input[i] = a_output_name


if __name__ == '__main__':
    # load onnxx models
    model_a = onnx.load("/content/FocusV2_without_2nd_conv.onnx")
    model_b = onnx.load("/content/yolox_tiny_without_beginning_slices.onnx")
    onnx_concatenater = OnnxConcatenater("/content/export_model.onnx", "/content/yolox_tiny_part.onnx")
    # merge models
    merged_model = onnx_concatenater.merge_models(model_a, model_b, '_b')
    # save merged model
    onnx.save(merged_model, "merged_model.onnx")
    print("Models merged successfully and saved as 'merged_model.onnx'")
