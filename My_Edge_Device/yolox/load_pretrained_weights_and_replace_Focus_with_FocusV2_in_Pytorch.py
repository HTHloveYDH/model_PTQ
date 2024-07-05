# add line 2 in this file to line 4 in YOLOX/tools/export_onnx.py
from yolox.models.network_blocks import FocusV2

# add line 5 - line 12 in this file to line 13 in YOLOX/tools/export_onnx.py
def replace_Focus_with_FocusV2(model):
    for name, module in model.named_children():
        if isinstance(module, Focus):
            focusv2 = FocusV2(in_channels=module.conv.conv.in_channels, out_channels=module.conv.conv.out_channels)
            focusv2.conv2 = module.conv
            setattr(model, name, focusv2)
        elif len(list(module.children())) > 0:
            replace_Focus_with_FocusV2(module)


# add line 16 in this file to line 78 after 'model.load_state_dict(ckpt)' in the YOLOX/tools/export_onnx.py
replace_Focus_with_FocusV2(yolox)
