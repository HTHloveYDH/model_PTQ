######################## option1 begin

# add line 2 in this file to line 4 in YOLOX/tools/export_onnx.py
from yolox.models.network_blocks import Focus, FocusV2

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

######################## option1 end


######################## option2 begin

import torch

from yolox.models.network_blocks import Focus, FocusV2
from yolox.models.yolox import YOLOX
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolo_pafpn import YOLOPAFPN

def replace_Focus_with_FocusV2(model):
    for name, module in model.named_children():
        if isinstance(module, Focus):
            focusv2 = FocusV2(in_channels=module.conv.conv.in_channels, out_channels=module.conv.conv.out_channels)
            focusv2.conv2 = module.conv
            setattr(model, name, focusv2)
        elif len(list(module.children())) > 0:
            replace_Focus_with_FocusV2(module)


if __name__ == '__main__':
    backbone = YOLOPAFPN(depth=0.33, width=0.50)  # yolox_s
    head = YOLOXHead(num_classes=80, width=0.50)  # yolox_s
    yolox = YOLOX(backbone=backbone, head=head)
    print(yolox)
    yolox.eval()
    print(yolox.head.training)
    print(yolox.training)
    print(yolox.head.decode_in_inference)
    yolox.load_state_dict(torch.load('path/to/yolox_s.pth', map_location=torch.device('cpu'))['model'])
    replace_Focus_with_FocusV2(yolox)
    torch.save(yolox.state_dict(), './yolox_s_with_FocusV2.pt')

######################## option2 end
