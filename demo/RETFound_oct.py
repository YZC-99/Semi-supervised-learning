import torch
import timm.models.vision_transformer as timm_vit
ckpt_path = "/dk1/oct_exp/RETFound_oct_weights.pth"
ckpt_dict = torch.load(ckpt_path,map_location='cpu')
print(ckpt_dict.keys())
print(ckpt_dict['model'].keys())

model = timm_vit.vit_large_patch16_224(pretrained=False,num_classes=4)
model.load_state_dict(ckpt_dict['model'],strict=False)
print(model)
