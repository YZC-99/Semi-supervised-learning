import torch
import timm.models.vision_transformer as timm_vit

model_ckpt = "/dk1/oct_exp/pretrain-vit_base_patch16_224_/SimCLR/ckpt_epoch_25.pth"
model_dict = torch.load(model_ckpt,map_location='cpu')['model']
# 由于是多卡训练，所以需要去掉module
model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
model_dict = {k.replace('encoder.', ''): v for k, v in model_dict.items()}

timm_model_dict = timm_vit.vit_base_patch16_224(pretrained=True).state_dict()

# 比较两个字典里面相同的key的value形状是否一致
for k in model_dict.keys():
    if k in timm_model_dict.keys():
        if model_dict[k].shape != timm_model_dict[k].shape:
            print(k, model_dict[k].shape, timm_model_dict[k].shape)
    else:
        print(k, "not in timm_model_dict")
