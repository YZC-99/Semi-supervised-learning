import timm
import torchvision.models as models
model_list = timm.list_models()
# with open('model_list.txt', 'w') as f:
#     for model in model_list:
#         f.write(model + '\n')

# model_names = sorted(
#     name
#     for name in models.__dict__
#     if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
# )
print(model_list)
