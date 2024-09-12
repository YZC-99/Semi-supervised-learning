import timm

model_list = timm.list_models()
with open('model_list.txt', 'w') as f:
    for model in model_list:
        f.write(model + '\n')