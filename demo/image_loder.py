import torchvision.datasets as datasets

image_path = "/home/gu721/yzc/data/ophthalmic_multimodal/OLIVES/"
dataset_train = datasets.ImageFolder(root=image_path, transform=None)
print(dataset_train)