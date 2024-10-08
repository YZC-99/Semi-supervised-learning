# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json

import pandas as pd
import torchvision
import numpy as np
import math
from PIL import Image

from torchvision import transforms
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data
from semilearn.datasets.multilabel_utils import split_ssl_multilabel_data



def get_olives(args, alg,num_labels, num_classes,include_lb_to_ulb=False):

    class_folder = 'olives'
    if args.num_classes == 5:
        class_folder = 'olives_5'
    if args.autodl:
        csv_dir = f'/root/Semi-supervised-learning/data/{class_folder}/'
        data_dir = f"/root/autodl-tmp/OLIVES"
    else:
        csv_dir = f'/home/gu721/yzc/Semi-supervised-learning/data/{class_folder}/'
        data_dir = f"/home/gu721/yzc/data/ophthalmic_multimodal/OLIVES"
    if args.clinical:
        label_end = 2 + args.num_classes + 4
    else:
        label_end = 2 + args.num_classes

    train_all_info = pd.read_csv(f"{csv_dir}train_dataset.csv")
    train_all_info = train_all_info.fillna(0)

    data = train_all_info.iloc[:, 0].values
    # 为每一个data添加前缀
    data = [data_dir + i for i in data]
    targets = train_all_info.iloc[:, 2:label_end].values

    # mean = [0, 0, 0]
    # std = [1, 1, 1]
    # imgnet_mean = (0.485, 0.456, 0.406)
    # imgnet_std = (0.229, 0.224, 0.225)
    imgnet_mean = (0, 0, 0)
    imgnet_std = (1, 1, 1)
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_multilabel_data(args, data, targets, num_classes,
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                include_lb_to_ulb=include_lb_to_ulb,
                                                                           load_exist=False)





    # print("lb count: {}".format(np.sum(lb_targets, axis=0)))
    # print("ulb count: {}".format(np.sum(ulb_targets, axis=0)))

    #
    # if alg == 'fullysupervised':
    #     lb_data = data
    #     lb_targets = targets



    # 读取外部未标记数据
    exter_total_unlabel = pd.read_csv(f"{csv_dir}exter_total_unlabel.csv")
    exter_data = exter_total_unlabel.iloc[:, 0].values
    exter_data = [data_dir + i for i in exter_data]
    exter_clinical_targets = exter_total_unlabel.iloc[:,1:].values # bcva,cst,eye_id,patient_id
    # 由于没有标签，所以需要将标签设置为全为0
    if args.clinical:
        exter_targets = np.zeros((len(exter_data), num_classes))
        exter_targets = np.concatenate([exter_targets,exter_clinical_targets],axis=-1)
        # exter_targets[:][-4:] = exter_clinical_targets
    else:
        exter_targets = np.zeros((len(exter_data), num_classes))

    # 输出外部数据集的数量
    print("exter data count: {}".format(len(exter_data)))

    # 根据args.exterrio的值（float），将外部数据集的数据添加到ulb_data中
    if args.exterrio > 0:
        exter_num = int(len(exter_data) * args.exterrio)
        # 将 ulb_data 和 exter_data[:exter_num] 连接在一起
        ulb_data = np.concatenate((ulb_data, exter_data[:exter_num]), axis=0)
        ulb_targets = np.concatenate((ulb_targets, exter_targets[:exter_num]), axis=0)

        print("exter_num: {}".format(exter_num))

    # 输出 ulb_data 的数量
    print("ulb data count: {}".format(len(ulb_data)))

    lb_dset = OLIVESDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong,transform_strong, False,clinical=args.clinical)
    ulb_dset = OLIVESDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong,transform_strong, False,clinical=args.clinical)

    val_all_info = pd.read_csv(f"{csv_dir}val_dataset.csv")
    val_all_info = val_all_info.fillna(0)
    val_data = val_all_info.iloc[:, 0].values
    val_data = [data_dir + i for i in val_data]
    val_targets = val_all_info.iloc[:, 2:args.num_classes + 2].values
    eval_dset = OLIVESDataset(alg, val_data, val_targets, num_classes, transform_val, False, None, False,is_test=True)

    test_all_info = pd.read_csv(f"{csv_dir}test_dataset.csv")
    # test_all_info = pd.read_csv(f"{csv_dir}train_dataset.csv")
    test_all_info = test_all_info.fillna(0)
    test_data = test_all_info.iloc[:, 0].values
    test_data = [data_dir + i for i in test_data]
    test_targets = test_all_info.iloc[:, 2:args.num_classes + 2].values
    test_dset = OLIVESDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False,is_test=True)
    print("lb: {}, ulb: {}, eval: {}, test: {}".format(len(lb_dset), len(ulb_dset), len(eval_dset), len(test_dset)))
    return lb_dset, ulb_dset, eval_dset,test_dset

#
class OLIVESDataset(BasicDataset):
    def __sample__(self, idx):
        if self.clinical is not None:
            path = self.data[idx]
            img = Image.open(path).convert("RGB")
            all_target = self.targets[idx]
            biomarker = all_target[:-4]
            clinical = all_target[-4:]
            return img,biomarker,clinical,path

        else:
            path = self.data[idx]
            img = Image.open(path).convert("RGB")
            target = self.targets[idx]
            return img,target,path

if __name__ == '__main__':
    from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer

    config = {
        'algorithm': 'fixmatch',
        # 'algorithm': 'fullysupervised',
        'net': 'densenet121',
        'use_pretrain': False,  # todo: add pretrain
        # resnet50
        # 'pretrain_path': None,
        # 'net': 'vit_tiny_patch2_32',
        # 'use_pretrain': True,
        # 'pretrain_path': 'pretrained/vit_tiny_patch2_32_mlp_im_1k_32.pth',

        'amp': False,

        # optimization configs
        'epoch': 60,
        'num_train_iter': 150,
        'num_eval_iter': 50,
        'optim': 'SGD',
        'lr': 0.03,
        'momentum': 0.9,
        'batch_size': 64,
        'eval_batch_size': 64,

        # dataset configs
        'dataset': 'olives',
        'exterrio': 1.0,
        'num_labels': 32,
        'num_classes': 16,
        'input_size': 32,
        'data_dir': './data',

        # algorithm specific configs
        'hard_label': True,
        'uratio': 3,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': 1,
        'world_size': 1,
        'distributed': False,
    }
    config = get_config(config)
    lb_dset, ulb_dset, eval_dset = get_olives(config, config.algorithm, config.num_labels, config.num_classes)
    print('len(lb_dset):', len(lb_dset), 'len(ulb_dset):', len(ulb_dset), 'len(eval_dset):', len(eval_dset))
