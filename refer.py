import sys
sys.path.append('../')
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from semilearn.lighting.sampler import Memory_NoReplacement_Sampler
import os
import torch
import pandas as pd
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
# 忽略警告
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import argparse
from seed import seed_everything
from semilearn.datasets.cv_datasets.olives import OLIVESDataset
# all 7408
# 0.05: 370
parser = argparse.ArgumentParser(description="Semi-Supervised Learning (USB)")
parser.add_argument("--algorithm", type=str, default="fixmatch")
parser.add_argument("--save_name", type=str, default="/dk1/oct_exp/vit_small_patch16_224_patient/LP_fixmatch_SGD_extr1.0_uratio3_nlratio0.05_lr0.03_num_train_iter20000_bs96_seed42/")
parser.add_argument("--net", type=str, default="vit_small_patch16_224")
parser.add_argument("--num_train_iter", type=int, default=12000)
parser.add_argument("--num_eval_iter", type=int, default=117)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--all_train_count", type=int, default=7408)
parser.add_argument("--num_labels_ratio", type=float, default=0.05)
parser.add_argument("--uratio", type=int, default=3)
parser.add_argument("--amp", type=bool, default=True)
parser.add_argument("--optim", type=str, default='Adam')
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--exterrio", type=float, default=0.0)
parser.add_argument("--clinical", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=1000000)
parser.add_argument("--autodl", default=False, action='store_true')

columns = [
    "Atrophy / thinning of retinal layers",
    "Disruption of EZ",
    "DRIL",
    "IR hemorrhages",
    "IR HRF",
    "Partially attached vitreous face",
    "Fully attached vitreous face",
    "Preretinal tissue/hemorrhage",
    "Vitreous debris",
    "VMT",
    "DRT/ME",
    "Fluid (IRF)",
    "Fluid (SRF)",
    "Disruption of RPE",
    "PED (serous)",
    "SHRM"
]

label_to_number = {
    "IR HRF": 1,
    "Fully attached vitreous face": 2,
    "Fluid (IRF)": 3,
    "DRT/ME": 4,
    "Partially attached vitreous face": 5,
    "Vitreous debris": 6,
    "Preretinal tissue/hemorrhage": 7,
    "Disruption of EZ": 8,
    "IR hemorrhages": 9,
    "Fluid (SRF)": 10,
    "Atrophy / thinning of retinal layers": 11,
    "SHRM": 12,
    "DRIL": 13,
    "PED (serous)": 14,
    "Disruption of RPE": 15,
    "VMT": 16
}
def visual_biomarker_bar_on_dataset_correct(csv_path, pred_path,save_path):
    dataset_name = csv_path.split("/")[-1].split(".")[0]
    dataset = pd.read_csv(csv_path)
    # 从第2列到第18列是标签
    labels = dataset.iloc[:, 2:18]


    pred = pd.read_csv(pred_path)
    pred = pred.iloc[:, 1:]
    pred = pred.applymap(lambda x: 1 if x > 0.5 else 0)
    pred = (labels == 1) & (pred == 1)
    pred = pred.applymap(lambda x: 1 if x else 0)
    # 统计每一列的总和
    labels_sum = labels.sum()
    pred_sum = pred.sum()

    # 转换标签为数字
    labels_sum.index = [label_to_number[x] for x in labels_sum.index]
    pred_sum.index = [label_to_number[x] for x in pred_sum.index]

    # 按数字标签排序
    labels_sum = labels_sum.sort_index()
    pred_sum = pred_sum.sort_index()

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    labels_sum.plot(kind='bar', color='blue', ax=ax, position=0, width=0.4, label='Labels')
    pred_sum.plot(kind='bar', color='orange', ax=ax, position=1, width=0.4, label='Predictions')

    plt.xticks(ticks=range(len(label_to_number)), labels=list(label_to_number.keys()), rotation=90)
    plt.title(f"Distribution of Biomarkers in {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/correct_{dataset_name}_comparison.png")
    plt.show()


def visual_biomarker_bar_on_dataset(csv_path, pred_path,save_path,exter=''):
    dataset_name = csv_path.split("/")[-1].split(".")[0]
    dataset = pd.read_csv(csv_path)
    # 从第2列到第18列是标签
    labels = dataset.iloc[:, 2:18]
    # 统计每一列的总和
    labels_sum = labels.sum()

    pred = pd.read_csv(pred_path)
    pred = pred.iloc[:, 1:]
    pred = pred.applymap(lambda x: 1 if x > 0.5 else 0)
    pred_sum = pred.sum()

    # 转换标签为数字
    labels_sum.index = [label_to_number[x] for x in labels_sum.index]
    pred_sum.index = [label_to_number[x] for x in pred_sum.index]

    # 按数字标签排序
    labels_sum = labels_sum.sort_index()
    pred_sum = pred_sum.sort_index()

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    labels_sum.plot(kind='bar', color='blue', ax=ax, position=0, width=0.4, label='Labels')
    pred_sum.plot(kind='bar', color='orange', ax=ax, position=1, width=0.4, label='Predictions')

    plt.xticks(ticks=range(len(label_to_number)), labels=list(label_to_number.keys()), rotation=90)
    plt.title(f"Distribution of Biomarkers in {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/{exter}{dataset_name}_comparison.png")
    plt.show()



def get_ds(config,exter):
    csv_dir = '/home/gu721/yzc/Semi-supervised-learning/data/olives/'
    data_dir = "/home/gu721/yzc/data/ophthalmic_multimodal/OLIVES"
    imgnet_mean = (0, 0, 0)
    imgnet_std = (1, 1, 1)
    transform_val = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])
    if exter:
        # 读取外部未标记数据
        exter_total_unlabel = pd.read_csv(f"{csv_dir}exter_total_unlabel.csv")
        exter_data = exter_total_unlabel.iloc[:, 0].values
        exter_data = [data_dir + i for i in exter_data]
        # 由于没有标签，所以需要将标签设置为全为0
        exter_targets = np.zeros((len(exter_data), config.num_classes))
        ds = OLIVESDataset('fullysupervised', exter_data, exter_targets, config.num_classes, transform_val, False, None,
                           False, is_test=True)
    else:
        test_all_info = pd.read_csv(f"{csv_dir}train_dataset.csv")
        test_all_info = test_all_info.fillna(0)
        test_data = test_all_info.iloc[:, 0].values
        test_data = [data_dir + i for i in test_data]
        test_targets = test_all_info.iloc[:, 2:18].values
        ds = OLIVESDataset('fullysupervised', test_data, test_targets, config.num_classes, transform_val, False, None, False,is_test=True)
    return ds

def refer_and_save_logits_bar(args,save_name,net,exter=False):
    num_labels = int(args.all_train_count * args.num_labels_ratio)
    ulb_num_labels = args.all_train_count - num_labels
    config = {
        'algorithm': 'fullysupervised',
        'save_name': save_name,
        'net': net,
        'use_pretrain': True,  # todo: add pretrain
        # training
        'epoch': args.epochs,
        'amp': args.amp,
        'num_eval_iter': args.num_eval_iter,
        'num_train_iter': args.num_train_iter,
        'save_dir': 'oct_exp',
        'exterrio':args.exterrio,
        # optimization configs
        'optim': args.optim,
        'lr': args.lr,
        'momentum': 0.9,
        'batch_size': args.batch_size,
        'eval_batch_size': 1024,
        # dataset configs
        'dataset': 'olives',
        'num_labels': num_labels,
        'ulb_num_labels': ulb_num_labels,
        'num_classes': 16,
        'img_size': 224,
        'data_dir': './data',
        'clinical': args.clinical,

        # algorithm specific configs
        'hard_label': True,
        'uratio': 3,
        'ulb_loss_ratio': 1.0,
        'loss': 'bce',

        # device configs
        'gpu': 0,
        'world_size': 1,
        'distributed': False,
        'autodl': args.autodl,
    }
    config = get_config(config)

    algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

    ds = get_ds(config,exter)


    test_loader = get_data_loader(config, ds, config.eval_batch_size,data_sampler=None,drop_last=False,num_workers=16)

    algorithm.loader_dict['test'] = test_loader
    best_model_path = os.path.join(save_name, 'model_best.pth')
    algorithm.model.load_state_dict(torch.load(best_model_path)['model'])
    test_dict = algorithm.test('test',return_logits=True,only_logits=True)

    save_path = os.path.join(save_name, 'exter1.0_train_logits.csv')
    df = pd.DataFrame.from_dict(test_dict['test/logits_dict'], orient='index').reset_index()
    # df.columns = ['name'] + ['logit_' + str(i) for i in range(len(df.columns) - 1)]  # 给列加上名称
    df.columns = ['name'] + columns  # 给列加上名称
    df.to_csv(save_path, index=False)
    # train_label_path = "/home/gu721/yzc/Semi-supervised-learning/data/olives/train_dataset.csv"
    train_label_path = "/home/gu721/yzc/Semi-supervised-learning/data/olives/train_dataset.csv"


    visual_biomarker_bar_on_dataset(train_label_path, save_path, save_name,exter='exter1.0')
    # visual_biomarker_bar_on_dataset_correct(train_label_path, save_path, save_name)


if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(args.seed)
    net='vit_small_patch16_224'
    exter = True

    # save_name_list = [
    #     "/dk1/oct-exp-v1/vit_small_patch16_224_/FT_fixmatch_AdamW_extr0.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
    #     "/dk1/oct-exp-v1/vit_small_patch16_224_SimCLR/FT_fixmatch_AdamW_extr0.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
    #     "/dk1/oct-exp-v1/vit_small_patch16_224_patient/FT_fixmatch_AdamW_extr0.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
    #     "/dk1/oct-exp-v1/vit_small_patch16_224_eye_id/FT_fixmatch_AdamW_extr0.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
    #     "/dk1/oct-exp-v1/vit_small_patch16_224_cst/FT_fixmatch_AdamW_extr0.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
    #     "/dk1/oct-exp-v1/vit_small_patch16_224_bcva/FT_fixmatch_AdamW_extr0.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
    # ]

    save_name_list = [
        "/dk1/oct-exp-v1/vit_small_patch16_224_/_fixmatch_AdamW_extr1.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
        "/dk1/oct-exp-v1/vit_small_patch16_224_SimCLR/FT_fixmatch_AdamW_extr1.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
        "/dk1/oct-exp-v1/vit_small_patch16_224_patient/FT_fixmatch_AdamW_extr1.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
        "/dk1/oct-exp-v1/vit_small_patch16_224_eye_id/FT_fixmatch_AdamW_extr1.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
        "/dk1/oct-exp-v1/vit_small_patch16_224_cst/FT_fixmatch_AdamW_extr1.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
        "/dk1/oct-exp-v1/vit_small_patch16_224_bcva/FT_fixmatch_AdamW_extr1.0_uratio3_nlratio0.05_lr8e-05_num_train_iter20000_bs48_seed42/",
    ]


    for save_name in save_name_list:
        refer_and_save_logits_bar(args,save_name,net,exter)

