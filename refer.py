import sys
sys.path.append('../')
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from semilearn.lighting.sampler import Memory_NoReplacement_Sampler
import os
import torch
import pandas as pd
# 忽略警告
import warnings
warnings.filterwarnings("ignore")
import argparse
from seed import seed_everything
# all 7408
# 0.05: 370
parser = argparse.ArgumentParser(description="Semi-Supervised Learning (USB)")
parser.add_argument("--algorithm", type=str, default="fixmatch")
parser.add_argument("--save_name", type=str, default="/dk1/oct_exp/resnet50/fixdamatch_SGD_extr1.0_uratio3_nlratio0.05__lr0.01_num_train_iter100000_bs64_seed42/")
parser.add_argument("--net", type=str, default="resnet50")
parser.add_argument("--num_train_iter", type=int, default=12000)
parser.add_argument("--num_eval_iter", type=int, default=117)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=64)
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


if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(args.seed)
    algorithm=args.algorithm
    num_train_iter=args.num_train_iter
    num_eval_iter = args.num_eval_iter
    lr=args.lr
    epochs=args.epochs
    batch_size=args.batch_size
    gpu=args.device
    amp=args.amp
    optim=args.optim
    uratio=args.uratio
    exterrio=args.exterrio
    other=''

    num_labels = int(args.all_train_count * args.num_labels_ratio)
    ulb_num_labels = args.all_train_count - num_labels
    config = {
        'algorithm': 'fullysupervised',
        'save_name': args.save_name,
        'net': args.net,
        'use_pretrain': True,  # todo: add pretrain
        # training
        'epoch': epochs,
        'amp': amp,
        'num_eval_iter': num_eval_iter,
        'num_train_iter': num_train_iter,
        'save_dir': 'oct_exp',
        'exterrio':exterrio,
        # optimization configs
        'optim': optim,
        'lr': lr,
        'momentum': 0.9,
        'batch_size': batch_size,
        'eval_batch_size': 64,
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
        'gpu': gpu,
        'world_size': 1,
        'distributed': False,
    }
    config = get_config(config)

    algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

    # create dataset
    dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels,
                               config.num_classes, data_dir=config.data_dir)

    test_loader = get_data_loader(config, dataset_dict['test'], config.eval_batch_size,data_sampler=None,drop_last=False)

    algorithm.loader_dict['test'] = test_loader
    best_model_path = os.path.join(args.save_name, 'model_best.pth')
    algorithm.model.load_state_dict(torch.load(best_model_path)['model'])
    test_dict = algorithm.test('test',return_logits=True)
    print(test_dict['test/mAP'])
    print(test_dict['test/OF1'])
    print(test_dict['test/CF1'])

    # df = pd.DataFrame.from_dict(test_dict['test/logits_dict'], orient='index').reset_index()
    # df.columns = ['name'] + columns  # 给列加上名称
    # save_path = os.path.join("/home/gu721/yzc/Semi-supervised-learning/demo/analysis_dataset/selection-bais/", '{}.csv'.format(args.save_name.split('/')[-2]))
    # df.to_csv(save_path, index=False)
