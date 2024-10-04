import sys
sys.path.append('../')
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from semilearn.lighting.sampler import Memory_NoReplacement_Sampler
import torch

import shutil
import os
# 忽略警告
import warnings
warnings.filterwarnings("ignore")
import argparse
from seed import seed_everything
# all 7408
# 0.05: 370
parser = argparse.ArgumentParser(description="Semi-Supervised Learning (USB)")
parser.add_argument("--algorithm", type=str, default="fixmatch")
parser.add_argument("--net", type=str, default="resnet50")
parser.add_argument("--finetune_mode", type=str, default="",help="FT , PL, P1")
parser.add_argument("--model_ckpt", type=str, default=None)
parser.add_argument("--dataset", type=str, default='olives')
parser.add_argument("--save_dir", type=str, default='oct_exp')
parser.add_argument("--num_train_iter", type=int, default=12000)
parser.add_argument("--num_warmup_iter", type=float, default=0.0)
parser.add_argument("--num_eval_iter", type=int, default=117)
parser.add_argument("--num_classes", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--all_train_count", type=int, default=7408)
parser.add_argument("--num_labels_ratio", type=float, default=0.05)
parser.add_argument("--num_labels_mode", type=str, default='ratio',help='N1,N2,N3')
parser.add_argument("--uratio", type=int, default=3)
parser.add_argument("--amp", type=bool, default=True)
parser.add_argument("--optim", type=str, default='Adam')
parser.add_argument("--loss", type=str, default='bce')
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--exterrio", type=float, default=0.0)
parser.add_argument("--clinical", type=str, default=None,help='simclr,eyeid,bcva,cst,patientid')
parser.add_argument("--other", type=str, default='')
parser.add_argument("--autodl", action='store_true',default=False)
parser.add_argument("--epochs", type=int, default=1000000)

# llm finetune
parser.add_argument("--vpt_shallow", action='store_true',default=False)
parser.add_argument("--vpt_deep", action='store_true',default=False)
parser.add_argument("--vpt_last", action='store_true',default=False)
parser.add_argument("--overfit", action='store_true',default=False)

parser.add_argument("--vpt_len", type=int,default=50)




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
    other=args.other

    if args.num_labels_mode != 'ratio':
        args.num_labels_ratio = 0.0

    if algorithm == 'fullysupervised':
        save_name = f'{args.net}_{other}/{algorithm}_nlratio{args.num_labels_ratio}_{args.finetune_mode}_{optim}_lr{lr}_num_train_iter{num_train_iter}_bs{batch_size}_seed{args.seed}'
    else:
        save_name = f'{args.net}_{args.clinical}/{algorithm}_{other}_{args.finetune_mode}_{args.num_labels_mode}_{optim}_extr{exterrio}_uratio{uratio}_nlratio{args.num_labels_ratio}_lr{lr}_num_train_iter{num_train_iter}_bs{batch_size}_seed{args.seed}'
    num_labels = int(args.all_train_count * args.num_labels_ratio)
    ulb_num_labels = args.all_train_count - num_labels
    config = {
        'algorithm': algorithm,
        'save_name': save_name,
        'net': args.net,
        'use_pretrain': True,  # todo: add pretrain
        # training
        'epoch': epochs,
        'amp': amp,
        'num_eval_iter': num_eval_iter,
        'num_train_iter': num_train_iter,
        'num_warmup_iter': int(args.num_warmup_iter * args.num_train_iter),
        'save_dir': args.save_dir,
        'exterrio':exterrio,
        'clinical':args.clinical,
        # optimization configs
        'optim': optim,
        'lr': lr,
        'momentum': 0.9,
        'seed': args.seed,
        'batch_size': batch_size,
        'eval_batch_size': 256,
        # dataset configs
        'dataset': args.dataset,
        'num_labels': num_labels,
        'num_labels_mode': args.num_labels_mode,
        'ulb_num_labels': ulb_num_labels,
        'num_classes': args.num_classes,
        'img_size': 224,
        'data_dir': './data',

        # algorithm specific configs
        'hard_label': True,
        'uratio': uratio,
        'ulb_loss_ratio': 1.0,
        'loss': args.loss,

        # device configs
        'gpu': gpu,
        'world_size': 1,
        'distributed': False,
        'autodl': args.autodl,
    }
    config = get_config(config)

    # Create TensorBoard SummaryWriter
    tb_log = SummaryWriter(log_dir=os.path.join(config.save_dir,config.save_name))

    shutil.copytree('./semilearn/algorithms/hyperplusfixmatchv3',os.path.join(config.save_dir,config.save_name,'code'),
                    ignore=shutil.ignore_patterns('data','output','__pycache__','code','runs','.*'),
                    dirs_exist_ok=True
                    )


    algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=tb_log, logger=None)

    # create dataset
    dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels,
                               config.num_classes, data_dir=config.data_dir)



    ulb_sampler = None
    lb_sampler = None
    if config.algorithm == 'fullysupervised':
        lb_bs = config.batch_size
        ulb_bs = 1
    else:
        lb_bs = config.batch_size // (1 + config.uratio)
        ulb_bs = config.batch_size - lb_bs
        print('lb_bs:', lb_bs, 'ulb_bs:', ulb_bs)

    lb_loader_length = len(dataset_dict['train_lb']) // lb_bs
    ulb_loader_length = len(dataset_dict['train_ulb']) // ulb_bs

    # lb_repeat_times = len(dataset_dict['train_ulb']) // len(dataset_dict['train_lb'])
    lb_repeat_times = ulb_loader_length // lb_loader_length

    # create data loader for unlabeled training set
    train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], ulb_bs,
                                       data_sampler=ulb_sampler,num_workers=args.num_workers,drop_last=True,shuffle= True if ulb_sampler is None else False)

    # create data loader for labeled training set
    train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], lb_bs,
                                      data_sampler=lb_sampler,num_workers=args.num_workers, pin_memory=True,drop_last=True,
                                      shuffle=True,lb_repeat_times=lb_repeat_times)

    # create data loader for evaluation
    eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size,data_sampler=None,drop_last=False)
    if args.overfit:
        eval_loader = get_data_loader(config, dataset_dict['test'], config.eval_batch_size,data_sampler=None,drop_last=False)

    test_loader = get_data_loader(config, dataset_dict['test'], config.eval_batch_size,data_sampler=None,drop_last=False)

    print('len(train_lb_loader):', len(train_lb_loader), 'len(train_ulb_loader):', len(train_ulb_loader), 'len(eval_loader):', len(eval_loader), 'len(test_loader):', len(test_loader))

    if args.model_ckpt is not None:
        ckpt_dict = torch.load(args.model_ckpt,map_location='cpu')['model']
        ckpt_dict = {k: v for k, v in ckpt_dict.items() if 'head' not in k}
        if args.algorithm == 'hyperplusfixmatchv2':
        # 由于现有的模型在所有的参数的前面都多了一个backbone.的前缀，所以权重加载的时候也要加上这个前缀
            ckpt_dict = {f'backbone.{k}': v for k, v in ckpt_dict.items()}
        # algorithm.model.load_state_dict(ckpt_dict)
        print('load model from:', args.model_ckpt)
        # exclude_list = ['classifier', 'head','deep_prompt_embeddings','prompt_embeddings']


    trainer = Trainer(config, algorithm)
    algorithm.loader_dict['train_lb'] = train_lb_loader
    algorithm.loader_dict['train_ulb'] = train_ulb_loader
    algorithm.loader_dict['eval'] = eval_loader
    algorithm.loader_dict['test'] = test_loader
    trainer.fit()
    # algorithm.model = algorithm.model.cuda()
    # algorithm.train()
    # trainer.evaluate(test_loader)
    # y_pred, y_logits = trainer.predict(test_loader)