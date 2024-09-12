import sys
sys.path.append('../')
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from semilearn.lighting.sampler import Memory_NoReplacement_Sampler
import torch
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
parser.add_argument("--finetune_mode", type=str, default="FT",help="FT or PL")
parser.add_argument("--model_ckpt", type=str, default=None)
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
parser.add_argument("--other", type=str, default='')


parser.add_argument("--epochs", type=int, default=1000000)


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

    if algorithm == 'fullysupervised':
        save_name = f'{args.net}_{other}/{args.finetune_mode}_{algorithm}_{optim}_lr{lr}_num_train_iter{num_train_iter}_bs{batch_size}_seed{args.seed}'
    else:
        save_name = f'{args.net}_{other}/{args.finetune_mode}_{algorithm}_{optim}_extr{exterrio}_uratio{uratio}_nlratio{args.num_labels_ratio}_lr{lr}_num_train_iter{num_train_iter}_bs{batch_size}_seed{args.seed}'
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
        'save_dir': 'oct_exp',
        'exterrio':exterrio,
        'clinical':args.clinical,
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

    # Create TensorBoard SummaryWriter
    tb_log = SummaryWriter(log_dir=os.path.join(config.save_dir,config.save_name))

    algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=tb_log, logger=None)

    # create dataset
    dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels,
                               config.num_classes, data_dir=config.data_dir)

    lb_loader_length = len(dataset_dict['train_lb']) // config.batch_size
    ulb_loader_length = len(dataset_dict['train_ulb']) // int(config.batch_size * config.uratio)
    # 这种分为俩种情况，一种是有标签数据多，一种是无标签数据多。
    #1、有标签loader的长度大于无标签loader的长度，那么共同训练得时候loader得结束是以短得为准，因此要对长得数据进行无放回采样
    #2、无标签loader的长度大于有标签loader的长度，那么共同训练得时候loader得结束是以短得为准，因此要对长得数据进行有放回采样
    if lb_loader_length > ulb_loader_length:
        # 对有标签数据进行无放回采样，无标记数据不需要采样
        lb_sampler = Memory_NoReplacement_Sampler(dataset_dict['train_lb'])
        ulb_sampler = None
    else:
        # 对无标签数据进行无放回采样，有标记数据不需要采样
        ulb_sampler = Memory_NoReplacement_Sampler(dataset_dict['train_ulb'])
        lb_sampler = None
    if config.algorithm == 'fullysupervised':
        lb_sampler = None
        ulb_sampler = None

    if config.algorithm == 'fullysupervised':
        lb_bs = config.batch_size
        ulb_bs = 1
    else:
        lb_bs = config.batch_size // (1 + config.uratio)
        ulb_bs = config.batch_size - lb_bs
        print('lb_bs:', lb_bs, 'ulb_bs:', ulb_bs)

    # create data loader for unlabeled training set
    train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], ulb_bs,
                                       data_sampler=ulb_sampler,num_workers=8,drop_last=True,shuffle= True if ulb_sampler is None else False)

    # create data loader for labeled training set
    train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], lb_bs,
                                      data_sampler=lb_sampler,num_workers=8, pin_memory=True,drop_last=True, shuffle=True if lb_sampler is None else False)

    # create data loader for evaluation
    eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size,data_sampler=None,drop_last=False)

    test_loader = get_data_loader(config, dataset_dict['test'], config.eval_batch_size,data_sampler=None,drop_last=False)

    print('len(train_lb_loader):', len(train_lb_loader), 'len(train_ulb_loader):', len(train_ulb_loader), 'len(eval_loader):', len(eval_loader), 'len(test_loader):', len(test_loader))

    if args.model_ckpt is not None:
        ckpt_dict = torch.load(args.model_ckpt)['model']
        # 由于是多卡训练，所以需要去掉module
        ckpt_dict = {k.replace('module.', ''): v for k, v in ckpt_dict.items()}
        ckpt_dict = {k.replace('encoder.', ''): v for k, v in ckpt_dict.items()}
        algorithm.model.load_state_dict(ckpt_dict,strict=False)
        print('load model from:', args.model_ckpt)
        if args.finetune_mode == 'LP': # 除了最后一层全连接层，其他层都冻结
            for name, param in algorithm.model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False


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