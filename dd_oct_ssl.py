import sys
sys.path.append('../')
import logging
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config
from semilearn.core.utils import (
    TBLog,
    count_parameters,
    get_logger,
    get_net_builder,
    get_port,
    over_write_args_from_file,
    send_model_cuda,
)
from semilearn.algorithms import get_algorithm, name2alg
from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from semilearn.lighting.sampler import Memory_NoReplacement_Sampler
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
# 忽略警告
import warnings
warnings.filterwarnings("ignore")
import argparse
from seed import seed_everything
import torch
# all 7408
# 0.05: 370
parser = argparse.ArgumentParser(description="Semi-Supervised Learning (USB)")
parser.add_argument("--algorithm", type=str, default="fixmatch")
parser.add_argument("--save_name", type=str, default="")
parser.add_argument("--net", type=str, default="resnet50")
parser.add_argument("--use_pretrain", type=bool, default=True)

#  training
parser.add_argument("--epochs", type=int, default=1000000)
parser.add_argument("--amp", type=bool, default=True)
parser.add_argument("--num_eval_iter", type=int, default=117)
parser.add_argument("--num_train_iter", type=int, default=12000)
parser.add_argument("--save_dir", type=str, default='oct_exp')
parser.add_argument("--exterrio", type=float, default=0.0)
parser.add_argument("--dist_url", type=str, default="")

parser.add_argument("--gpu", type=int, default=1)




parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--optim", type=str, default='Adam')
parser.add_argument("--lr", type=float, default=0.0002)
# data configuration
parser.add_argument("--all_train_count", type=int, default=7408)
parser.add_argument("--num_labels_ratio", type=float, default=0.05)
parser.add_argument("--uratio", type=int, default=3)

parser.add_argument("--train_sampler", type=str, default="RandomSampler")



def main(args):
    seed_everything(args.seed)


    if args.algorithm == 'fullysupervised':
        args.save_name = f'{args.net}/{args.algorithm}_{args.optim}_{args.other}_lr{args.lr}_num_train_iter{args.num_train_iter}_bs{args.batch_size}_seed{args.seed}'
    else:
        args.save_name = f'{args.net}/{args.algorithm}_{args.optim}_extr{args.exterrio}_uratio{args.uratio}_nlratio{args.num_labels_ratio}_{args.other}_lr{args.lr}_num_train_iter{num_train_iter}_bs{batch_size}_seed{args.seed}'

    if args.gpu == "None":
        args.gpu = None
    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    args.world_size = ngpus_per_node * args.world_size
    # args=(,) means the arguments of main_worker
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    """
       main_worker is conducted on each GPU.
       """

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the synchronization of labeled data sampling in each
    # process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, "tensorboard", use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")

    _net_builder = get_net_builder(args.net, args.net_from_name)
    # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    if args.imb_algorithm is not None:
        model = get_imb_algorithm(args, _net_builder, tb_log, logger)
    else:
        model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f"Number of Trainable Params: {count_parameters(model.model)}")

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    if hasattr(model, "warmup"):
        logger.info(("Warmup stage"))
        model.warmup()

    # START TRAINING of FixMatch
    logger.info("Model training")
    model.train()

    # print validation (and test results)
    for key, item in model.results_dict.items():
        logger.info(f"Model result - {key} : {item}")

    if hasattr(model, "finetune"):
        logger.info("Finetune stage")
        model.finetune()

    logging.warning(f"GPU {args.rank} training is FINISHED")
if __name__ == '__main__':
    args = parser.parse_args()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)

