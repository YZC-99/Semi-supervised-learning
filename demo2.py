import sys
sys.path.append('../')
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer

config = {
    # 'algorithm': 'fixmatch',
    'algorithm': 'fullysupervised',
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
    'num_labels': 40,
    'num_classes': 16,
    'input_size': 32,
    'data_dir': './data',

    # algorithm specific configs
    'hard_label': True,
    'uratio': 3,
    'ulb_loss_ratio': 1.0,
    'loss': 'bce',

    # device configs
    'gpu': 1,
    'world_size': 1,
    'distributed': False,
}
config = get_config(config)

algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

# create dataset
dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels,
                           config.num_classes, data_dir=config.data_dir)
# create data loader for labeled training set
train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], config.batch_size)
# create data loader for unlabeled training set
train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], int(config.batch_size * config.uratio))
# create data loader for evaluation
eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size)

trainer = Trainer(config, algorithm)
trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)

trainer.evaluate(eval_loader)

y_pred, y_logits = trainer.predict(eval_loader)