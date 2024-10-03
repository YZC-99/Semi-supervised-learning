# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from progress.bar import Bar
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from semilearn.core.utils import get_optimizer, get_cosine_schedule_with_warmup, get_logger, EMA
from semilearn.lighting.evaluator import Evaluator
from semilearn.core.hooks import TimerHook, ParamUpdateHook,EMAHook,CheckpointHook


class Trainer:
    def __init__(self, config, algorithm, verbose=0):
        self.config = config
        self.verbose = verbose
        self.algorithm = algorithm


        # TODO: support distributed training?
        torch.cuda.set_device(config.gpu)
        self.algorithm.model = self.algorithm.model.cuda(config.gpu)

        # setup logger
        self.save_path = os.path.join(config.save_dir, config.save_name)
        self.logger = get_logger(config.save_name, save_path=self.save_path, level="INFO")

    def fit(self):

        self.evaluator = Evaluator(self.config.num_classes)
        self.algorithm.model.train()

        # EMA Init
        self.algorithm.ema = EMA(self.algorithm.model, self.algorithm.ema_m)
        self.algorithm.ema.register()

        # train
        self.algorithm.it = 0
        self.algorithm.best_eval_acc = 0.0
        self.algorithm.best_eval_mAP = 0.0
        self.algorithm.best_eval_mAP_patience = 0
        self.algorithm.best_epoch = 0
        self.algorithm.num_eval_iter = self.config.num_eval_iter

        self.algorithm.train()

        if self.config.loss == 'ce':
            self.logger.info(
                "Best acc {:.4f} at epoch {:d}".format(self.algorithm.best_eval_acc, self.algorithm.best_it))
        elif self.config.loss == 'bce':
            self.logger.info(
                "Best mAP {:.4f} at epoch {:d}".format(self.algorithm.best_eval_mAP, self.algorithm.best_it))

        self.logger.info("Training finished.")

    def evaluate(self, data_loader, use_ema_model=False,epoch=None):

        if self.config.loss == 'ce':
            y_pred, y_logits, y_true = self.predict(data_loader, use_ema_model, return_gt=True)
            top1 = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
            self.logger.info("confusion matrix")
            self.logger.info(cf_mat)
            result_dict = {'acc': top1, 'precision': precision, 'recall': recall, 'f1': f1}
            self.logger.info("evaluation metric")
            for key, item in result_dict.items():
                self.logger.info("{:s}: {:.4f}".format(key, item))
                self.algorithm.tb_log.add_scalar(key, item, epoch)
            return result_dict
        elif self.config.loss == 'bce':  # 多标签分类
            y_pred, y_logits, y_true = self.predict(data_loader, use_ema_model, return_gt=True)
            result_dict = self.evaluator.compute(y_logits, y_true)
            self.logger.info("evaluation metric")
            for key, item in result_dict.items():
                if key == 'AP':
                    for i, ap in enumerate(item):
                        self.algorithm.tb_log.add_scalar("AP/{:d}".format(i), ap, epoch)
                else:
                    self.logger.info(
                        "{:s}: {:.4f}".format(key, item) if isinstance(item, float) else "{:s}: {}".format(key, item))
                    self.algorithm.tb_log.add_scalar('val/' + key, item, epoch)
            self.algorithm.tb_log.add_scalar('train/lr', self.algorithm.optimizer.param_groups[0]['lr'], epoch)
            return result_dict

    # def predict(self, data_loader, use_ema_model=False, return_gt=False):
    #     # self.algorithm.model = torch.load(os.path.join(self.save_path, 'model_best.pth'))
    #     if self.config.loss == 'ce':
    #         self.algorithm.model.eval()
    #         if use_ema_model:
    #             self.algorithm.ema.apply_shadow()
    #
    #         y_true = []
    #         y_pred = []
    #         y_logits = []
    #         with torch.no_grad():
    #
    #             for data in data_loader:
    #                 x = data['x_lb']
    #                 y = data['y_lb']
    #
    #                 if isinstance(x, dict):
    #                     x = {k: v.cuda(self.config.gpu) for k, v in x.items()}
    #                 else:
    #                     x = x.cuda(self.config.gpu)
    #                 y = y.cuda(self.config.gpu)
    #
    #                 logits = self.algorithm.model(x)['logits']
    #
    #                 y_true.extend(y.cpu().tolist())
    #                 y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
    #                 y_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
    #         y_true = np.array(y_true)
    #         y_pred = np.array(y_pred)
    #         y_logits = np.concatenate(y_logits)
    #
    #         if use_ema_model:
    #             self.algorithm.ema.restore()
    #         self.algorithm.model.train()
    #
    #         if return_gt:
    #             return y_pred, y_logits, y_true
    #         else:
    #             return y_pred, y_logits
    #     elif self.config.loss == 'bce':
    #         self.algorithm.model.eval()
    #         if use_ema_model:
    #             self.algorithm.ema.apply_shadow()
    #
    #         y_true = []
    #         y_pred = []
    #         y_logits = []
    #         with torch.no_grad():
    #             for data in data_loader:
    #                 x = data['x_lb']
    #                 y = data['y_lb']
    #                 if isinstance(x, dict):
    #                     x = {k: v.cuda(self.config.gpu) for k, v in x.items()}
    #                 else:
    #                     x = x.cuda(self.config.gpu)
    #                 y = y.cuda(self.config.gpu)
    #
    #                 logits = self.algorithm.model(x)['logits']
    #                 y_logits.append(torch.sigmoid(logits).float())
    #                 y_true.append(y)
    #
    #         if use_ema_model:
    #             self.algorithm.ema.restore()
    #         self.algorithm.model.train()
    #
    #         if return_gt:
    #             return y_pred, y_logits, y_true
    #         else:
    #             return y_pred, y_logits
