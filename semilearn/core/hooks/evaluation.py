# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook
import pandas as pd

class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            # eval_dict = algorithm.evaluate('eval')
            eval_dict = algorithm.test('eval')
            algorithm.log_dict.update(eval_dict)

            # if algorithm.log_dict['eval/mAP'] > algorithm.best_eval_mAP:
            #     algorithm.best_eval_mAP = algorithm.log_dict['eval/mAP']
            if algorithm.log_dict['eval/SENS'] > algorithm.best_eval_OF1:
                algorithm.best_eval_OF1 = algorithm.log_dict['eval/SENS']
                algorithm.best_it = algorithm.it
                algorithm.best_epoch = algorithm.epoch
            else:
                # algorithm.best_eval_mAP_patience += 1
                if (algorithm.it > int(algorithm.num_train_iter * 0.6)):
                    algorithm.best_eval_mAP_patience += 1 * algorithm.num_eval_iter
                else:
                    algorithm.best_eval_mAP_patience = 0

            if (algorithm.best_eval_mAP_patience > int(algorithm.num_train_iter * 0.2)) and (algorithm.it > int(algorithm.num_train_iter * 0.6)):
            # if (algorithm.best_eval_mAP_patience > 30):
                print('Early stopping at iteration {}'.format(algorithm.it))
                algorithm.it = 1000000000

    
    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)
        print('===============Testing================')
        # results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it}
        results_dict = {'test/best_OF1': 0, 'eval/best_it': algorithm.best_it}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.test('test')
            results_dict['test/best_OF1'] = test_dict['test/F1']
            # algorithm.results_dict['test/logits_dict']是一个字典，key是名字，value是一个list，将其保存为一个csv文件
            logits_save_path = os.path.join(algorithm.save_dir,algorithm.save_name, 'test_logits.csv')
            logits_df = pd.DataFrame.from_dict(test_dict['test/logits_dict'], orient='index').reset_index()
            logits_df.columns = ['name'] + ['logit_' + str(i) for i in range(len(logits_df.columns) - 1)]  # 给列加上名称
            logits_df.to_csv(logits_save_path, index=False)
            # 存储为csv文件
            save_path = os.path.join(algorithm.save_dir,algorithm.save_name, 'test_results.csv')
            df = pd.DataFrame({ 'AUC': [test_dict['test/AUC']],
                                'SENS': [test_dict['test/SENS']],
                                'SPEC': [test_dict['test/SPEC']],
                                'ACC': [test_dict['test/ACC']],
                                'F1': [test_dict['test/F1']]})
            df.to_csv(save_path, index=False)

            # 模拟计算
            from compute_by_logits_isic2018 import get_target_names_labels
            from semilearn.lighting.compute_metircs import compute_metrics
            import torch
            import numpy as np
            target_names, target_labels = get_target_names_labels()
            # 读取预测的 logits
            test_logits_path = logits_save_path
            # pred_logits_df = pd.read_csv(test_logits_path)
            pred_logits_df = logits_df
            pred_names = pred_logits_df.iloc[:, 0].values
            pred_probs = pred_logits_df.iloc[:, 1:].values

            # 去除预测名称中的前缀
            prefix = "/home/gu721/yzc/data/ISIC2018/images/"
            pred_names = [name.replace(prefix, '') for name in pred_names]
            pred_names = [name.replace('.jpg', '') for name in pred_names]
            # 确保预测和标签的样本顺序一致
            name_to_index = {name: idx for idx, name in enumerate(target_names)}
            indices = [name_to_index[name] for name in pred_names]
            target_labels_ordered = target_labels[indices]

            # 对预测 logits 应用 softmax 并选取最大值作为预测类别
            pred_probs = torch.softmax(torch.tensor(pred_probs), dim=-1).numpy()  # (N,num_classes)
            target_labels_ordered = np.argmax(target_labels_ordered, axis=1)  # (N,)
            result = compute_metrics(target_labels_ordered, pred_probs, num_classes=7)
            df = pd.DataFrame({'AUC': [result['AUC']],
                               'SENS': [result['SENS']],
                               'SPEC': [result['SPEC']],
                               'ACC': [result['ACC']],
                               'F1': [result['F1']]})
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name, 'test_results-v2.csv')
            df.to_csv(save_path, index=False)



        algorithm.results_dict = results_dict
        