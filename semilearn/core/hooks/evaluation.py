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


            if algorithm.num_classes == 7:
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
            else:
                if algorithm.log_dict['eval/AUC'] > algorithm.best_eval_OF1:
                    algorithm.best_eval_OF1 = algorithm.log_dict['eval/AUC']
                    algorithm.best_it = algorithm.it
                    algorithm.best_epoch = algorithm.epoch
                else:
                    # algorithm.best_eval_mAP_patience += 1
                    if (algorithm.it > int(algorithm.num_train_iter * 0.6)):
                        algorithm.best_eval_mAP_patience += 1 * algorithm.num_eval_iter
                    else:
                        algorithm.best_eval_mAP_patience = 0



            if (algorithm.best_eval_mAP_patience > int(algorithm.num_train_iter * 0.1)) and (algorithm.it > int(algorithm.num_train_iter * 0.6)):
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

        algorithm.results_dict = results_dict
        