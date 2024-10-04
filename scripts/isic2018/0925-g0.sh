#python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 9000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
#--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
#--all_train_count 7010 --clinical 'lesion_id' --other 'KL+KL+supcon' --num_eval_iter 110

python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 3400 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010  --num_eval_iter 3400


python oct_ssl.py --algorithm 'hyperplusfixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'sex' --other 'KL+KL+supcon' --num_eval_iter 110 \
--model_ckpt "/dk1/isic2018-exp/densenet121_/fullysupervised_nlratio0.05__Adam_lr0.001_num_train_iter3400_bs64_seed42/latest_model.pth"


python oct_ssl.py --algorithm 'hyperplusfixmatch' --num_train_iter 15000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'KL+KL+supcon' --num_eval_iter 110 \
--model_ckpt "/dk1/isic2018-exp/densenet121_/fullysupervised_nlratio0.05__Adam_lr0.001_num_train_iter3400_bs64_seed42/latest_model.pth"


python oct_ssl.py --algorithm 'hyperplusfixmatch' --num_train_iter 9000 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'KL+KL+supcon-sup-graph_s-overfit-woEMA' --num_eval_iter 200 \
--model_ckpt "/dk1/isic2018-exp/densenet121_/fullysupervised_nlratio0.05__Adam_lr0.001_num_train_iter3400_bs64_seed42/latest_model.pth"


python oct_ssl.py --algorithm 'hyperplusfixmatch' --num_train_iter 9000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'KL+KL+supcon-sup-graph_s-overfit' --num_eval_iter 500 \
--model_ckpt "/home/gu721/yzc/Semi-supervised-learning/ckpt/isic0.05-densnet/latest_model.pth" --overfit





python oct_ssl.py --algorithm 'hyperplusfixmatchv2' --num_train_iter 2000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'th0.9' --num_eval_iter 50 --overfit \
--model_ckpt "/home/gu721/yzc/Semi-supervised-learning/ckpt/isic0.05-densnet/latest_model.pth"


python oct_ssl.py --algorithm 'hyperplusfixmatchv2' --num_train_iter 2000 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'th0.9' --num_eval_iter 50 --overfit \
--model_ckpt "/home/gu721/yzc/Semi-supervised-learning/ckpt/isic0.05-densnet/latest_model.pth"


python oct_ssl.py --algorithm 'hyperplusfixmatchv2' --num_train_iter 8400 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'th0.9' --num_eval_iter 150 --overfit --seed 3407


python oct_ssl.py --algorithm 'hyperplusfixmatchv2' --num_train_iter 5500 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'th0.9' --num_eval_iter 150 --overfit --seed 3407 --lr 0.0001


python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 5500 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'FC和标记HGNN的ce+未标记的HGNN+supcon+gmm选择的ce+fix的0.95' \
--num_eval_iter 150 --overfit --seed 3407 --lr 0.0001



python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 5500 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other '计算HGNN特征的Supcon+lb锚点感知的probs调整-memory-且只要正样本' \
--num_eval_iter 150 --overfit --seed 3407 --lr 0.0001


python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 5500 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other '计算HGNN特征的Supcon+lb锚点感知的probs调整-全局平均值-mask视角-post0.95阈值-但排除0.95阈值neg排除' \
--num_eval_iter 150 --overfit --seed 3407 --lr 0.0001


python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 5500 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other '双0.95种子实验' \
--num_eval_iter 150 --overfit --seed 3407 --lr 0.0001


python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 5500 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.2 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'localization' --other '双0.95种子实验' \
--num_eval_iter 150 --overfit --seed 768 --lr 0.0001






python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 5500 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'lb分类头logits分别计算损失-ulb两个hyperkl散度-通过whole建模' \
--num_eval_iter 150 --overfit --seed 2300 --lr 0.0001



























