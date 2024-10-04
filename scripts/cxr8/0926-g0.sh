python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 30000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.02 \
--net 'densenet121' --batch_size 64 --num_classes 14 --dataset 'cxr8' --loss 'bce' --save_dir 'cxr8-exp' \
--all_train_count 80232 --num_eval_iter 1630 --overfit --seed 42 --lr 0.0001



python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 30000 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.02 \
--net 'densenet121' --batch_size 64 --num_classes 14 --dataset 'cxr8' --loss 'bce' --save_dir 'cxr8-exp' \
--all_train_count 80232 --clinical 'patient_id' --other '双0.5' \
--num_eval_iter 1630 --overfit --seed 42 --lr 0.0001


python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 30000 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.02 \
--net 'densenet121' --batch_size 64 --num_classes 14 --dataset 'cxr8' --loss 'bce' --save_dir 'cxr8-exp' \
--all_train_count 80232 --clinical 'view_position' --other '全0.95' \
--num_eval_iter 1630 --overfit --seed 42 --lr 0.0001



python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 30000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.02 \
--net 'densenet121' --batch_size 64 --num_classes 14 --dataset 'cxr8' --loss 'bce' --save_dir 'cxr8-exp' \
--all_train_count 80232 --clinical 'view_position' --other '标签数优先采样-全0.95' \
--num_eval_iter 1630 --overfit --seed 42 --lr 0.0001






