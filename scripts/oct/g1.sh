python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 30000 --device 1 --num_labels_mode 'N200' \
--net 'densenet121' --batch_size 64 --num_classes 5 --dataset 'olives_5' --loss 'bce' --save_dir 'oct_exp' \
--all_train_count 7408 --num_eval_iter 1600 --seed 42 --lr 0.0001

python oct_ssl.py --algorithm 'flexmatch' --num_train_iter 30000 --device 1 --num_labels_mode 'N200' \
--net 'densenet121' --batch_size 64 --num_classes 5 --dataset 'olives_5' --loss 'bce' --save_dir 'oct_exp' \
--all_train_count 7408 --num_eval_iter 1600 --seed 42 --lr 0.0001

python oct_ssl.py --algorithm 'comatch' --num_train_iter 30000 --device 1 --num_labels_mode 'N200' \
--net 'densenet121' --batch_size 64 --num_classes 5 --dataset 'olives_5' --loss 'bce' --save_dir 'oct_exp' \
--all_train_count 7408 --num_eval_iter 1600 --seed 42 --lr 0.0001

python oct_ssl.py --algorithm 'simmatch' --num_train_iter 30000 --device 1 --num_labels_mode 'N200' \
--net 'densenet121' --batch_size 64 --num_classes 5 --dataset 'olives_5' --loss 'bce' --save_dir 'oct_exp' \
--all_train_count 7408 --num_eval_iter 1600 --seed 42 --lr 0.0001

python oct_ssl.py --algorithm 'mixmatch' --num_train_iter 30000 --device 1 --num_labels_mode 'N200' \
--net 'densenet121' --batch_size 64 --num_classes 5 --dataset 'olives_5' --loss 'bce' --save_dir 'oct_exp' \
--all_train_count 7408 --num_eval_iter 1600 --seed 42 --lr 0.0001