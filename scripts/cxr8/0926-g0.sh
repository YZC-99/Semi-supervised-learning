python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 30000 --device 0 --num_labels_mode 'ratio' \
--num_labels_ratio 0.02 --net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 14 \
--dataset 'cxr8' --save_dir 'cxr8-exp' --all_train_count 78484 --num_eval_iter 500

python oct_ssl.py --algorithm 'flexmatch' --num_train_iter 30000 --device 1 --num_labels_mode 'ratio' \
--num_labels_ratio 0.02 --net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 14 \
--dataset 'cxr8' --save_dir 'cxr8-exp' --all_train_count 78484 --num_eval_iter 500


