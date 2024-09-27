python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010

python oct_ssl.py --algorithm 'flexmatch' --num_train_iter 20000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010

python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 20000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010

python oct_ssl.py --algorithm 'simmatch' --num_train_iter 20000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010

python oct_ssl.py --algorithm 'mixmatch' --num_train_iter 20000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010

python oct_ssl.py --algorithm 'meanteacher' --num_train_iter 20000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010


python oct_ssl.py --algorithm 'comatch' --num_train_iter 20000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010

#hyperfixmatch

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'localization' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'sex' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'age' --other 'KL+KL+supcon'
