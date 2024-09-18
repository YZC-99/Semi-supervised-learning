python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 20000 --device 0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5'

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5'
