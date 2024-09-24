python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid' --other 'KL+KL+supcon'


python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'bcva' --other 'KL+KL+supcon'


python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'cst' --other 'KL+KL+supcon'


python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid-cst' --other 'KL+KL+supcon'


