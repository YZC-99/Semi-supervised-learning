python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 30000 --device 0 --net resnet50
python oct_ssl.py --algorithm 'meanteacher' --num_train_iter 12000 --device 0 --num_labels_ratio 0.10 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'mixmatch' --num_train_iter 12000 --device 0 --num_labels_ratio 0.10 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'uda' --num_train_iter 12000 --device 0 --num_labels_ratio 0.10 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 12000 --device 0 --num_labels_ratio 0.10 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'comatch' --num_train_iter 12000 --device 0 --num_labels_ratio 0.10 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'flexmatch' --num_train_iter 12000 --device 0 --num_labels_ratio 0.10 --exterrio 0.0 --net resnet50


