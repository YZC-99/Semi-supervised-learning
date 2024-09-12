python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 12000 --device 0 --num_labels_ratio 0.5

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 12000 --device 1 --num_labels_ratio 0.1


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 12000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0
python oct_ssl.py --algorithm 'mixmatch' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0
python oct_ssl.py --algorithm 'comatch' --num_train_iter 12000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0
python oct_ssl.py --algorithm 'flexmatch' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0

python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 30000 --device 1