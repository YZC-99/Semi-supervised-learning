python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2


python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 128  --lr 3e-2 --clinical 'cst' --other 'cst'



python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 120  --lr 3e-2 --clinical 'bcva' --other 'bcva'


python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 0.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 120  --lr 3e-2 --clinical 'simclr' --other 'simclr'



python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --num_labels_mode N1 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2


python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2