python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 20000 --device 0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5'

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5'


python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5'


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'SimCLR' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/SimCLR/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5'


python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --clinical 'cst'



python oct_ssl.py --algorithm 'conmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 64  --lr 1e-2 --num_classes 5 --dataset 'olives_5'

python oct_ssl.py --algorithm 'comatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'



python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 6000 --device 0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --num_labels_ratio 0.103 --num_eval_iter 6


python oct_ssl.py --algorithm 'hypercomatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' --clinical 'eyeid' --other 'eyeid'

python oct_ssl.py --algorithm 'hypercomatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' --clinical 'bcva' --other 'bcva'


python oct_ssl.py --algorithm 'comatch_wo_memory' --num_train_iter 20000 --device 1 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5'

python oct_ssl.py --algorithm 'comatch_wo_graph' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5'


python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' --clinical 'bcva' --other 'bcva'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' --clinical 'bcva' --other 'KL+KL'


python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'cst' --other 'KL+KL+supcon'


python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid-cst' --other 'KL+KL+supcon'






