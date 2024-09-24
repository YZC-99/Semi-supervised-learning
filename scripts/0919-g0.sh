

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5'

python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5'

# clinical
python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'cst' --other 'cst'

python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'bcva' --other 'bcva'

python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid' --other 'eyeid'

python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'patientid' --other 'patientid'

python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'simclr' --other 'simclr'
#FT
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.103 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 120  --lr 1e-2 --num_classes 5 --dataset 'olives_5' --other 'SimCLR' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/SimCLR/ckpt_epoch_25.pth"


