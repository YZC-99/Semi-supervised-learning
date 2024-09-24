python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'cst' --other 'cst'

python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'bcva' --other 'bcva'

python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid' --other 'eyeid'

python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'patientid' --other 'patientid'

python oct_ssl.py --algorithm 'clinfixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50_clinical' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'simclr' --other 'simclr'

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'cst-eye_id' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-eye_id/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'cst-bcva' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'bcva-eye_id' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/bcva-eye_id/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'cst-bcva-eye_id' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva-eye_id/ckpt_epoch_25.pth"

#LP
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'cst-eye_id' \
--finetune_mode 'LP' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-eye_id/ckpt_epoch_25.pth" &
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'cst-bcva' \
--finetune_mode 'LP' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva/ckpt_epoch_25.pth" &
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'bcva-eye_id' \
--finetune_mode 'LP' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/bcva-eye_id/ckpt_epoch_25.pth" &
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'cst-bcva-eye_id' \
--finetune_mode 'LP' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva-eye_id/ckpt_epoch_25.pth"