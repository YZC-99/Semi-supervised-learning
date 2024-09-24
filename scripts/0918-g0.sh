python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'patient' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/patient/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'eye_id' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/eye_id/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'cst' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'bcva' \
--finetune_mode 'FT' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/bcva/ckpt_epoch_25.pth"

# LP
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'SimCLR' \
--finetune_mode 'LP' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/SimCLR/ckpt_epoch_25.pth" &
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'patient' \
--finetune_mode 'LP' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/patient/ckpt_epoch_25.pth" &
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'eye_id' \
--finetune_mode 'LP' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/eye_id/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'cst' \
--finetune_mode 'LP' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst/ckpt_epoch_25.pth" &
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 128  --lr 3e-2 --num_classes 5 --dataset 'olives_5' --other 'bcva' \
--finetune_mode 'LP' --model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/bcva/ckpt_epoch_25.pth"