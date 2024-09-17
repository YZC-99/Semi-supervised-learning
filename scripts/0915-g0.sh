#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
#--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'eyeid' --finetune_mode 'FT' \
#--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/eyeid/ckpt_epoch_25.pth"
#
#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
#--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'cst-eyeid' --finetune_mode 'FT' \
#--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-eyeid/ckpt_epoch_25.pth"
#
#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
#--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'cst-bcva' --finetune_mode 'FT' \
#--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva/ckpt_epoch_25.pth"
#
## LP
#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
#--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'SimCLR' --finetune_mode 'LP' \
#--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/SimCLR/ckpt_epoch_25.pth" &
#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
#--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'cst' --finetune_mode 'LP' \
#--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst/ckpt_epoch_25.pth" &
#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
#--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'bcva' --finetune_mode 'LP' \
#--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/bcva/ckpt_epoch_25.pth"
#
#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
#--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'eyeid' --finetune_mode 'LP' \
#--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/eyeid/ckpt_epoch_25.pth" &
#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
#--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'cst-eyeid' --finetune_mode 'LP' \
#--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-eyeid/ckpt_epoch_25.pth" &
#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
#--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'cst-bcva' --finetune_mode 'LP' \
#--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'patient' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/patient/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'bcva-eyeid' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/bcva-eyeid/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'cst-bcva-eyeid' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva-eyeid/ckpt_epoch_25.pth"

# LP
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'patient' --finetune_mode 'LP' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/patient/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'bcva-eyeid' --finetune_mode 'LP' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/bcva-eyeid/ckpt_epoch_25.pth" &

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'resnet50' --optim SGD --batch_size 96  --lr 3e-2 --other 'cst-bcva-eyeid' --finetune_mode 'LP' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva-eyeid/ckpt_epoch_25.pth"