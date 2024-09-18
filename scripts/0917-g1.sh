python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3 --other 'SimCLR' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/SimCLR/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3 --other 'patient' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/patient/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3 --other 'eyeid' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/eyeid/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3 --other 'cst' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3 --other 'bcva' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/bcva/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3 --other 'cst-eyeid' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-eyeid/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3 --other 'cst-bcva' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3 --other 'bcva-eyeid' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/bcva-eyeid/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.5 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 120  --lr 1e-3 --other 'cst-bcva-eyeid' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-resnet50/cst-bcva-eyeid/ckpt_epoch_25.pth"

