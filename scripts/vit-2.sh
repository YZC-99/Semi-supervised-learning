python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 20000 --device 1 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'SimCLR' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_base_patch16_224_/SimCLR/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'patient' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_base_patch16_224_/patient/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'eye_id' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_base_patch16_224_/eye_id/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'cst' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_base_patch16_224_/cst/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'bcva' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_base_patch16_224_/bcva/ckpt_epoch_25.pth"


