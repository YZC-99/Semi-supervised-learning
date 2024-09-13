
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 48  --lr 8e-5 \
--other 'patient' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct_exp/pretrain-vit_base_patch16_224_/patient/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 48  --lr 8e-5 \
--other 'eye_id' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct_exp/pretrain-vit_base_patch16_224_/eye_id/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 48  --lr 8e-5 \
--other 'bcva' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct_exp/pretrain-vit_base_patch16_224_/bcva/ckpt_epoch_25.pth"




python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'shallow_prompt_vit_small_patch16_224' --optim SGD --batch_size 48  --lr 3e-2 \
--other 'SimCLR' --finetune_mode 'VPT' \
--model_ckpt "/dk1/oct_exp/pretrain-vit_base_patch16_224_/SimCLR/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'deep_prompt_vit_small_patch16_224' --optim SGD --batch_size 48  --lr 3e-2 \
--other 'SimCLR' --finetune_mode 'VPT' \
--model_ckpt "/dk1/oct_exp/pretrain-vit_base_patch16_224_/SimCLR/ckpt_epoch_25.pth"










python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 48  --lr 8e-5 \
--other 'SimCLR' --finetune_mode 'LP' \
--model_ckpt "/dk1/oct_exp/pretrain-vit_base_patch16_224_/SimCLR/ckpt_epoch_25.pth"


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net 'vit_small_patch16_224' --optim SGD --batch_size 48  --lr 5e-3 \
--other 'SimCLR' --finetune_mode 'LP' \
--model_ckpt "/dk1/oct_exp/pretrain-vit_base_patch16_224_/SimCLR/ckpt_epoch_25.pth"






