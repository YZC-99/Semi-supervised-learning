# 刚好10G显存
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'cst-eye_id' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_small_patch16_224_/cst-eye_id/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'cst-bcva' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_small_patch16_224_/cst-bcva/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'bcva-eye_id' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_small_patch16_224_/bcva-eye_id/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'cst-bcva-eye_id' --finetune_mode 'FT' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_small_patch16_224_/cst-bcva-eye_id/ckpt_epoch_25.pth"

# 刚好10G显存
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'cst-eye_id' --finetune_mode 'LP' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_small_patch16_224_/cst-eye_id/ckpt_epoch_25.pth" &
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'cst-bcva' --finetune_mode 'LP' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_small_patch16_224_/cst-bcva/ckpt_epoch_25.pth"

python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'bcva-eye_id' --finetune_mode 'LP' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_small_patch16_224_/bcva-eye_id/ckpt_epoch_25.pth" &
sleep 5
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 96  --lr 1e-3 \
--other 'cst-bcva-eye_id' --finetune_mode 'LP' \
--model_ckpt "/dk1/oct-exp-v1/pretrain-vit_small_patch16_224_/cst-bcva-eye_id/ckpt_epoch_25.pth"

