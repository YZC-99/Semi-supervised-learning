python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 20000 --device 1 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 64  --lr 1e-4




python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_small_patch16_224' --optim AdamW --batch_size 48  --lr 8e-5



python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 20000 --device 0 \
--net 'vit_large_patch16_224' --optim AdamW --batch_size 16  --lr 3e-5 --num_eval_iter 400


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 20000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 \
--net 'vit_large_patch16_224' --optim AdamW --batch_size 16  --lr 3e-5

