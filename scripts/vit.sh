

python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 100000 --device 1 --lr 1e-4 \
--net 'vit_base_patch16_224' --optim AdamW --batch_size 64




ps -ef | grep oct  | awk '{print $2}' | xargs kill -9