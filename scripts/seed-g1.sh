#!/bin/bash

for ((i=0; i<10; i++))
do
  seed=$((5 + i * 109))
  python oct_ssl.py --algorithm 'hyperplusfixmatchv3' --num_train_iter 5500 --device 1 \
  --num_labels_mode 'ratio' --num_labels_ratio 0.2 --net 'densenet121' --batch_size 64 \
  --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
  --all_train_count 7010 --clinical 'lesion_id' --other '双0.95种子实验' \
  --num_eval_iter 150 --overfit --seed $seed --lr 0.0001
done
