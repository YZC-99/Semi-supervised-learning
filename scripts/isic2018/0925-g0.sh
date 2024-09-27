#python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 9000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
#--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
#--all_train_count 7010 --clinical 'lesion_id' --other 'KL+KL+supcon' --num_eval_iter 110

python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 3400 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010  --num_eval_iter 3400


python oct_ssl.py --algorithm 'hyperplusfixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'sex' --other 'KL+KL+supcon' --num_eval_iter 110 \
--model_ckpt "/dk1/isic2018-exp/densenet121_/fullysupervised_nlratio0.05__Adam_lr0.001_num_train_iter3400_bs64_seed42/latest_model.pth"


python oct_ssl.py --algorithm 'hyperplusfixmatch' --num_train_iter 15000 --device 1 --num_labels_mode 'ratio' --num_labels_ratio 0.05 \
--net 'densenet121' --batch_size 64 --num_classes 7 --dataset 'isic2018' --loss 'ce' --save_dir 'isic2018-exp' \
--all_train_count 7010 --clinical 'lesion_id' --other 'KL+KL+supcon' --num_eval_iter 110 \
--model_ckpt "/dk1/isic2018-exp/densenet121_/fullysupervised_nlratio0.05__Adam_lr0.001_num_train_iter3400_bs64_seed42/latest_model.pth"




