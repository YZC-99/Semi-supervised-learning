python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 12000 --device 1 --net resnet50
python oct_ssl.py --algorithm 'meanteacher' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'mixmatch' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'uda' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'comatch' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'flexmatch' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'simmatch' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 --net resnet50
python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 12000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 --net resnet50


python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 30000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 --net resnet50


python oct_ssl.py --algorithm 'fullysupervised' --num_train_iter 100000 --device 1 --lr 1e-4 --net 'vit_base_patch16_224' --optim AdamW --batch_size 128




python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 100000 --device 0 --num_labels_ratio 0.05 --exterrio 1.0 --net resnet50 --lr 1e-2 --optim SGD --batch_size 64



python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 100000 --device 1 --num_labels_ratio 0.05 --exterrio 1.0 --net resnet50 --lr 1e-2 --optim SGD --batch_size 64


python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 100000 --device 1 --num_labels_ratio 0.05 --exterrio 0.0 \
--net resnet50 --lr 1e-2 --optim SGD --batch_size 64 --other patient-sl \
--model_ckpt "/home/gu721/yzc/OLIVES_Biomarker/save/SupCon/Prime_TREX_DME_Fixed_models/patient_n_n_n_n_1_1_10_Prime_TREX_DME_Fixed_lr_resnet50_0.001_decay_0.0001_bsz_96_temp_0.07_trial_0__0/last.pth"

python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 100000 --device 0 --num_labels_ratio 0.05 --exterrio 0.0 \
--net resnet50 --lr 1e-2 --optim SGD --batch_size 64 --other patient-sl --finetune_mode LP \
--model_ckpt "/home/gu721/yzc/OLIVES_Biomarker/save/SupCon/Prime_TREX_DME_Fixed_models/patient_n_n_n_n_1_1_10_Prime_TREX_DME_Fixed_lr_resnet50_0.001_decay_0.0001_bsz_96_temp_0.07_trial_0__0/last.pth"

python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 100000 --device 0 --num_labels_ratio 0.05 --exterrio 1.0 \
--net resnet50 --lr 1e-2 --optim SGD --batch_size 64 --other patient-sl --finetune_mode LP \
--model_ckpt "/home/gu721/yzc/OLIVES_Biomarker/save/SupCon/Prime_TREX_DME_Fixed_models/patient_n_n_n_n_1_1_10_Prime_TREX_DME_Fixed_lr_resnet50_0.001_decay_0.0001_bsz_96_temp_0.07_trial_0__0/last.pth"


ps -ef | grep oct  | awk '{print $2}' | xargs kill -9