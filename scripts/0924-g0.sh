#python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
#--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' &
#python oct_ssl.py --algorithm 'flexmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
#--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
#python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
#--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
#上面的跑了
python oct_ssl.py --algorithm 'simmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
python oct_ssl.py --algorithm 'mixmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
python oct_ssl.py --algorithm 'meanteacher' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
python oct_ssl.py --algorithm 'comatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'bcva' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'cst' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid-cst' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid-bcva' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'bcva-cst' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 12000 --device 0 --num_labels_mode 'N1' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid-bcva-cst' --other 'KL+KL+supcon'

# N20
python oct_ssl.py --algorithm 'fixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' &
python oct_ssl.py --algorithm 'flexmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
python oct_ssl.py --algorithm 'fixdamatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
python oct_ssl.py --algorithm 'simmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
python oct_ssl.py --algorithm 'mixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
python oct_ssl.py --algorithm 'meanteacher' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'
python oct_ssl.py --algorithm 'comatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'bcva' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'cst' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid-cst' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid-bcva' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'bcva-cst' --other 'KL+KL+supcon'

python oct_ssl.py --algorithm 'hyperfixmatch' --num_train_iter 15000 --device 0 --num_labels_mode 'N20' --exterrio 1.0 \
--net 'resnet50' --optim SGD --batch_size 80  --lr 1e-2 --num_classes 5 --dataset 'olives_5' \
--clinical 'eyeid-bcva-cst' --other 'KL+KL+supcon'