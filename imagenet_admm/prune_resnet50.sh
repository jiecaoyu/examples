SPARSITY=0.75 # this number includes the weights in fully-connected layers
DATA_DIR=/data2/jiecaoyu/imagenet/imgs/

python main.py ${DATA_DIR} --arch resnet50 --pretrained                                --prune admm     --prune-ratio ${SPARSITY}\
	--lr 0.01 --admm-iter 15 --epochs 450 --lr-epochs 450
python main.py ${DATA_DIR} --arch resnet50 --resume saved_models/resnet50.admm.pth.tar --prune retrain  --prune-ratio ${SPARSITY}\
	--lr 0.1  --start-epoch 30
