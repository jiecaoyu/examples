SPARSITY=0.90 # this number includes the weights in fully-connected layers
DATA_DIR=/data2/jiecaoyu/imagenet/imgs/

python main.py ${DATA_DIR} --arch resnet50 --pretrained                                --prune admm     --prune-ratio ${SPARSITY}\
	--lr 0.01 --admm-iter 15 --epochs 150 --lr-epochs 150
