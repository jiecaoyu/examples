python main.py /data2/jiecaoyu/imagenet/imgs/ --arch alexnet --pretrained --prune admm --lr 0.001 --epochs 375 --lr-epochs 400
python main.py /data2/jiecaoyu/imagenet/imgs/ --arch alexnet --resume saved_models/admm.pth.tar --prune retrain --lr 0.01 --start-epoch 30
