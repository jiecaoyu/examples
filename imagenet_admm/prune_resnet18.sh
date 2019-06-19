python main.py /data2/jiecaoyu/imagenet/imgs/ --arch resnet18 --pretrained --prune admm --prune-ratio 0.85 --lr 0.01 --epochs 375 --lr-epochs 400
python main.py /data2/jiecaoyu/imagenet/imgs/ --arch resnet18 --resume saved_models/resnet18.admm.pth.tar --prune retrain  --prune-ratio 0.85 --lr 0.1 --start-epoch 30
