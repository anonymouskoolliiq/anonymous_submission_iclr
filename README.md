# anonymous_submission_iclr

train resnet with class-wise, sample-wise losses

`python3 train.py --sgpu 0 --lr 0.1 --epoch 200 --model resnet18 --name test_resnet --batch-size 32 --decay 1e-4 --dataset CUB200   --dataroot ~/data/ -cls -sam`
