 anonymous_submission_iclr2020

Requirements: `torch==1.2.0`, `torchvision==0.4.0`

train cifar100 on resnet with class-wise regularization losses

`python3 train.py --sgpu 0 --lr 0.1 --epoch 200 --model CIFAR_ResNet18 --name test_cifar --decay 1e-4 --dataset cifar100 --dataroot ~/data/ -cls`

train cifar100 on densenet with class-wise regularization losses

`python3 train.py --sgpu 0 --lr 0.1 --epoch 200 --model CIFAR_DenseNet121 --name test_cifar --decay 1e-4 --dataset cifar100 --dataroot ~/data/ -cls`

train fine-grained dataset on resnet with class-wise and sample-wise regularization losses

`python3 train.py --sgpu 0 --lr 0.1 --epoch 200 --model resnet18 --name test_cub200 --batch-size 32 --decay 1e-4 --dataset CUB200   --dataroot ~/data/ -cls -sam`

train fine-grained dataset on densenet with class-wise and sample-wise regularization losses

`python3 train.py --sgpu 0 --lr 0.1 --epoch 200 --model densenet121 --name test_cub200 --batch-size 32 --decay 1e-4 --dataset CUB200   --dataroot ~/data/ -cls -sam`

