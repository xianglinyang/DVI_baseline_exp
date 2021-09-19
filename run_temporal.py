import os

os.system("python temporal_exp.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_cifar10 --dataset cifar10 --dim 512 -s 1 -e 7 -p 1 --train_l 50000 --test_l 10000")

os.system("python temporal_exp.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_mnist --dataset mnist --dim 512 -s 1 -e 7 -p 1 --train_l 60000 --test_l 10000")

os.system("python temporal_exp.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_fmnist --dataset fmnist --dim 512 -s 1 -e 7 -p 1 --train_l 60000 --test_l 10000")

# os.system("python temporal_exp.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet50_cifar10 --dataset cifar10 --dim 2048 -s 1 -e 7 -p 1 --train_l 50000 --test_l 10000")
