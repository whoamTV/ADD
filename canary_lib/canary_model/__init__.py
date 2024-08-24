# 模型(ImageNet)
from .imagenet import model_list as ml_imagenet
# 模型(CIFAR-10)
from .cifar_10 import model_list as ml_cifar10

model_list = ml_imagenet + ml_cifar10
