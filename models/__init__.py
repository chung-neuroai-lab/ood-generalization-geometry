from .resnet import ResNet18, ResNet34, ResNet50
from .vgg import VGG11, VGG13, VGG16, VGG19
from .densenet import DenseNet121, DenseNet169, DenseNet201
from .mobilenet import MobileNet
from .efficientnet import EfficientNetB0

ARCH_NAMES = [
    'ResNet18', 'ResNet34', 'ResNet50',
    'VGG13', 'VGG19',
    'DenseNet121',
    'MobileNet',
    'EfficientNetB0',
]

def build_model(arch_name, dataset='cifar'):
    num_classes = 10 if dataset == 'cifar' else 200
    if arch_name == 'ResNet18':
        return ResNet18(dataset=dataset)
    elif arch_name == 'ResNet34':
        return ResNet34(dataset=dataset)
    elif arch_name == 'ResNet50':
        return ResNet50(dataset=dataset)
    elif arch_name in ('VGG', 'VGG19'):
        return VGG19(dataset=dataset)
    elif arch_name == 'VGG13':
        return VGG13(dataset=dataset)
    elif arch_name == 'VGG16':
        return VGG16(dataset=dataset)
    elif arch_name == 'VGG11':
        return VGG11(dataset=dataset)
    elif arch_name == 'DenseNet121':
        return DenseNet121(dataset=dataset)
    elif arch_name == 'MobileNet':
        return MobileNet(num_classes=num_classes)
    elif arch_name == 'EfficientNetB0':
        return EfficientNetB0(dataset=dataset)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Choose from {ARCH_NAMES}")
