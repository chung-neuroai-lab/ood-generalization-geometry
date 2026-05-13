import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, dataset='cifar'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.dataset = dataset

        if dataset == 'cifar':
            input_size = 32
            num_classes = 10
        elif dataset == 'tinyimagenet':
            input_size = 64
            num_classes = 200
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            dummy_out = self.features(dummy)
            flattened_dim = dummy_out.view(1, -1).size(1)

        self.classifier = nn.Linear(flattened_dim, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        return nn.Sequential(*layers)


def VGG11(dataset='cifar'):
    return VGG('VGG11', dataset)

def VGG13(dataset='cifar'):
    return VGG('VGG13', dataset)

def VGG16(dataset='cifar'):
    return VGG('VGG16', dataset)

def VGG19(dataset='cifar'):
    return VGG('VGG19', dataset)
