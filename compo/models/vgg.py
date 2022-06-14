# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn


__all__ = ["vgg"]


ARCHITECTURES = {
    "O": [4, "M", 8, "M", 16, 16, "M", 32, 32, "M", 32, 32, "M"],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, nn_arch, n_classes, use_bn=True):
        super(VGG, self).__init__()

        # init parameters.
        self.use_bn = use_bn
        self.nn_arch = nn_arch

        # init models.
        self.features = self._make_layers()
        self.intermediate_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
        )
        self.classifier = nn.Linear(512, n_classes)

        # weight initialization.
        self._weight_initialization()

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self):
        layers = []
        in_channels = 3
        for v in ARCHITECTURES[self.nn_arch]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.use_bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, feats_also=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        feat = self.intermediate_classifier(x)
        x = self.classifier(feat)

        if feats_also:
            return feat, x
        else:
            return x


class VGG_S(nn.Module):
    def __init__(self, nn_arch, n_classes, width=1, use_bn=True):
        super(VGG_S, self).__init__()

        # init parameters.
        self.use_bn = use_bn
        self.nn_arch = nn_arch
        self.width = width

        # init models.
        self.features = self._make_layers()
        self.classifier = nn.Linear(int(32 * width), n_classes)

        # weight initialization.
        self._weight_initialization()

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self):
        layers = []
        in_channels = 3
        for v in ARCHITECTURES[self.nn_arch]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_planes = int(v * self.width)
                conv2d = nn.Conv2d(in_channels, out_planes, kernel_size=3, padding=1)
                if self.use_bn:
                    layers += [conv2d, nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, feats_also=False):
        x = self.features(x)
        feat = x.view(x.size(0), -1)
        x = self.classifier(feat)

        if feats_also:
            return feat, x
        else:
            return x


def vgg(size, n_classes, vgg_scaling=None, use_bn=True):
    if size == 9:
        width = vgg_scaling if vgg_scaling is not None else 8
        return VGG_S(nn_arch="O", n_classes=n_classes, width=width, use_bn=use_bn)
    if size == 11:
        return VGG(nn_arch="A", n_classes=n_classes, use_bn=use_bn)
    elif size == 13:
        return VGG(nn_arch="B", n_classes=n_classes, use_bn=use_bn)
    elif size == 16:
        return VGG(nn_arch="D", n_classes=n_classes, use_bn=use_bn)
    elif size == 19:
        return VGG(nn_arch="E", n_classes=n_classes, use_bn=use_bn)
    else:
        exit(f'no such VGG-{size}')


if __name__ == "__main__":

    def get_n_model_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    width = 8
    net = VGG_S(nn_arch="O", n_classes=10, width=width, use_bn=True).to(0)
    # net = VGG(nn_arch="A", n_classes=10, use_bn=True).to(0)
    print(f"VGG with width={width} has n_params={get_n_model_params(net)}M.")

    x = torch.randn(1, 3, 32, 32).to(0)
    y = net(x)
    print(y.shape)
