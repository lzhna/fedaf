import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)

class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1
    
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, group_norm_num_groups=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, group_norm_num_groups=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.conv3 = nn.Conv2d(out_planes, out_channels=out_planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = norm2d(group_norm_num_groups, planes=out_planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBase(nn.Module):
    def _make_block(self, block_fn, planes, n_blocks, stride=1, group_norm_num_groups=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_fn.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm2d(group_norm_num_groups, planes=planes * block_fn.expansion),
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
            )
        )
        self.inplanes = planes * block_fn.expansion
        for _ in range(1, n_blocks):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                )
            )

        return nn.Sequential(*layers)
    
    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def train(self, mode=True):
        super(ResNetBase, self).train(mode)

    def forward(self, x, feats_also=False):
        pstem = self.conv1(x) # pstem: pre stem before activation
        pstem = self.bn1(pstem)
        stem  = self.relu(pstem) # 32x32
        if self.layer4 is not None:
            stem = self.maxpool(stem)
        stem  = (pstem, stem)

        rb1 = self.layer1(stem[1])  # 32x32
        rb2 = self.layer2(rb1[1])  # 16x16
        rb3 = self.layer3(rb2[1])  # 8x8
        rbs = [rb1, rb2, rb3]

        if self.layer4 is not None:
            rb4 = self.layer4(rb3[1])
            rbs.append(rb4)
        
        feat = self.avgpool(rbs[-1][1])
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)

        if feats_also:
            return feat, out
        else:
            return out


class ResNet_imagenet(ResNetBase):
    def __init__(self, resnet_size, n_classes, group_norm_num_groups=None):
        super(ResNet_imagenet, self).__init__()

        # define model param.
        model_params = {
            18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
        }
        block_fn = model_params[resnet_size]["block"]
        block_nums = model_params[resnet_size]["layers"]

        # define layers.
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=64,
            n_blocks=block_nums[0],
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=128,
            n_blocks=block_nums[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=256,
            n_blocks=block_nums[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer4 = self._make_block(
            block_fn=block_fn,
            planes=512,
            n_blocks=block_nums[3],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(
            in_features=512 * block_fn.expansion, out_features=n_classes
        )

        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward(self, x, feats_also=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        x = self.classifier(feat)

        if feats_also:
            return feat, x
        else:
            return x


class ResNet_cifar(ResNetBase):
    def __init__(self, resnet_size, n_classes, scaling=1, group_norm_num_groups=None):
        super(ResNet_cifar, self).__init__()

        # define model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        n_blocks = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

        # define layers.
        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            n_blocks=n_blocks,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            n_blocks=n_blocks,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling),
            n_blocks=n_blocks,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=int(64 * scaling * block_fn.expansion),
            out_features=n_classes,
        )

        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward(self, x, feats_also=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        x = self.classifier(feat)

        if feats_also:
            return feat, x
        else:
            return x

def resnet(resnet_size=8, n_classes=10, group_norm_num_groups=None, mode='cifar'):
    if mode == 'cifar':
        model = ResNet_cifar(resnet_size=resnet_size,
                            n_classes=n_classes,
                            group_norm_num_groups=group_norm_num_groups)
    elif mode == 'imagenet_downsampled':
        model = ResNet_cifar(resnet_size=resnet_size,
                            n_classes=n_classes,
                            scaling=4,
                            group_norm_num_groups=group_norm_num_groups)
    elif mode == 'imagenet':
        model = ResNet_imagenet(resnet_size=resnet_size,
                            n_classes=n_classes,
                            group_norm_num_groups=group_norm_num_groups)
    else:
        exit('no such mode f{mode}')

    return model


if __name__ == '__main__':
    def get_n_model_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    resnet_size=20
    net = resnet(resnet_size=resnet_size,
                n_classes=10,
                group_norm_num_groups=None,
                mode='cifar')
    print(f"resnet-{resnet_size} has n_params={get_n_model_params(net)}M.")

    # data = torch.randn(5,3,32,32)
    # pred = net(data)
    # print(pred)