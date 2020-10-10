# https://github.com/NVlabs/AdaBatch/blob/master/models/cifar/resnet.py
'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import torch.nn as nn
import math
import torch


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, leaky_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, before_relu=False, intermediate=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if intermediate:
            return out if before_relu else self.relu(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out if before_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, leaky_relu=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, before_relu=False):
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
        return out if before_relu else self.relu(out)


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=10, leaky_relu=False):
        super(ResNet, self).__init__()
        self.leaky_relu = leaky_relu
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        self.n = n
        self.depth = depth

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)  # original 8
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # self.sm = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, leaky_relu=self.leaky_relu))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, leaky_relu=self.leaky_relu))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_features(self, x, layer, before_relu=False):
        layers_per_block = 2 * self.n

        x = self.conv1(x)
        x = self.bn1(x)

        if layer == 1:
            return x

        x = self.relu(x)

        if layer > 1 and layer <= 1 + layers_per_block:
            relative_layer = layer - 1 
            x = self.layer_block_forward(x, self.layer1, relative_layer, before_relu=before_relu)
            return x

        x = self.layer1(x)
        if layer > 1 + layers_per_block and layer <= 1 + 2*layers_per_block:
            relative_layer = layer - (1 + layers_per_block)
            x = self.layer_block_forward(x, self.layer2, relative_layer, before_relu=before_relu)
            return x
        
        x = self.layer2(x)
        if layer > 1 + 2*layers_per_block and layer <= 1 + 3*layers_per_block:
            relative_layer = layer - (1 + 2*layers_per_block)
            x = self.layer_block_forward(x, self.layer3, relative_layer, before_relu=before_relu)
            return x
        
        x = self.layer3(x)
        if layer == self.depth:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
        else:
            raise ValueError('layer {:d} is out of index!'.format(layer))
    
    def layer_block_forward(self, x, layer_block, relative_layer, before_relu=False):
        out = x
        if relative_layer == 1:
            return layer_block[0](out, before_relu, intermediate=True)

        if relative_layer == 2:
            return layer_block[0](out, before_relu, intermediate=False)
        
        out = layer_block[0](out)
        if relative_layer == 3:
            return layer_block[1](out, before_relu, intermediate=True)

        if relative_layer == 4:
            return layer_block[1](out, before_relu, intermediate=False)
        
        out = layer_block[1](out)
        if relative_layer == 5:
            return layer_block[2](out, before_relu, intermediate=True)

        if relative_layer == 6:
            return layer_block[2](out, before_relu, intermediate=False)

        raise ValueError('relative_layer is invalid')

    def infer(self, x):
        return self.forward(x).max(1, keepdim=True)[1]


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)