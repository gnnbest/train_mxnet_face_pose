import sys
sys.path.append('..')
import math
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn, utils as gutils
import numpy as np
import time
import zipfile


class Hopenet(nn.Block):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()

        self.conv1 = nn.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding=(1,1),
                               use_bias=False, activation='relu')
        self.bn1 = nn.BatchNorm(in_channels=64)
        self.maxpool = nn.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding=(0,0))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2D((4,4))
        self.fc_yaw = nn.Dense(num_bins)
        self.fc_pitch = nn.Dense(num_bins)
        self.fc_roll = nn.Dense(num_bins)
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential()
            downsample.add(
                nn.Conv2D(planes * block.expansion,
                            kernel_size=(1,1), strides=(stride,stride), use_bias=False, activation='relu'),
                nn.BatchNorm(in_channels=planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        net = nn.Sequential()
        net.add(*layers)
        return net


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
#        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0],x.shape[1]))
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll

class Bottleneck(nn.Block):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(planes, kernel_size=(1,1), use_bias=False, activation='relu')
        self.bn1 = nn.BatchNorm(in_channels=planes)
        self.conv2 = nn.Conv2D(planes, kernel_size=(3,3), strides=(stride,stride),
                               padding=(1,1), use_bias=False, activation='relu')
        self.bn2 = nn.BatchNorm(in_channels=planes)
        self.conv3 = nn.Conv2D(planes * 4, kernel_size=(1,1), use_bias=False)
        self.bn3 = nn.BatchNorm(in_channels=planes * 4)
        #self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
#        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
#        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out =out+residual
#        out = self.relu(out)

        return out

class downsample(nn.Block):
    def __init__(self, planes_block, stride):
        super(downsample, self).__init__()
        
    
    def forward(self, F, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return out
