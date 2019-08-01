import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from collections import OrderedDict
import time
import os


# Ordinal Regression
def relu_fn(x):
    return x * torch.sigmoid(x)

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, expansion=4, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
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


class multiScale_Bx(nn.Module):
    def __init__(self, model, num_classes=6, block=Bottleneck):
        super(multiScale_Bx, self).__init__()
        self.con_stem = model._conv_stem
        self.bn0 = model._bn0
        self.relu = relu_fn
        self.layer1 = nn.Sequential(*(model._blocks[0:-2]))
        self.layer2 = nn.Sequential(model._blocks[-2])
        self.layer3 = nn.Sequential(model._blocks[-1])
        self.layer4 = nn.Sequential(model._conv_head, model._bn1)
        n2 = self.layer4[-1].num_features
        self.n2 = n2
        
        
        self.layer4_adv = nn.Sequential(
                nn.Conv2d(n2, n2, kernel_size=3, padding=1),
                nn.BatchNorm2d(n2),
                nn.Conv2d(n2, n2, kernel_size=1),
                nn.ReLU(inplace=True)
                )
        
        self.conv5 = self._make_layer(block, n2, 128, stride=2, expansion=2)
        self.conv6 = self._make_layer(block, 256, 64, stride=2, expansion=2)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(n2+256+128, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, num_classes)
                )
        
    def _make_layer(self, block, inplanes, planes, stride=1, expansion=4):
        downsample = None
        if stride !=1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * expansion),
            )
                
        layers = []
        layers.append(block(inplanes, planes, expansion, stride, downsample))
        layers.append(block(planes * expansion, planes, expansion))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.con_stem(x)
        x = self.bn0(x)
        x = self.relu(x)
        x_l1 = self.layer1(x)
        x_l2 = self.layer2(x_l1)
        x_l3 = self.layer3(x_l2)
        x_l4 = self.relu(self.layer4(x_l3))
        
        x_c5 = self.conv5(x_l4)
        x_c6 = self.conv6(x_c5)
        
        # concat
        x_l4_adv = self.layer4_adv(x_l4)
        x_l4_adv_pool = self.pool(x_l4_adv).view(x.size(0), -1)
        
        x_c5_pool = self.pool(x_c5).view(x.size(0), -1)
        x_c6_pool = self.pool(x_c6).view(x.size(0), -1)
        x_all = torch.cat([x_l4_adv_pool, x_c5_pool, x_c6_pool], 1)
        # classify
        x_pred = self.fc(x_all)
        return x_pred

