from audioop import bias
from re import S

import torch.nn.functional as F
import torch
import torch.nn as nn
from torchsummary import summary

import torchvision.models as models


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1) -> None:
        super(ResBlock, self).__init__()

        self.in_channel = 64

        self.conv_chunk = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        x = self.conv_chunk(x) + self.shortcut(x)
        x = nn.ReLU(inplace=True)(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes) -> None:
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.max_pooling = nn.MaxPool2d(3, 2, 1)

        self.conv2 = self.generate_conv_block(ResBlock, 64, 2, 1)
        self.conv3 = self.generate_conv_block(ResBlock, 128, 2, 2)
        self.conv4 = self.generate_conv_block(ResBlock, 256, 2, 2)
        self.conv5 = self.generate_conv_block(ResBlock, 512, 2, 2)

        self.ave_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

    def generate_conv_block(self, block, out_channels, num_blocks, stride):
        layers = []
        stride_arr = [stride] + [num_blocks - 1]

        for i in stride_arr:
            layers.append(block(self.in_channels, out_channels, i))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pooling(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ave_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    print(torch.__version__)
    input = torch.randn(2, 3, 256, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet(ResBlock, 100).to(device)
    print(net(input))

    summary(net, (3, 224, 224))

    resnet18 = models.resnet18()
    summary(resnet18, (3, 224, 224))