from audioop import bias
from re import S

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2_1 = nn.MaxPool2d(3, 2)
        self.conv2_2 = self.generate_conv_block(ResBlock, 64, 2, 2)
        self.conv3 = self.generate_conv_block(ResBlock, 128, 2, 2)
        self.conv4 = self.generate_conv_block(ResBlock, 256, 2, 2)
        self.conv5 = self.generate_conv_block(ResBlock, 512, 2, 2)

    def generate_conv_block(self, block, out_channels, num_blocks, stride):
        layers = []
        stride_arr = [stride] + [num_blocks - 1]
        
        for i in stride_arr:
            layers.append(block(self.in_channels, out_channels, i))
            self.in_channels = out_channels  # auto update channel

        return nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = nn.ave_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = nn.Linear(512, self.num_classes)
        return x


if __name__ == '__main__':
    input = torch.randn(2,3, 256,256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet(ResBlock, 10).to(device)
    print(net(input))