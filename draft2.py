import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, channel, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        out_channel = int(channel * (base_width /64.)) * 1  # groups

        self.chunk = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers_arr, num_classes=1000):
        super(ResNet, self).__init__()

        self.in_channel = 64
        self.base_width = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True)
        )

        self.max_pool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers_arr[0])
        self.layer2 = self._make_layer(block, 128, layers_arr[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers_arr[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers_arr[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channel, num_blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * block.expansion)
            )


        layers = [block(self.in_channel, out_channel, stride, downsample, self.base_width)]

        self.in_channel = out_channel * block.expansion  # update channel num

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, out_channel, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
"""
AA torch.Size([2, 64, 56, 56])
down torch.Size([2, 128, 28, 28])
AA torch.Size([2, 128, 28, 28])
down torch.Size([2, 256, 14, 14])
AA torch.Size([2, 256, 14, 14])
down torch.Size([2, 512, 7, 7])

"""




if __name__ == '__main__':
    net = ResNet(BasicBlock,[2,2,2,2], 100)
    summary(net, (3, 224, 224))

    resnet18 = models.resnet18()
    summary(resnet18, (3, 224, 224))



