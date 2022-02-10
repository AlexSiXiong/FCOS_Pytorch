import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50


class BackBone(nn.Module):
    def __init__(self) -> None:
        super(BackBone, self).__init__()
        self.backbone = resnet50(pretrained=True)

        self.stage0 = nn.Sequential(*list(self.backbone.children()))[:4]
        self.stage1 = self.backbone.layer1
        self.stage2 = self.backbone.layer2
        self.stage3 = self.backbone.layer3
        self.stage4 = self.backbone.layer4

    def forward(self, x):
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return [c3, c4, c5]


class FPN(nn.Module):
    def __init__(self) -> None:
        super(FPN, self).__init__()

    def forward(self, x):
        c3, c4, c5 = x

        c3_middle = nn.Conv2d(512, 256, 1)(c3)
        c4_middle = nn.Conv2d(1024, 256, 1)(c4)

        p5 = nn.Conv2d(2048, 256, 1)(c5)
        p5 = nn.Conv2d(256, 256, 1)(p5)

        p6 = nn.Conv2d(256, 256, 3, 2, 1)(p5)

        p7 = nn.ReLU(True)(p6)
        p7 = nn.Conv2d(256, 256, 3, 2, 1)(p7)

        p4 = nn.Upsample(scale_factor=2)(p5) + c4_middle
        p4 = nn.Conv2d(256, 256, 3, 1, 1)(p4)

        p3 = nn.Upsample(scale_factor=2)(p4) + c3_middle
        p3 = nn.Conv2d(256, 256, 3, 1, 1)(p3)

        return [p3, p4, p5, p6, p7]


class Heads(nn.Module):
    def __init__(self, num_class) -> None:
        super(Heads, self).__init__()

        self.num_class = num_class

        # branch 1 - classification and centerness
        self.branch_position_chunk = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.GroupNorm(1, 256)
        )

        # branch 2 - regression
        self.branch_reg_chunk = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.GroupNorm(1, 256)
        )

        # output part
        self.reg = nn.Conv2d(256, 4, 1)
        self.centerness = nn.Conv2d(256, 1, 1)
        self.classification = nn.Conv2d(256, self.num_class, 1)

    def forward(self, x):
        x1 = self.branch_position_chunk(x)
        x2 = self.branch_reg_chunk(x)

        centerness = self.centerness(x1)
        classification = self.classification(x1)

        centerness = nn.Sigmoid()(centerness)  # 1
        classification = torch.exp(classification)  # 1 # why use torch.exp?

        reg = self.reg(x2)
        reg = nn.Sigmoid()(reg)  # 4

        # print('reg', reg.shape)
        # print('center', centerness.shape)
        # print('class', classification.shape)

        return torch.cat([reg, centerness, classification], dim=1)


class FCOS(nn.Module):
    def __init__(self):
        super(FCOS, self).__init__()

        self.backbone = BackBone()
        self.fpn = FPN()
        self.heads = Heads(10)

    def forward(self, x):
        x = self.backbone(x)

        p_arr = self.fpn(x)

        res = []

        for i in p_arr:
            res.append(self.heads(i))
        return res


if __name__ == '__main__':

    net = FCOS()
    input = torch.randn(2, 3, 224, 224)

    for i in np.array(net(input)):
        print(i.shape)
