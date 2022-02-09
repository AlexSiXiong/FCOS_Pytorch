import torch
from torch import nn 
from torchvision.models import resnet50


class Heads(nn.Module):
    def __init__(self, num_class) -> None:
        super(Heads, self).__init__()

        self.num_class = num_class

        # branch 1 - classification and centerness
        self.branch_conv_chunk =  nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1)
        )

        # branch 2 - regression
        # same as the one above

        # output part
        self.reg = nn.Conv2d(256, 4, 1)
        self.centerness = nn.Conv2d(256, 1, 1)
        self.classification = nn.Conv2d(256, self.num_class, 1)
    
    def forward(self, x):
        x = self.branch_conv_chunk(x)
        
        reg = self.reg(x)
        centerness = self.centerness(x)
        classification = self.classification(x)

        reg = nn.Sigmoid(reg)
        centerness = nn.Sigmoid(centerness)
        classification = torch.exp(classification)  # why use torch.exp?

        return torch.cat([reg, centerness, classification], dim=1)

