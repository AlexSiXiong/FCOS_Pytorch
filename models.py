import torch
from torch import nn 
from torchvision.models import resnet50

class BackBone(nn.Module):
    def __init__(self) -> None:
        super(BackBone).__init__()
        self.backbone = resnet50(pretrained=True, progress=True)

        

class Heads(nn.Module):
    def __init__(self, num_class) -> None:
        super(Heads, self).__init__()

        self.num_class = num_class

        # branch 1 - classification and centerness
        self.branch_position_chunk =  nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.GroupNorm(1, 256)
        )

        # branch 2 - regression
        self.branch_reg_chunk =  nn.Sequential(
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
        
        centerness = nn.Sigmoid(centerness)
        classification = torch.exp(classification)  # why use torch.exp?
        
        reg = self.reg(x2)
        reg = nn.Sigmoid(reg)
        return torch.cat([reg, centerness, classification], dim=1)

