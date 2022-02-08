import torch
from torch import nn 
from torchvision.models import resnet50


class Heads(nn.Module):
    def __init__(self, num_class) -> None:
        super(Heads, self).__init__()

        self.num_class = num_class

        
