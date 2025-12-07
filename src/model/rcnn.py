import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn


class RCNN_VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Use pretrained VGG16 as feature extractor
        self.features = vgg16_bn(pretrained=True).features

        # Flatten output size = 512 * 7 * 7 (VGG16 default)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # this no matter how will output 7x7

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        # RCNN output heads:
        self.cls_head = nn.Linear(4096, num_classes)   # classification (2 classes)
        self.bbox_head = nn.Linear(4096, 4)            # regression (dx,dy,dw,dh)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        cls_logits = self.cls_head(x)
        bbox_regs = self.bbox_head(x)

        return cls_logits, bbox_regs
