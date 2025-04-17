import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLab(nn.Module):
    def __init__(self):
        super(DeepLab, self).__init__()
        self.deeplabv3_model = deeplabv3_resnet50(pretrained=True)

        # 获取分类器模块
        deeplabv3_classifier = self.deeplabv3_model.classifier

        # 修改分类器的最后一层，将输出通道数改为 2
        in_channels = deeplabv3_classifier[-1].in_channels
        self.deeplabv3_model.classifier[-1] = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, x):
        x = self.deeplabv3_model(x)
        return x["out"]