import torchvision.models as models
from torch import nn

from utils import *

class FCN_Resnet50(nn.Module):
    def __init__(self, num_classes=2):
        super(FCN_Resnet50, self).__init__()
        self.main_model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=num_classes, pretrained_backbone=False)
        self.main_model.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        def convertBNtoGN(module, num_groups=16):
            if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
                return nn.GroupNorm(num_groups, module.num_features,
                                    eps=module.eps, affine=module.affine)
                if module.affine:
                    mod.weight.data = module.weight.data.clone().detach()
                    mod.bias.data = module.bias.data.clone().detach()

            for name, child in module.named_children():
                module.add_module(name, convertBNtoGN(child, num_groups=num_groups))

            return module

        self.main_model = convertBNtoGN(self.main_model)

    def forward(self, x):
        x = self.main_model(x)
        return x["out"]


# def get_fcn_resnet50_net():
#     net = models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, pretrained_backbone=False)
#     net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
#     def convertBNtoGN(module, num_groups=16):
#         if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
#             return nn.GroupNorm(num_groups, module.num_features,
#                                 eps=module.eps, affine=module.affine)
#             if module.affine:
#                 mod.weight.data = module.weight.data.clone().detach()
#                 mod.bias.data = module.bias.data.clone().detach()
#
#         for name, child in module.named_children():
#             module.add_module(name, convertBNtoGN(child, num_groups=num_groups))
#
#         return module
#
#     return convertBNtoGN(net)