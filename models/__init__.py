import numpy as np

from .fcn_resnet50 import FCN_Resnet50
from .si_unet import S1UNet
from models.Unet_DeCA.unet_deca import UNetDeCA



from models.SwinUNet.swin_unet_config import get_swin_unet_config
from models.SwinUNet.vision_transformer import SwinUnet


def get_net(net_name):
    if net_name == "fcn_resnet50":
        return FCN_Resnet50()

    elif net_name == "s1_unet" or net_name == "s2_unet" or net_name == "unet_s1_with_rate":
        in_channels = 2 if net_name == "s1_unet" else 3
        return S1UNet(in_channels=in_channels)


    elif net_name == "UNetDeCA_S2" or net_name == "UNetDeCA_S1":
        net = UNetDeCA(n_classes=2, in_channels=3)
        return net

    elif net_name == "UNetDeCA_S2" or net_name == "UNetDeCA_S1":
        net = UNetDeCA(n_classes=2, in_channels=3)
        return net
    elif net_name == "UNetDeCA_ST2" or net_name == "UNetDeCA_ST1":
        net = UNetDeCA(n_classes=2, in_channels=3, is_distillation=True)
        return net

    elif net_name == "SwinUNet_S1R" or net_name == "SwinUNet_S2":
        cfig = get_swin_unet_config()
        net = SwinUnet(config=cfig, img_size=256, num_classes=2).cuda()
        net.load_from(cfig)
        return net