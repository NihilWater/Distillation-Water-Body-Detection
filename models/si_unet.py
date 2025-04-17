import torch
import torch.nn as nn
import torch.nn.functional as F

class S1UNet(nn.Module):
    def __init__(self, n_classes=2, in_channels=3):
        super(S1UNet, self).__init__()

        # Contraction path
        self.c1_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1) # Assuming input has 3 channels (e.g., RGB)
        self.c1_dropout = nn.Dropout(0.1)
        self.c1_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2_conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.c2_dropout = nn.Dropout(0.1)
        self.c2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.c3_dropout = nn.Dropout(0.2)
        self.c3_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c4_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.c4_dropout = nn.Dropout(0.2)
        self.c4_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c5_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.c5_dropout = nn.Dropout(0.3)
        self.c5_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Expansive path
        self.u6_convt = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c6_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.c6_dropout = nn.Dropout(0.2)
        self.c6_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.u7_convt = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c7_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.c7_dropout = nn.Dropout(0.2)
        self.c7_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.u8_convt = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c8_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.c8_dropout = nn.Dropout(0.1)
        self.c8_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.u9_convt = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c9_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.c9_dropout = nn.Dropout(0.1)
        self.c9_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.outputs = nn.Conv2d(16, n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # 初始化上采样卷积层和输出卷积层
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # Contraction path
        c1 = F.relu(self.c1_conv1(x))
        c1 = self.c1_dropout(c1)
        c1 = F.relu(self.c1_conv2(c1))
        p1 = self.p1(c1)

        c2 = F.relu(self.c2_conv1(p1))
        c2 = self.c2_dropout(c2)
        c2 = F.relu(self.c2_conv2(c2))
        p2 = self.p2(c2)

        c3 = F.relu(self.c3_conv1(p2))
        c3 = self.c3_dropout(c3)
        c3 = F.relu(self.c3_conv2(c3))
        p3 = self.p3(c3)

        c4 = F.relu(self.c4_conv1(p3))
        c4 = self.c4_dropout(c4)
        c4 = F.relu(self.c4_conv2(c4))
        p4 = self.p4(c4)

        c5 = F.relu(self.c5_conv1(p4))
        c5 = self.c5_dropout(c5)
        c5 = F.relu(self.c5_conv2(c5))

        # Expansive path
        u6 = self.u6_convt(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = F.relu(self.c6_conv1(u6))
        c6 = self.c6_dropout(c6)
        c6 = F.relu(self.c6_conv2(c6))

        u7 = self.u7_convt(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = F.relu(self.c7_conv1(u7))
        c7 = self.c7_dropout(c7)
        c7 = F.relu(self.c7_conv2(c7))

        u8 = self.u8_convt(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = F.relu(self.c8_conv1(u8))
        c8 = self.c8_dropout(c8)
        c8 = F.relu(self.c8_conv2(c8))

        u9 = self.u9_convt(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = F.relu(self.c9_conv1(u9))
        c9 = self.c9_dropout(c9)
        c9 = F.relu(self.c9_conv2(c9))

        res = self.outputs(c9)

        return res

if __name__ == "__main__":
    model = S1UNet()
    print(model)

    # 假设输入是一个 RGB 图像，大小为 512 x 512
    x = torch.randn(1, 2, 512, 512)
    preds = model(x)
    print(preds.shape)  # 应该输出 torch.Size([1, 1, 512, 512])