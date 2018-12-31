import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            # image 1 x 57 x 116
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # quater

            # image 64 x 28 x 56
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # quater

            # image 128 x 14 x 27
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # quater

            # image 256 x 7 x 13
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # quater

            # image 512 x 3 x 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # quater
            # image 512 x 1 x 3
        )

        self.fc = nn.Sequential(
            nn.Linear(1 * 3 * 512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class ConvNet_half(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            # image 1 x 57 x 114
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # quater

            # image 64 x 28 x 56
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # quater

            # image 128 x 14 x 27
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # quater

            # image 256 x 7 x 13
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # quater

            # image 512 x 3 x 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # quater
            # image 512 x 1 x 3
        )

        self.fc = nn.Sequential(
            nn.Linear(1 * 3 * 512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

class _conv_ds_res(nn.Module):
    def __init__(self):
        super(_conv_ds_res, self).__init__()

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, padding=1, groups=inp, bias=False),
                # nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
                # nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        pass


class ConvNet_ds(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet_ds, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, 1, padding=1, bias=False),
                # nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),

                nn.Conv2d(oup, oup, 3, stride, padding=1, bias=False),
                # nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        self.conv = nn.Sequential(
            # image 1 x 57 x 114
            # 114 57 29 15 8 4 2
            conv_bn(1, 12, 2),  # 64 x 29 x 58
            conv_dw(12, 12, 2),  # 128 x 15 x 29
            conv_dw(12, 12, 2),  # 256 x 8 x 16
            conv_dw(12, 12, 2),  # 512 x 4 x 8
            conv_dw(12, 12, 2)  # 512 x 2 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 4 * 12, 32),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


if __name__ == '__main__':
    image = Image.open("./dataset/EGIS_NEG_Dataset/Clay_Q1_NEG_EGIS/Clay_Q1_NEG_EGIS_000_00.bmp")
    print(np.shape(image))
