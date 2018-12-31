import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


class Ann(nn.Module):
    def __init__(self, node, num_classes=2):
        super(Ann, self).__init__()

        self.fc = nn.Sequential(
            # image 1 x 57 x 114
            nn.Linear(1 * 57 * 114, node),
            nn.ReLU(True),
            nn.Linear(node, node),
            nn.ReLU(True),
            nn.Linear(node, node),
            nn.ReLU(True),
            nn.Linear(node, node),
            nn.ReLU(True),
            nn.Linear(node, node),
            nn.ReLU(True),
            nn.Linear(node, node),
            nn.ReLU(True),
            nn.Linear(node, num_classes)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class Ann_small(nn.Module):
    def __init__(self, node, num_classes=2):
        super(Ann_small, self).__init__()

        self.fc = nn.Sequential(
            # image 1 x 57 x 114
            nn.Linear(1 * 57 * 114, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    image = Image.open("./dataset/EGIS_NEG_Dataset/Clay_Q1_NEG_EGIS/Clay_Q1_NEG_EGIS_000_00.bmp")
    print(np.shape(image))
