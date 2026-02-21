import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(48)
        #self.ReLU = nn.ReLU()
        #self.ReLU = nn.GELU()
        self.ReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # 或 0.1
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(3200, 2048)  # 5*5*128 = 3200
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)
        self.f9 = nn.Linear(1000, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.ReLU(self.bn1(self.c1(x)))
        x = self.s1(x)
        x = self.ReLU(self.bn2(self.c2(x)))
        x = self.s2(x)
        x = self.ReLU(self.bn3(self.c3(x)))
        x = self.ReLU(self.bn4(self.c4(x)))
        x = self.ReLU(self.bn5(self.c5(x)))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.ReLU(self.f6(x))
        x = self.dropout(x)
        x = self.ReLU(self.f7(x))
        x = self.dropout(x)
        x = self.ReLU(self.f8(x))
        x = self.dropout(x)
        x = self.f9(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = AlexNet()
    y = model(x)


