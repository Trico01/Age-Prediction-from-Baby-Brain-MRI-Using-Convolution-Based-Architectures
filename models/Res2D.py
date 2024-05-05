import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            dim_in, dim_out, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, stride=stride, bias=False),
            nn.BatchNorm2d(dim_out)
        )

    def forward(self, x):
        residual = x
        x = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class Res2D(nn.Module):
    def __init__(self, dim_in, dim_out, num_layer):
        super(Res2D, self).__init__()

        self.conv1 = nn.Conv2d(dim_in, 16, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 32, num_layer[0])
        self.layer2 = self._make_layer(32, 64, num_layer[1], 2)
        self.layer3 = self._make_layer(64, 128, num_layer[2], 2)
        self.layer4 = self._make_layer(128, 256, num_layer[3], 2)
        self.fc = nn.Linear(70*256, dim_out)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, dim_in, dim_out, num_block, stride=1):
        layers = []
        layers.append(ResBlock(dim_in, dim_out, stride))
        for _ in range(1, num_block):
            layers.append(ResBlock(dim_out, dim_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x.squeeze(1)
