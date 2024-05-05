import torch
import torch.nn as nn
import torch.nn.functional as F


def aspp_conv(in_channels, out_channels, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)))
        for rate in atrous_rates:
            self.convs.append(aspp_conv(in_channels, out_channels, rate))
        self.convs.append(ASPPPooling(in_channels, out_channels))

        self.projection = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels,
                      out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.projection(res)


def conv_block(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(dim_out),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(kernel_size=2)
    )


class ConvASPP(nn.Module):
    def __init__(self, in_c, in_h, in_w):
        super(ConvASPP, self).__init__()

        self.conv1 = conv_block(1, 16)
        self.aspp1 = ASPP(16, 16, [1, 2, 3])
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        self.conv4 = conv_block(64, 128)
        self.conv5 = conv_block(128, 256)

        reduced_h = in_h // 32
        reduced_w = in_w // 32
        self.fc1 = nn.Linear(reduced_h * reduced_w * 256, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.act = nn.LeakyReLU(0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.aspp1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x.squeeze(1)
