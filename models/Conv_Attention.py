import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
    
class CBAM(nn.Module):
    def __init__(self, channel, reduction=8, kernel_size=3):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


def conv_block(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in,dim_out,kernel_size,stride,padding,bias=bias),
        nn.BatchNorm2d(dim_out),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(kernel_size=2)
    )


class ConvAttention(nn.Module):
    def __init__(self, in_c, in_h, in_w):
        super(ConvAttention, self).__init__()
        self.conv1 = conv_block(in_c, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        self.cbam1 = CBAM(64)
        self.conv4 = conv_block(64, 128)
        self.cbam2 = CBAM(128)
        self.conv5 = conv_block(128, 256)
        self.cbam3 = CBAM(256)

        self.avg_pool=nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64+128+256, 256)
        # self.fc1 = nn.Linear(6*9*256, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.act = nn.LeakyReLU(0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        y1 = self.cbam1(x)

        x = self.conv4(y1)
        y2 = self.cbam2(x)

        x = self.conv5(y2)
        y3 = self.cbam3(x)

        y1=self.avg_pool(y1).permute(0,2,3,1)
        y2=self.avg_pool(y2).permute(0,2,3,1)
        y3=self.avg_pool(y3).permute(0,2,3,1)

        x=torch.cat((y1,y2,y3),3)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x.squeeze(1)
