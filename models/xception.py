from thop import clever_format, profile

import torch
import torch.nn as nn
import torch.nn.functional as F


# #{ init_weights()

def init_weights(module):
    for m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Sequential,
                                Conv2dBNReLU,
                                SeparableConv2d,
                                BlockA,
                                Enc,
                                FCAttention)):
                init_weights(m)
            elif isinstance(m, (nn.ReLU, nn.ReLU6)):
                pass
            else:
                pass

# #}


# #{ Conv2dBNReLU

class Conv2dBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d):
        super(Conv2dBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False
        )

        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

# #}


# #{ SeparableConv2d

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=bias
        )

        self.bn = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)

        return x

# #}


# #{ BlockA

class BlockA(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BlockA, self).__init__()

        inter_channels = out_channels // 4

        self.skip_conv = None
        if stride != 1 or in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.separable_conv0 = nn.Sequential(
            nn.ReLU(), # ReLU is often applied before convolution
            SeparableConv2d(in_channels, inter_channels, stride=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.separable_conv1 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(inter_channels, inter_channels, stride=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.separable_conv2 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(inter_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        skip = x
        if self.skip_conv is not None:
            skip = self.skip_conv(x)

        residual = self.separable_conv0(x)
        residual = self.separable_conv1(residual)
        residual = self.separable_conv2(residual)

        out = residual + skip

        return out

# #}


# #{ Enc

class Enc(nn.Module):

    def __init__(self, in_channels, out_channels, blocks):
        super(Enc, self).__init__()
        block = list()
        block.append(BlockA(in_channels, out_channels, 2))
        for i in range(blocks - 1):
            block.append(BlockA(out_channels, out_channels, 1))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

# #}


# #{ FCAttention

class FCAttention(nn.Module):

    def __init__(self, in_channels):
        super(FCAttention,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(
            nn.Conv2d(1000, in_channels, 1, bias=False),
            # nn.BatchNorm2d(in_channels),
            nn.GroupNorm(32, in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)

# #}


# #{ XceptionA

class XceptionA(nn.Module):

    def __init__(self, num_classes):
        super(XceptionA, self).__init__()

        # Conv2d (3x3, stride 2)
        self.conv1 = Conv2dBNReLU(3, 8, 3, 2, 1)

        # Enc2
        self.enc2 = Enc(8, 48, 4)

        # Enc3
        self.enc3 = Enc(48, 96, 6)

        # Enc4
        self.enc4 = Enc(96, 192, 4)

        # FCA module
        self.fca = FCAttention(192)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.fca(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# #}


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} as device')

    xception = XceptionA(1000)
    init_weights(xception)

    dummy_input = torch.randn(1, 3, 224, 224)
    flops, params = profile(xception, inputs=(dummy_input, ))
    flops, params = clever_format([flops, params], '%.3f')

    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')

    xception = xception.to(device)
    dummy_input = dummy_input.to(device)
    output = xception(dummy_input)
    print('Output size:', output.size())
