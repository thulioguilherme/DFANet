from thop import profile, clever_format

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias
        )

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            1, # kernel_size
            1, # stride
            0, # padding
            1, # dilation
            bias=bias
        )

    def forward(self, x):
        x = self.fix_padding(x, self.kernel_size, self.dilation)

        x = self.depthwise(x)

        x = self.bn(x)
        x = self.relu(x)

        x = self.pointwise(x)

        return x

    def fix_padding(self, x, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))

        return padded_inputs

# #}

# #{ BlockA

class BlockA(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BlockA, self).__init__()

        self.stride = stride

        if stride != 1:
            # Conv2d (1x1, stride 2)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 2, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

        rep = list()
        inter_channels = out_channels // 4

        # SeparableConv2d (3x3, stride 1)
        rep.append(SeparableConv2d(in_channels, inter_channels, 3, 1))
        rep.append(nn.BatchNorm2d(inter_channels))
        rep.append(nn.ReLU())

        # SeparableConv2d (3x3, stride 1)
        rep.append(SeparableConv2d(inter_channels, inter_channels, 3, 1))
        rep.append(nn.BatchNorm2d(inter_channels))
        rep.append(nn.ReLU())

        # SeparableConv2d (3x3, stride 1 or 2)
        rep.append(SeparableConv2d(inter_channels, out_channels, 3, stride))
        rep.append(nn.BatchNorm2d(out_channels))
        rep.append(nn.ReLU())

        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)

        if self.stride != 1:
            out = out + self.bn(self.conv1(x))
        else:
            out = out + x

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

if __name__=='__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    xception = XceptionA(1000)
    dummy_input = torch.randn(1, 3, 224, 224)

    flops, params = profile(xception, inputs=(dummy_input, ))
    flops, params = clever_format([flops, params], '%.3f')

    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')

    xception = xception.to(device)
    dummy_input = dummy_input.to(device)
    output = xception(dummy_input)
    print('Output size:', output.size())

    # torch.onnx.export(
    #     xception,
    #     dummy_input,
    #     "xception.onnx",
    #     verbose=False,
    #     input_names=['input'],
    #     output_names=['output']
    # )
