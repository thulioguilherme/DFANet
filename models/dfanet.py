from thop import profile, clever_format

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import xception

import os
import sys

# #{ include this project packages

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.join(current_dir, '..')

sys.path.append(project_root)

# #}

from config.settings import DFANetConfig


# #{ load_dfanet_backbone_weights()

def load_dfanet_backbone_weights(dfanet, backbone_weights_path='xception.pth'):
    backbone_weights = torch.load(backbone_weights_path)
    backbone_weights_dict = dict(backbone_weights)
    # print(backbone_weights_dict.keys())

    dfanet_state_dict = dfanet.state_dict()
    # print(dfanet_state_dict.keys())

    for key in backbone_weights_dict:
        if key.split('.')[0] == 'conv1':
            new_key = 'encoder.' + key
            if dfanet_state_dict[new_key].size() == backbone_weights_dict[key].size():
                dfanet_state_dict[new_key] = backbone_weights_dict[key]
                # print('Successfully load weights for layer', new_key)
            else:
                print('Unable to load weights for layer', new_key)

        if 'block' in key.split('.'):
            new_key = 'encoder.stage0.' + key
            if backbone_weights_dict[key].size() == dfanet_state_dict[new_key].size():
                dfanet_state_dict[new_key] = backbone_weights_dict[key]
                # print('Successfully load weights for layer', new_key)
            else:
                print('Unable to load weights for layer', new_key)

            # new_key = 'encoder.stage1.' + key
            # if backbone_weights_dict[key].size() == dfanet_state_dict[new_key].size():
            #     dfanet_state_dict[new_key] = backbone_weights_dict[key]
            #     print('Successfully load weights for layer', new_key)
            # else:
            #     print('Unable to load weights for layer', new_key)

            # new_key = 'encoder.stage2.' + key
            # if backbone_weights_dict[key].size() == dfanet_state_dict[new_key].size():
            #     dfanet_state_dict[new_key] = backbone_weights_dict[key]
            #     print('Successfully load weights for layer', new_key)
            # else:
            #     print('Unable to load weights for layer', new_key)

    dfanet.load_state_dict(dfanet_state_dict)

    return dfanet

# #}

# #{ Stage

class Stage(nn.Module):
    def __init__(self, channels_config, stage_index):
        super(Stage, self).__init__()
        self.stage_index = stage_index

        # Enc2
        self.enc2 = Enc(channels_config[0], 48, 4)

        # Enc3
        self.enc3 = Enc(channels_config[1], 96, 6)

        # Enc4
        self.enc4 = Enc(channels_config[2], 192, 4)

        # FCA module
        self.fca = FCAttention(192)

    def forward(self, x0, *args):
        out0 = self.enc2(x0)
        if self.stage_index in [1, 2]:
            out1 = self.enc3(torch.cat([out0, args[0]], 1))
            out2 = self.enc4(torch.cat([out1, args[1]], 1))
        else:
            out1 = self.enc3(out0)
            out2 = self.enc4(out1)
        out3 = self.fca(out2)

        return [out0, out1, out2, out3]

# #}

# #{ Encoder

class Encoder(nn.Module):
    def __init__(self, channels_config):
        super(Encoder, self).__init__()

        # #{ Modified Xception as backbone

        self.conv1 = Conv2dBNReLU(3, 8, 3, 2, 1)

        self.stage0 = Stage(channels_config[0], 0)

        # #}

        self.stage1 = Stage(channels_config[1], 1)
        self.stage2 = Stage(channels_config[2], 2)

    def forward(self, x):
        x = self.conv1(x)

        s0_enc2_out, s0_enc3_out, s0_enc4_out, s0_fca_out = self.stage0(x)
        s0_out = F.interpolate(s0_fca_out, scale_factor=4, mode='bilinear', align_corners=True)

        s1_enc2_out, s1_enc3_out, s1_enc4_out, s1_fca_out = self.stage1(torch.cat([s0_enc2_out, s0_out], 1), s0_enc3_out, s0_enc4_out)
        s1_out = F.interpolate(s1_fca_out, scale_factor=4, mode='bilinear', align_corners=True)

        s2_enc2_out, s2_enc3_out, s2_enc4_out, s2_fca_out = self.stage2(torch.cat([s1_enc2_out, s1_out], 1), s1_enc3_out, s1_enc4_out)

        return [s0_enc2_out, s1_enc2_out, s2_enc2_out, s0_fca_out, s1_fca_out, s2_fca_out]

# #}

# #{ Decoder

class Decoder(nn.Module):
    def __init__(self, num_classes, decoder_channels):
        super(Decoder, self).__init__()

        self.s0_enc2_reducer = Conv2dBNReLU(48, decoder_channels, 1)
        self.s1_enc2_reducer = Conv2dBNReLU(48, decoder_channels, 1)
        self.s2_enc2_reducer = Conv2dBNReLU(48, decoder_channels, 1)

        self.llf_reducer = Conv2dBNReLU(decoder_channels, decoder_channels, 1)

        self.s0_fca_reducer = Conv2dBNReLU(192, decoder_channels, 1)
        self.s1_fca_reducer = Conv2dBNReLU(192, decoder_channels, 1)
        self.s2_fca_reducer = Conv2dBNReLU(192, decoder_channels, 1)

        self.output_refiner = Conv2dBNReLU(decoder_channels, num_classes, 3, padding=1)

    def forward(self, s0_enc2_out, s1_enc2_out, s2_enc2_out, s0_fca_out, s1_fca_out, s2_fca_out):
        s0_enc2_out = self.s0_enc2_reducer(s0_enc2_out)
        s1_enc2_out = F.interpolate(self.s1_enc2_reducer(s1_enc2_out), scale_factor=2, mode='bilinear', align_corners=True)
        s2_enc2_out = F.interpolate(self.s2_enc2_reducer(s2_enc2_out), scale_factor=4, mode='bilinear', align_corners=True)

        # print('s0_out', s0_enc2_out.size())
        # print('s1_out', s1_enc2_out.size())
        # print('s2_out', s2_enc2_out.size())

        llf_out = s0_enc2_out + s1_enc2_out + s2_enc2_out
        llf_out = self.llf_reducer(llf_out)

        s0_fca_out = F.interpolate(self.s0_fca_reducer(s0_fca_out), scale_factor=4, mode='bilinear', align_corners=True)
        s1_fca_out = F.interpolate(self.s1_fca_reducer(s1_fca_out), scale_factor=8, mode='bilinear', align_corners=True)
        s2_fca_out = F.interpolate(self.s2_fca_reducer(s2_fca_out), scale_factor=16, mode='bilinear', align_corners=True)

        # print('s0_fca_out', s0_fca_out.size())
        # print('s1_fca_out', s1_fca_out.size())
        # print('s2_fca_out', s2_fca_out.size())
        # print('llf_out', llf_out.size())

        out = llf_out + s0_fca_out + s1_fca_out + s2_fca_out
        out = self.output_refiner(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        return out

# #}

# #{ DFANet

class DFANet(nn.Module):
    def __init__(self, config):
        super(DFANet,self).__init__()
        self.encoder = Encoder(config.ENCODER_CHANNELS)
        if not config.USE_PRETRAINED_WEIGHTS:
            print('Initializing encoder weights')
            self.init_weights(self.encoder)

        self.decoder = Decoder(config.NUM_CLASSES, config.DECODER_CHANNELS)
        if not config.USE_PRETRAINED_WEIGHTS:
            print('Initializing decoder weights')
            self.init_weights(self.decoder)

    def forward(self, x):
        x0, x1, x2, x5, x6, x7 = self.encoder(x)
        x = self.decoder(x0, x1, x2, x5, x6, x7)

        return x

    def init_weights(self, module):
        for n, m in module.named_children():
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
                                Stage,
                                Enc,
                                BlockA,
                                SeparableConv2d,
                                FCAttention,
                                Encoder,
                                Decoder)):
                self.init_weights(m)
            elif isinstance(m, (nn.ReLU, nn.ReLU, nn.ReLU6)):
                pass
            else:
                pass

# #}

if __name__=='__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dfanet_config = DFANetConfig()
    dfanet = DFANet(dfanet_config)

    load_dfanet_backbone_weights(dfanet, backbone_weights_path='../xception.pth')

    dummy_input = torch.randn(1, 3, 512, 1024)
    flops, params = profile(dfanet, inputs=(dummy_input, ))
    flops, params = clever_format([flops, params], '%.3f')

    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')

    dfanet = dfanet.to(device)
    dummy_input = dummy_input.to(device)
    output = dfanet(dummy_input)
    print('Output size:', output.size())

    # torch.onnx.export(
    #     dfanet,
    #     dummy_input,
    #     "dfanet.onnx",
    #     verbose=False,
    #     input_names=['input'],
    #     output_names=['output']
    # )

    # start = time.time()
    # outputs = dfanet(dummy_input)
    # end = time.time()

    # print(outputs.size())
    # print('Inference time', end - start)
