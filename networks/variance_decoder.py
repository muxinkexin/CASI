"""Training-only ResNet variance decoder, following the released DSI architecture.

The output is a positive sigma (not sigma squared), with sigmoid as in DSI.
DepthNet itself is unchanged. Source: zhangtian33/DSI-training,
networks/variance_decoder.py; this implementation uses explicit tensor sizes.
"""
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from layers import ConvBlock, Conv3x3


class VarDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=(0, 1, 2)):
        super().__init__()
        self.scales = tuple(scales)
        channels = (16, 32, 64, 128, 256)
        convs = OrderedDict()
        for i in range(4, -1, -1):
            incoming = num_ch_enc[-1] if i == 4 else channels[i + 1]
            convs[("upconv", i, 0)] = ConvBlock(int(incoming), channels[i])
            incoming = channels[i] + (int(num_ch_enc[i - 1]) if i > 0 else 0)
            convs[("upconv", i, 1)] = ConvBlock(incoming, channels[i])
        for scale in self.scales:
            convs[("varconv", scale)] = Conv3x3(channels[scale], 1)
        self.convs = convs
        self.decoder = nn.ModuleList(convs.values())

    def forward(self, features):
        outputs = {}
        x = features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            size = features[i - 1].shape[-2:] if i > 0 else (features[0].shape[-2] * 2, features[0].shape[-1] * 2)
            x = F.interpolate(x, size=size, mode="nearest")
            if i > 0:
                x = torch.cat((x, features[i - 1]), dim=1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("var", i)] = torch.sigmoid(self.convs[("varconv", i)](x))
        return outputs
