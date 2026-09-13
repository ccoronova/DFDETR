"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]   # These modules are injected externally (dependency injection)
    # Initialize the key modules and multi-scale training
    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        # print("decoder", decoder)
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        # Define the forward pass
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)  # If multi-scale training is enabled, the input tensor is resized randomly.
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)   # backbone extracts features
        x = self.encoder(x)        # encoder encodes the features
        x = self.decoder(x, targets)   # decoder produces the final output from features and targets

        return x
    # Switch the model to inference mode and convert every module with a convert_to_deploy method to the deploy format.
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
