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
    __inject__ = ['backbone', 'encoder', 'decoder', ]   #依赖注意机制，模型的这几个是外部动态传入的模块
#初始化这几个关键模块，还有一个多尺度训练
    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        # print("decoder", decoder)
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        #定义前向传播
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)  #如果启用了多尺度训练（self.multi_scale），会对输入张量进行随机缩放。
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)   #backbone提取特征
        x = self.encoder(x)        #encoder对特征进行编码
        x = self.decoder(x, targets)   #decoder根据特征和目标生成最终输出

        return x
    #将模型切换到推理模式，并调用所有具有 convert_to_deploy 方法的模块，将其转换为部署格式。
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
