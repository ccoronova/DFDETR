'''by lyuwenyu
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from sympy.codegen import Print

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d

from src.core import register


__all__ = ['PResNet']

#cfg字典定义了不同深度的resnet模型中的每个阶段的残差块数量
ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}

#预训练权重下载链接
donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}

#定义基本残差块
#basicblock用于浅层的resnet例如r18
class BasicBlock(nn.Module):
    # 输出通道数相对于输入通道数的扩展倍数，为1，相同
    expansion = 1
   #初始化，shortcut，布尔值，指是否使用捷径连接。如果为False，则需要通过额外的卷积层或池化层来调整输入x的形状，以便与主路径的输出相匹配。variant，为指定残差块变种b
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__() #执行初始化函数逻辑

        self.shortcut = shortcut   #有就用捷径连接
 #如果没有
        if not shortcut:
            if variant == 'd' and stride == 2:
                #顺序组合多个神经网络层，方便在前向传播等过程中依次调用这些层进行计算。ordereddict，有序字典
                self.short = nn.Sequential(OrderedDict([
                    #步长为2，代表下采样，ceil_mode设置为ture，代表向上取整
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    #卷积核1，用于调整特征图的通道数，步长为1，不会改变特征图的尺寸（除了通道维度）
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                #传入参数stride作为实际步长值，stride大于1，代表特征图进行下采样操作。
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
      #卷积操作以及归一化操作
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

#用于深层的resnet（resnet50，resnet101）
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

#定义残差块序列，用于resnet的每个阶段
class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in,
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act)
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out

#定义presnet模型
@register
class PResNet(nn.Module):
    def __init__(
        self,
        depth,
        variant='d',
        num_stages=4,
        return_idx=[0, 1, 2, 3],
        act='relu',
        freeze_at=-1,
        freeze_norm=True,
        pretrained=False):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self.load_state_dict(state)
            print(f'Load PResNet{depth} state_dict')

          #辅助函数
        #冻结模块参数，用于冻结给定模块参数，使其不再更新
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False
#冻结批量归一化层，递归遍历模型，将所有的批量归一化层替换为冻结的批量归一化层
    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        # print("PResNet_forward中x", x.shape)
        conv1 = self.conv1(x)
        # print("conv1", conv1.shape)

        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        # print("x", x.shape)

        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


