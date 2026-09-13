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

# The cfg dict defines the number of residual blocks per stage for ResNet models of different depths
ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}

# Pretrained weight download links
donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}

# Define the basic residual block
# BasicBlock is used for shallow ResNets, e.g., ResNet18
class BasicBlock(nn.Module):
    # Expansion factor of the output channels relative to the input channels, set to 1 (same size)
    expansion = 1
   # init: shortcut is a bool indicating whether to use a skip connection. If False, the input x must be reshaped by an extra conv or pooling layer to match the main-path output. variant specifies the residual block variant 'b'.
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__() # Execute the initialization logic

        self.shortcut = shortcut   # Use the skip connection if available
 # Otherwise
        if not shortcut:
            if variant == 'd' and stride == 2:
                # Sequentially combine multiple neural layers so they can be called in order during forward propagation. OrderedDict is an ordered dictionary.
                self.short = nn.Sequential(OrderedDict([
                    # Stride 2 means downsampling; ceil_mode=True rounds up.
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    # 1x1 conv to adjust the number of channels; stride 1 keeps the feature map size (except channel dim).
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                # Use the passed stride as the actual stride; a stride greater than 1 downsamples the feature map.
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
      # Convolution and normalization operations
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

# Used for deep ResNets (ResNet50, ResNet101)
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

# Define a sequence of residual blocks for each stage of ResNet
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

# Define the PResNet model
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

          # Helper functions
        # Freeze module parameters so the given module's parameters are no longer updated
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False
# Freeze batch norm layers: recursively traverse the model and replace every batch norm layer with a frozen one
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
        # print("x in PResNet_forward", x.shape)
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


