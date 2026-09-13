
# Multi-dimensional Feature Refinement and Fusion Network

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation
__all__ = ['AMSF']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        # self.norm = nn.GroupNorm(num_groups=32,num_channels=ch_in,eps=1e-5)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
class ChannelSelector(nn.Module):
    def __init__(self, in_channels, select_ratio=0.25):
        super().__init__()
        self.select_ratio = select_ratio
        self.select_channels = int(in_channels * select_ratio)
        
        # Channel importance evaluation network
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # spatial pooling
            nn.Conv2d(in_channels, in_channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Compute the importance score of each channel
        channel_scores = self.importance_net(x)  # [B, C, 1, 1]
        
        # Select the most important channels
        B, C = x.shape[:2]
        scores = channel_scores.view(B, C)
        
        # Get the indices of the most important channel in each batch
        _, top_indices = torch.topk(scores, self.select_channels, dim=1)  # [B, select_channels]
        
        # Prepare the selection mask
        mask = torch.zeros_like(scores)
        mask.scatter_(1, top_indices, 1.0)
        mask = mask.view(B, C, 1, 1)
        
        # Apply the mask to keep important channels
        selected_features = x * mask
        
        return selected_features, top_indices

class EnhancedDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Channel selector
        self.channel_selector = ChannelSelector(in_channels)
        
        # Depthwise separable convolution
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        
        # Channel reassembly and projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Feature enhancement
        self.enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Select important channels
        selected_x, _ = self.channel_selector(x)
        
        # Depthwise separable convolution
        feat = self.depthwise(selected_x)
        
        # Project to the output dimension
        out = self.project(feat)
        
        # Feature enhancement
        enhance_weight = self.enhance(out)
        out = out * enhance_weight
        
        return out

# Direction-aware feature extraction
class DirectionalConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Convolution kernels for horizontal, vertical, and diagonal directions
        self.h_conv = nn.Conv2d(channels, channels//4, kernel_size=(1,5), padding=(0,2))
        self.v_conv = nn.Conv2d(channels, channels//4, kernel_size=(5,1), padding=(2,0))
        self.d1_conv = nn.Conv2d(channels, channels//4, kernel_size=3, padding=2, dilation=2)
        self.d2_conv = nn.Conv2d(channels, channels//4, kernel_size=3, padding=4, dilation=4)
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        h = self.h_conv(x)
        v = self.v_conv(x)
        d1 = self.d1_conv(x)
        d2 = self.d2_conv(x)
        return self.act(self.fusion(torch.cat([h,v,d1,d2], dim=1)))

# PCB weight generator
class PCBWeightGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels//2)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels//2, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, attention=None):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        if attention is not None:
            x = x * torch.sigmoid(attention)
        x = self.conv2(x)
        return self.sigmoid(x)

# Edge enhancement module
class EdgeEnhancer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Laplacian operator approximation
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.edge_weight = nn.Parameter(torch.FloatTensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]).reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.edge_bias = nn.Parameter(torch.zeros(channels))
        
        self.gate = nn.Sequential(
            nn.Conv2d(channels*2, channels//2, 1),
            nn.ReLU(),
            nn.Conv2d(channels//2, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Edge detection using the predefined weights
        self.edge_conv.weight = nn.Parameter(self.edge_weight)
        self.edge_conv.bias = self.edge_bias
        edge = self.edge_conv(x)

        # Dynamic fusion
        gate = self.gate(torch.cat([x, edge], dim=1))
        return x + edge * gate

# Texture-aware module
class TextureAwareModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.fusion = nn.Conv2d(channels*2, channels, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        texture1 = self.conv1(x) - x  # extract texture variations
        texture2 = self.conv2(x) - x
        return self.act(self.fusion(torch.cat([texture1, texture2], dim=1)))

# Adaptive threshold attention
class AdaptiveThresholdAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(channels*2, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg = self.pool(x)
        diff = torch.abs(x - avg)  # local difference
        attention = self.sigmoid(self.conv(torch.cat([x, diff], dim=1)))
        return x * attention

# PCB defect context module
class PCBDefectContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Local features
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        # Global features
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        # Feature fusion
        self.fusion = nn.Conv2d(channels*2, channels, 1)

    def forward(self, x):
        local_feat = self.local_branch(x)
        global_feat = self.global_branch(x)
        return self.fusion(torch.cat([local_feat, local_feat*global_feat], dim=1))

# Noise robustness enhancement
class DenoiseModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.smooth_conv = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        smooth = self.smooth_conv(x)
        noise = x - smooth
        gate = self.gate(torch.cat([x, noise], dim=1))
        return x - noise * gate


class AMSF(nn.Module):
    def __init__(self, in_channels, out_channels, scale_num=3):
        super(AMSF, self).__init__()
        self.scale_num = scale_num
        self.out_channels = out_channels
        mid_channels = in_channels//4
        
        # # Cascade convolution - use depthwise separable convolution to reduce parameters
        # self.conv0 = EnhancedDepthwiseConv(in_channels, mid_channels, 3, 1, 1)
        # self.conv1 = ConvNormLayer(mid_channels, mid_channels, kernel_size=3, padding=1)
        # self.conv2 = ConvNormLayer(mid_channels, mid_channels, kernel_size=3, padding=1)
        # self.conv3 = ConvNormLayer(mid_channels, mid_channels, kernel_size=3, padding=1)
        
        # Direction-aware feature extraction - for defects such as broken traces on PCBs
        # self.directional_conv = DirectionalConv(mid_channels)
        
        # Improved weight generation network - finer weight allocation
        # self.weight_conv1 = PCBWeightGenerator(mid_channels*3, mid_channels)
        # self.weight_conv2 = PCBWeightGenerator(mid_channels*3, mid_channels)
        # self.weight_conv3 = PCBWeightGenerator(mid_channels*3, mid_channels)
        
        # Feature fusion - change the output channels to in_channels
        # self.conv_last = nn.Conv2d(3 * mid_channels, in_channels, kernel_size=1)
        # self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # change the output channels
        # self.norm = nn.BatchNorm2d(in_channels)  # change the normalization layer channels
        
        # Edge enhancement module - enhance PCB trace edge features
        self.edge_enhancer = EdgeEnhancer(in_channels)
        
        # Texture-aware module - detect PCB surface anomalies
        self.texture_module = TextureAwareModule(in_channels)
        
        # Adaptive threshold attention - handle brightness differences across PCB regions
        # self.threshold_attn = AdaptiveThresholdAttention(in_channels)
        
        # Local-global feature integration - comprehensive analysis of PCB defects
        # self.pcb_context = PCBDefectContext(in_channels)
        
        # Noise robustness enhancement - handle PCB image noise
        # self.denoise_module = DenoiseModule(in_channels)

    def forward(self, x, level=None):
        # Residual connection
        # residual = self.residual_conv(x)

        # # Channel compression
        # x = self.conv0(x)

        # # Cascade feature extraction
        # x3 = self.conv1(x)
        # x5 = self.conv2(x3)
        # x7 = self.conv3(x5)

        # # Directional feature extraction
        # dir_feat = self.directional_conv(x7)
        # x7 = x7 + dir_feat
        
        # # Max pooling to extract the most salient features
        # x3_mp, _ = torch.max(x3, dim=1, keepdim=True)
        # x5_mp, _ = torch.max(x5, dim=1, keepdim=True)
        # x7_mp, _ = torch.max(x7, dim=1, keepdim=True)
        
        # # Feature fusion
        # x_cat = torch.cat([x3, x5, x7], dim=1)
        
        # # Dynamic weight generation
        # w3 = self.weight_conv1(x_cat, x3_mp)
        # w5 = self.weight_conv2(x_cat, x5_mp)
        # w7 = self.weight_conv3(x_cat, x7_mp)
        
        # # Weighted fusion and restore the number of channels
        # y = self.conv_last(torch.cat([w3*x3, w5*x5, w7*x7], dim=1))
        
        # Edge enhancement
        y = self.edge_enhancer(x)

        # Texture perception
        texture_feat = self.texture_module(y)
        y = y + texture_feat

        # # Adaptive threshold attention (ablation: disabled)
        # y = self.threshold_attn(y)

        # # Denoising (ablation: disabled)
        # y = self.denoise_module(y)

        # # PCB defect context integration (ablation: disabled)
        # defect_context = self.pcb_context(y)
        # y = y + defect_context

        # Normalization and residual connection
        # y = self.norm(y)
        # y += residual
        
        return y
