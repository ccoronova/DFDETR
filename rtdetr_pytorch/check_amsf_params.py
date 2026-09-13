"""Standalone parameter counter for AMSF module"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal get_activation for standalone use
def get_activation(act):
    if act is None:
        return nn.Identity()
    elif isinstance(act, str):
        act_map = {
            'relu': nn.ReLU(), 'silu': nn.SiLU(), 
            'gelu': nn.GELU(), 'sigmoid': nn.Sigmoid()
        }
        return act_map.get(act, nn.Identity())
    return act

# Copy all classes from AMSF.py
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride,
            padding=(kernel_size-1)//2 if padding is None else padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ChannelSelector(nn.Module):
    def __init__(self, in_channels, select_ratio=0.25):
        super().__init__()
        self.select_ratio = select_ratio
        self.select_channels = int(in_channels * select_ratio)
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        channel_scores = self.importance_net(x)
        B, C = x.shape[:2]
        scores = channel_scores.view(B, C)
        _, top_indices = torch.topk(scores, self.select_channels, dim=1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, top_indices, 1.0)
        mask = mask.view(B, C, 1, 1)
        selected_features = x * mask
        return selected_features, top_indices

class EnhancedDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.channel_selector = ChannelSelector(in_channels)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        selected_x, _ = self.channel_selector(x)
        feat = self.depthwise(selected_x)
        out = self.project(feat)
        enhance_weight = self.enhance(out)
        out = out * enhance_weight
        return out

class DirectionalConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
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

class EdgeEnhancer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.edge_weight = nn.Parameter(torch.FloatTensor([
            [0, 1, 0], [1, -4, 1], [0, 1, 0]
        ]).reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.edge_bias = nn.Parameter(torch.zeros(channels))
        self.gate = nn.Sequential(
            nn.Conv2d(channels*2, channels//2, 1),
            nn.ReLU(),
            nn.Conv2d(channels//2, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        self.edge_conv.weight = nn.Parameter(self.edge_weight)
        self.edge_conv.bias = self.edge_bias
        edge = self.edge_conv(x)
        gate = self.gate(torch.cat([x, edge], dim=1))
        return x + edge * gate

class TextureAwareModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.fusion = nn.Conv2d(channels*2, channels, 1)
        self.act = nn.GELU()
    def forward(self, x):
        texture1 = self.conv1(x) - x
        texture2 = self.conv2(x) - x
        return self.act(self.fusion(torch.cat([texture1, texture2], dim=1)))

class AdaptiveThresholdAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(channels*2, channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.pool(x)
        diff = torch.abs(x - avg)
        attention = self.sigmoid(self.conv(torch.cat([x, diff], dim=1)))
        return x * attention

class PCBDefectContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(channels*2, channels, 1)
    def forward(self, x):
        local_feat = self.local_branch(x)
        global_feat = self.global_branch(x)
        return self.fusion(torch.cat([local_feat, local_feat*global_feat], dim=1))

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
        
        self.conv0 = EnhancedDepthwiseConv(in_channels, mid_channels, 3, 1, 1)
        self.conv1 = ConvNormLayer(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = ConvNormLayer(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = ConvNormLayer(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.directional_conv = DirectionalConv(mid_channels)
        self.weight_conv1 = PCBWeightGenerator(mid_channels*3, mid_channels)
        self.weight_conv2 = PCBWeightGenerator(mid_channels*3, mid_channels)
        self.weight_conv3 = PCBWeightGenerator(mid_channels*3, mid_channels)
        self.conv_last = nn.Conv2d(3 * mid_channels, in_channels, kernel_size=1)
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.edge_enhancer = EdgeEnhancer(in_channels)
        self.texture_module = TextureAwareModule(in_channels)
        self.threshold_attn = AdaptiveThresholdAttention(in_channels)
        self.pcb_context = PCBDefectContext(in_channels)
        self.denoise_module = DenoiseModule(in_channels)

    def forward(self, x, level=None):
        residual = self.residual_conv(x)
        x = self.conv0(x)
        x3 = self.conv1(x)
        x5 = self.conv2(x3)
        x7 = self.conv3(x5)
        dir_feat = self.directional_conv(x7)
        x7 = x7 + dir_feat
        x3_mp, _ = torch.max(x3, dim=1, keepdim=True)
        x5_mp, _ = torch.max(x5, dim=1, keepdim=True)
        x7_mp, _ = torch.max(x7, dim=1, keepdim=True)
        x_cat = torch.cat([x3, x5, x7], dim=1)
        w3 = self.weight_conv1(x_cat, x3_mp)
        w5 = self.weight_conv2(x_cat, x5_mp)
        w7 = self.weight_conv3(x_cat, x7_mp)
        y = self.conv_last(torch.cat([w3*x3, w5*x5, w7*x7], dim=1))
        y = self.edge_enhancer(y)
        texture_feat = self.texture_module(y)
        y = y + texture_feat
        y = self.norm(y)
        y += residual
        return y


if __name__ == '__main__':
    print("=" * 70)
    print("AMSF Module — Detailed Parameter Analysis")
    print("=" * 70)
    
    for hidden_dim in [256, 192, 128]:
        model = AMSF(in_channels=hidden_dim, out_channels=hidden_dim, scale_num=3)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--- in_channels={hidden_dim}, mid_channels={hidden_dim//4} ---")
        print(f"Total params: {total:,}")
        
        for name, child in model.named_children():
            n = sum(p.numel() for p in child.parameters())
            pct = n / total * 100
            marker = " <-- HIGH" if pct > 10 else ""
            print(f"  {name:25s}: {n:>10,}  ({pct:5.1f}%){marker}")
