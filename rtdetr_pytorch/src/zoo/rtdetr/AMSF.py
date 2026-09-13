
#多维特征精炼与融合网络 (Multi-dimensional Feature Refinement and Fusion Network)

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
        
        # 通道重要性评估网络
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 空间池化
            nn.Conv2d(in_channels, in_channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 计算每个通道的重要性分数
        channel_scores = self.importance_net(x)  # [B, C, 1, 1]
        
        # 选择最重要的通道
        B, C = x.shape[:2]
        scores = channel_scores.view(B, C)
        
        # 获取每个batch中最重要的通道索引
        _, top_indices = torch.topk(scores, self.select_channels, dim=1)  # [B, select_channels]
        
        # 准备选择掩码
        mask = torch.zeros_like(scores)
        mask.scatter_(1, top_indices, 1.0)
        mask = mask.view(B, C, 1, 1)
        
        # 应用掩码,保留重要通道
        selected_features = x * mask
        
        return selected_features, top_indices

class EnhancedDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 通道选择器
        self.channel_selector = ChannelSelector(in_channels)
        
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        
        # 通道重组和投影
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 选择重要通道
        selected_x, _ = self.channel_selector(x)
        
        # 深度可分离卷积
        feat = self.depthwise(selected_x)
        
        # 投影到输出维度
        out = self.project(feat)
        
        # 特征增强
        enhance_weight = self.enhance(out)
        out = out * enhance_weight
        
        return out

# 方向感知特征提取
class DirectionalConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 水平、垂直、对角线方向的卷积核
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

# PCB权重生成器
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

# 边缘增强模块
class EdgeEnhancer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 拉普拉斯算子近似
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
        # 使用预定义权重进行边缘检测
        self.edge_conv.weight = nn.Parameter(self.edge_weight)
        self.edge_conv.bias = self.edge_bias
        edge = self.edge_conv(x)
        
        # 动态融合
        gate = self.gate(torch.cat([x, edge], dim=1))
        return x + edge * gate

# 纹理感知模块
class TextureAwareModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.fusion = nn.Conv2d(channels*2, channels, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        texture1 = self.conv1(x) - x  # 提取纹理变化
        texture2 = self.conv2(x) - x
        return self.act(self.fusion(torch.cat([texture1, texture2], dim=1)))

# 自适应阈值注意力
class AdaptiveThresholdAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(channels*2, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg = self.pool(x)
        diff = torch.abs(x - avg)  # 局部差异
        attention = self.sigmoid(self.conv(torch.cat([x, diff], dim=1)))
        return x * attention

# PCB缺陷上下文模块
class PCBDefectContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 局部特征
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        # 全局特征
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        # 特征融合
        self.fusion = nn.Conv2d(channels*2, channels, 1)
        
    def forward(self, x):
        local_feat = self.local_branch(x)
        global_feat = self.global_branch(x)
        return self.fusion(torch.cat([local_feat, local_feat*global_feat], dim=1))

# 噪声鲁棒性增强
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
        
        # # 级联卷积 - 使用深度可分离卷积减少参数量
        # self.conv0 = EnhancedDepthwiseConv(in_channels, mid_channels, 3, 1, 1)
        # self.conv1 = ConvNormLayer(mid_channels, mid_channels, kernel_size=3, padding=1)
        # self.conv2 = ConvNormLayer(mid_channels, mid_channels, kernel_size=3, padding=1)
        # self.conv3 = ConvNormLayer(mid_channels, mid_channels, kernel_size=3, padding=1)
        
        # 方向感知特征提取 - 针对PCB中的线路断路等缺陷
        # self.directional_conv = DirectionalConv(mid_channels)
        
        # 改进的权重生成网络 - 更精细的权重分配
        # self.weight_conv1 = PCBWeightGenerator(mid_channels*3, mid_channels)
        # self.weight_conv2 = PCBWeightGenerator(mid_channels*3, mid_channels)
        # self.weight_conv3 = PCBWeightGenerator(mid_channels*3, mid_channels)
        
        # 特征融合 - 修改输出通道数为in_channels
        # self.conv_last = nn.Conv2d(3 * mid_channels, in_channels, kernel_size=1)
        # self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 修改输出通道数
        # self.norm = nn.BatchNorm2d(in_channels)  # 修改归一化层的通道数
        
        # 边缘增强模块 - 增强PCB线路边缘特征
        self.edge_enhancer = EdgeEnhancer(in_channels)
        
        # 纹理感知模块 - 检测PCB表面异常
        self.texture_module = TextureAwareModule(in_channels)
        
        # 自适应阈值注意力 - 处理PCB不同区域亮度差异
        # self.threshold_attn = AdaptiveThresholdAttention(in_channels)
        
        # 局部-全局特征整合 - 综合分析PCB缺陷
        # self.pcb_context = PCBDefectContext(in_channels)
        
        # 噪声鲁棒性增强 - 处理PCB图像噪声
        # self.denoise_module = DenoiseModule(in_channels)

    def forward(self, x, level=None):
        # 残差连接
        # residual = self.residual_conv(x)
        
        # # 通道压缩
        # x = self.conv0(x)
        
        # # 级联特征提取
        # x3 = self.conv1(x)
        # x5 = self.conv2(x3)
        # x7 = self.conv3(x5)
        
        # # 方向特征提取
        # dir_feat = self.directional_conv(x7)
        # x7 = x7 + dir_feat
        
        # # 最大池化提取显著特征
        # x3_mp, _ = torch.max(x3, dim=1, keepdim=True)
        # x5_mp, _ = torch.max(x5, dim=1, keepdim=True)
        # x7_mp, _ = torch.max(x7, dim=1, keepdim=True)
        
        # # 特征融合
        # x_cat = torch.cat([x3, x5, x7], dim=1)
        
        # # 动态权重生成
        # w3 = self.weight_conv1(x_cat, x3_mp)
        # w5 = self.weight_conv2(x_cat, x5_mp)
        # w7 = self.weight_conv3(x_cat, x7_mp)
        
        # # 加权融合并恢复通道数
        # y = self.conv_last(torch.cat([w3*x3, w5*x5, w7*x7], dim=1))
        
        # 边缘增强
        y = self.edge_enhancer(x)
        
        # 纹理感知
        texture_feat = self.texture_module(y)
        y = y + texture_feat
        
        # # 自适应阈值注意力 (消融：关闭)
        # y = self.threshold_attn(y)
        
        # # 去噪处理 (消融：关闭)
        # y = self.denoise_module(y)
        
        # # PCB缺陷上下文整合 (消融：关闭)
        # defect_context = self.pcb_context(y)
        # y = y + defect_context
        
        # 归一化和残差连接
        # y = self.norm(y)
        # y += residual
        
        return y
