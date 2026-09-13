import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LightCrossScaleFeatureBridge']


# 轻量级跨尺度注意力模块
class LightCrossAttention(nn.Module):
    """
    轻量级跨尺度注意力：减少计算量和参数量
    通过共享投影并使用深度可分离卷积降低参数量
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channels = in_channels
        self.reduction_channels = max(in_channels // reduction_ratio, 8)
        
        # 共享投影矩阵减少参数量
        self.qkv_proj = nn.Sequential(
            nn.Conv2d(in_channels, self.reduction_channels * 2 + in_channels, 
                     kernel_size=1, groups=4),  # 使用组卷积减少参数
            nn.BatchNorm2d(self.reduction_channels * 2 + in_channels)
        )
        
        # 轻量化输出投影
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=4)
        
        # 缩放因子
        self.scale = self.reduction_channels ** -0.5
        
    def forward(self, x_target, x_source):
        batch, _, h_t, w_t = x_target.shape
        
        # 使用共享投影计算目标特征的Q和源特征的K,V
        target_qkv = self.qkv_proj(x_target)
        q = target_qkv[:, :self.reduction_channels]
        
        source_qkv = self.qkv_proj(x_source)
        k = source_qkv[:, :self.reduction_channels]
        v = source_qkv[:, self.reduction_channels*2:]
        
        # 扁平化以计算注意力
        q = q.flatten(2)  # B, C', H_t*W_t
        k = k.flatten(2)  # B, C', H_s*W_s
        v = v.flatten(2)  # B, C, H_s*W_s
        
        # 高效注意力计算 - 使用矩阵乘法
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # B, H_t*W_t, H_s*W_s
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力权重
        out = torch.bmm(attn, v.transpose(1, 2))  # B, H_t*W_t, C
        out = out.transpose(1, 2).reshape(batch, -1, h_t, w_t)
        
        # 轻量级输出投影
        out = self.out_proj(out)
        
        return out


# 轻量级特征增强模块
class LightFeatureEnhancement(nn.Module):
    """
    轻量级特征增强：替代之前的空间重组模块
    使用深度可分离卷积和更少的分组数量
    """
    def __init__(self, channels, num_groups=4):
        super().__init__()
        # 使用更少的分组减少参数量
        self.num_groups = num_groups
        
        # 深度可分离卷积 - 降低参数量
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        
        # 通道注意力 - 增强重要特征
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 特征增强 - 使用深度可分离卷积
        feat = self.depthwise(x)
        feat = self.pointwise(feat)
        feat = self.act(self.norm(feat))
        
        # 通道注意力
        attention = self.channel_attention(feat)
        enhanced = feat * attention
        
        # 残差连接
        return enhanced + x


# 轻量级特征融合模块
class LightFeatureFuser(nn.Module):
    """
    轻量级特征融合模块：替代异步特征融合器
    使用加法而非拼接，减少后续处理的参数量
    """
    def __init__(self, channels):
        super().__init__()
        # 特征转换 - 使用组卷积减少参数
        self.transform = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, groups=4),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])
        
        # 特征加权 - 学习不同特征的重要性
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
        # 输出投影 - 使用深度可分离卷积
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feat1, feat2, feat3):
        # 确保空间尺寸一致
        size = feat2.shape[2:]  # 使用中层特征的尺寸
        if feat1.shape[2:] != size:
            feat1 = F.interpolate(feat1, size=size, mode='bilinear', align_corners=False)
        if feat3.shape[2:] != size:
            feat3 = F.interpolate(feat3, size=size, mode='bilinear', align_corners=False)
        
        # 特征转换
        feats = [transform(feat) for transform, feat in zip(self.transform, [feat1, feat2, feat3])]
        
        # 加权求和 - 比拼接使用更少的参数
        weights = F.softmax(self.weights, dim=0)
        fused = sum(w * f for w, f in zip(weights, feats))
        
        # 输出投影
        return self.out_proj(fused)


class LightCrossScaleFeatureBridge(nn.Module):
    """
    轻量级跨尺度特征桥接模块: 
    大幅度减少参数量，同时保持多尺度特征融合的核心功能
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 跨尺度注意力 - 只保留两个关键的注意力模块
        self.high_to_mid_attention = LightCrossAttention(channels)
        self.low_to_mid_attention = LightCrossAttention(channels)
        
        # 中层特征增强 - 使用轻量级模块替代空间重组
        self.feature_enhancement = LightFeatureEnhancement(channels)
        
        # 特征融合 - 使用轻量级融合模块
        self.feature_fuser = LightFeatureFuser(channels)
        
        # 最终处理 - 使用深度可分离卷积
        self.final_process = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, low_feat, mid_feat, high_feat):
        """
        将低层、中层、高层特征融合到中层特征中
        更高效的实现，显著减少参数量和计算量
        """
        # 保存原始特征用于残差连接
        identity = mid_feat
        
        # 1. 跨尺度注意力交互
        high_to_mid = self.high_to_mid_attention(mid_feat, high_feat)
        low_to_mid = self.low_to_mid_attention(mid_feat, low_feat)
        
        # 2. 中层特征增强
        mid_enhanced = self.feature_enhancement(mid_feat)
        
        # 3. 特征融合 - 将三个特征流融合在一起
        fused = self.feature_fuser(low_to_mid, mid_enhanced, high_to_mid)
        
        # 4. 最终处理
        output = self.final_process(fused)
        
        # 5. 残差连接 - 确保信息流畅通
        return output + identity
