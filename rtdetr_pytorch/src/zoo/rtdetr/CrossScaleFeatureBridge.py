import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CrossScaleFeatureBridge']


# 跨尺度注意力模块 - 完全不同于AMSF的注意力机制
class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力模块：在不同尺度特征间建立关联
    与AMSF不同，该模块专注于跨尺度交互而非单尺度特征增强
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channels = in_channels
        self.reduction_channels = max(in_channels // reduction_ratio, 8)
        
        # 查询变换 - 用于目标层特征
        self.query_conv = nn.Conv2d(in_channels, self.reduction_channels, kernel_size=1)
        
        # 键值变换 - 用于源层特征
        self.key_conv = nn.Conv2d(in_channels, self.reduction_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 缩放因子
        self.scale = nn.Parameter(torch.ones(1) * self.reduction_channels ** -0.5)
        
        # 输出投影
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x_target, x_source):
        """
        x_target: 目标特征，注意力的应用对象 [B, C, H, W]
        x_source: 源特征，提供注意力的信息来源 [B, C, H', W']
        """
        batch, _, h_t, w_t = x_target.shape
        _, _, h_s, w_s = x_source.shape
        
        # 生成查询、键、值
        q = self.query_conv(x_target).view(batch, self.reduction_channels, -1)  # B, C', H_t*W_t
        k = self.key_conv(x_source).view(batch, self.reduction_channels, -1)    # B, C', H_s*W_s
        v = self.value_conv(x_source).view(batch, self.channels, -1)             # B, C, H_s*W_s
        
        # 计算注意力分数 (与AMSF完全不同的注意力计算方式)
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale  # B, H_t*W_t, H_s*W_s
        attn = F.softmax(attn, dim=-1)  # 归一化注意力权重
        
        # 应用注意力权重
        out = torch.bmm(v, attn.permute(0, 2, 1))  # B, C, H_t*W_t
        out = out.view(batch, self.channels, h_t, w_t)
        out = self.proj(out)
        
        return out


# 特征空间重组模块 - AMSF没有的组件
class FeatureSpatialRearrangement(nn.Module):
    """
    特征空间重组：重新排列特征图的空间关系
    与AMSF的处理方式完全不同，强调空间维度的变换
    """
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.num_groups = num_groups
        assert channels % num_groups == 0, "通道数必须能被组数整除"
        
        # 特征分组处理
        group_channels = channels // num_groups
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1, groups=group_channels),
                nn.InstanceNorm2d(group_channels),  # 使用实例归一化而非批归一化(AMSF)
                nn.GELU()  # GELU与AMSF的ReLU差异化
            )
            for _ in range(num_groups)
        ])
        
        # 空间重组混合器 - 修复LayerNorm不兼容的问题
        self.mixer = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),  # 使用BatchNorm替代LayerNorm以避免尺寸错误
            nn.ReLU(inplace=True),     # 使用ReLU替代，确保区别于AMSF
            nn.Dropout2d(0.1)          # 添加空间Dropout
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        group_size = C // self.num_groups
        
        # 通道分组
        groups = torch.split(x, group_size, dim=1)
        processed_groups = []
        
        # 分组处理
        for i, (group, conv) in enumerate(zip(groups, self.group_convs)):
            # 对每个组应用不同方向的变换
            if i % 4 == 0:  # 水平方向变换
                group = torch.flip(group, [3])
            elif i % 4 == 1:  # 垂直方向变换
                group = torch.flip(group, [2])
            elif i % 4 == 2:  # 旋转变换
                group = torch.rot90(group, 1, [2, 3])
            
            processed = conv(group)
            
            # 恢复原始排列
            if i % 4 == 0:
                processed = torch.flip(processed, [3])
            elif i % 4 == 1:
                processed = torch.flip(processed, [2])
            elif i % 4 == 2:
                processed = torch.rot90(processed, -1, [2, 3])
                
            processed_groups.append(processed)
            
        # 重组后的特征
        rearranged_feat = torch.cat(processed_groups, dim=1)
        mixed_feat = self.mixer(rearranged_feat)
        
        return mixed_feat + x  # 残差连接


# 多级特征分解模块 - 与AMSF显著不同的特征分解方式
class MultiLevelFeatureDecomposition(nn.Module):
    """
    将特征分解为多个组件，每个组件捕获不同频率的信息
    与AMSF的边缘检测和纹理分析方式完全不同
    """
    def __init__(self, channels, levels=3):
        super().__init__()
        self.levels = levels
        
        # 多级特征提取器，使用不同膨胀率捕获不同尺度
        self.decomposers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2**i, dilation=2**i, groups=channels),
                nn.GroupNorm(channels // 16, channels),  # 使用组归一化代替AMSF的批归一化
                nn.SiLU()  # 使用SiLU激活函数，与AMSF的GELU不同
            )
            for i in range(levels)
        ])
        
        # 频率通道注意力 - 修复LayerNorm不兼容问题
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * levels, channels, 1),
            nn.BatchNorm2d(channels),  # 使用批归一化替代层归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, levels, 1),
            nn.Softmax(dim=1)
        )
        
        # 输出融合
        self.fusion = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        # 多级特征分解
        decomposed_feats = [decomposer(x) for decomposer in self.decomposers]
        
        # 计算不同频率级别的权重
        freq_feats = torch.cat(decomposed_feats, dim=1)
        weights = self.freq_attention(freq_feats)
        
        # 加权融合
        weighted_sum = sum(w * feat for w, feat in zip(
            weights.chunk(self.levels, dim=1), 
            decomposed_feats
        ))
        
        # 残差连接
        return self.fusion(weighted_sum) + x


# 异步特征融合器 - 与AMSF的同步融合不同
class AsynchronousFeatureFuser(nn.Module):
    """
    异步融合来自不同尺度的特征
    与AMSF的同步处理方式形成明显对比
    """
    def __init__(self, channels, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or channels // 2
        
        # 低层特征转换 - 使用空间注意力与通道压缩
        self.low_transform = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # 中层特征转换 - 保持原始分辨率
        self.mid_transform = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # 高层特征转换 - 使用上下文增强
        self.high_transform = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # 特征异步融合 - 顺序处理而非AMSF的并行融合
        self.fuse_low_mid = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        self.fuse_all = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # 梯度调节器 - AMSF中不存在
        self.gradient_modulator = nn.Parameter(torch.ones(3) * 0.33)
        self.normalize = lambda x: x / x.sum()
        
    def forward(self, low_feat, mid_feat, high_feat):
        # 确保尺寸一致
        if low_feat.shape[2:] != mid_feat.shape[2:]:
            low_feat = F.interpolate(low_feat, size=mid_feat.shape[2:], mode='bilinear', align_corners=False)
        if high_feat.shape[2:] != mid_feat.shape[2:]:
            high_feat = F.interpolate(high_feat, size=mid_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 特征转换
        weights = self.normalize(self.gradient_modulator)
        low_feat = self.low_transform(low_feat) * weights[0]
        mid_feat = self.mid_transform(mid_feat) * weights[1]
        high_feat = self.high_transform(high_feat) * weights[2]
        
        # 异步融合 - 先融合低+中，再融合与高
        low_mid_feat = self.fuse_low_mid(torch.cat([low_feat, mid_feat], dim=1))
        fused_feat = self.fuse_all(torch.cat([low_mid_feat, high_feat], dim=1))
        
        return fused_feat


class CrossScaleFeatureBridge(nn.Module):
    """
    跨尺度特征桥接模块: 连接不同尺度的特征并融合到中间层
    核心差异：与AMSF是完全不同的架构理念，AMSF关注单尺度特征增强，本模块专注于多尺度特征交互
    """
    def __init__(self, channels, mode='advanced'):
        super().__init__()
        self.channels = channels
        self.mode = mode
        
        # 跨尺度注意力 - 连接高层到中层
        self.high_to_mid_attention = CrossScaleAttention(channels)
        
        # 跨尺度注意力 - 连接低层到中层 
        self.low_to_mid_attention = CrossScaleAttention(channels)
        
        # 特征空间重组 - 用于中层特征
        self.spatial_rearrangement = FeatureSpatialRearrangement(channels)
        
        # 多级特征分解 - 用于处理多尺度融合特征
        self.feature_decomposition = MultiLevelFeatureDecomposition(channels)
        
        # 异步特征融合器 - 最终融合阶段
        self.async_fuser = AsynchronousFeatureFuser(channels)
        
        # 最终特征优化
        self.final_process = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),  # 使用SiLU与AMSF的非线性单元形成区别
            nn.Conv2d(channels, channels, 1)
        )
        
    def forward(self, low_feat, mid_feat, high_feat):
        """
        将低层、中层、高层特征融合到中层特征中
        low_feat: 低层特征 [B, C, H_l, W_l]
        mid_feat: 中层特征 [B, C, H_m, W_m]
        high_feat: 高层特征 [B, C, H_h, W_h]
        """
        # 保存原始中层特征作为残差
        identity = mid_feat
        
        # 1. 跨尺度注意力 - 高层到中层
        high_to_mid = self.high_to_mid_attention(mid_feat, high_feat)
        
        # 2. 跨尺度注意力 - 低层到中层
        low_to_mid = self.low_to_mid_attention(mid_feat, low_feat)
        
        # 3. 特征空间重组 - 重组中层特征
        mid_rearranged = self.spatial_rearrangement(mid_feat)
        
        # 4. 异步特征融合 - 将三个特征层融合在一起
        fused_features = self.async_fuser(low_to_mid, mid_rearranged, high_to_mid)
        
        # 5. 多级特征分解 - 处理融合特征
        enhanced_features = self.feature_decomposition(fused_features)
        
        # 6. 最终处理
        output = self.final_process(enhanced_features)
        
        # 残差连接
        return output + identity
