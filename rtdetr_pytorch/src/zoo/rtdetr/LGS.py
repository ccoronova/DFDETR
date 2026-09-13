import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LocalGlobalSynergy']
#多粒度自适应融合模块（MGAF）

#全局上下文注意力（GCA）与局部细节提取器（LDE）
#跨层上下文（CLC），特征融合增强器（FFE）
# 改进1: 高效全局注意力机制（内存优化版）
class EnhancedGlobalAttention(nn.Module):
    def __init__(self, dim, reduction=16):  # 增大了reduction降低内存使用
        super().__init__()
        # 线性投影降维
        self.query = nn.Conv2d(dim, dim//reduction, 1)
        self.key = nn.Conv2d(dim, dim//reduction, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # 添加学习温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
    def forward(self, x):
        # 如果输入特征图太大，先进行下采样降低内存压力
        h, w = x.shape[2:]
        if h * w > 4096:  # 阈值可以根据GPU内存调整
            x_small = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            x_res = x  # 保存原始特征用于残差连接
        else:
            x_small = x
            x_res = x
            
        # 空间维度展平
        b, c = x_small.shape[:2]
        h_s, w_s = x_small.shape[2:]
        
        q = self.query(x_small).view(b, -1, h_s*w_s).permute(0, 2, 1)  # B×HW×C/r
        k = self.key(x_small).view(b, -1, h_s*w_s)  # B×C/r×HW
        v = self.value(x_small).view(b, -1, h_s*w_s).permute(0, 2, 1)  # B×HW×C
        
        # 添加温度参数调节和归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=1)
        
        # 使用高效率版本的注意力机制
        with torch.cuda.amp.autocast(enabled=True):  # 混合精度加速
            context = torch.bmm(q, k) / self.temperature  # B×HW×HW
            attn = F.softmax(context, dim=-1)
            
            # 添加稀疏化 (执行前先释放context以节省内存)
            del context
            attn = attn * (attn > attn.mean(dim=-1, keepdim=True)).float()
            
            out = torch.bmm(attn, v)  # B×HW×C
            
        # 释放注意力矩阵以节省内存
        del attn
        
        out = out.permute(0, 2, 1).view(b, c, h_s, w_s)
        
        # 如果做了下采样，需要上采样回原始尺寸
        if h_s != h or w_s != w:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            
        return self.gamma * out + x_res


# 改进2: 深度可分离卷积优化
class EnhancedLocalFeature(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 深度可分离卷积优化
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pwconv3 = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.dwconv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.pwconv5 = nn.Conv2d(dim, dim, kernel_size=1)
        
        # 添加轻量级SE模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dim//8, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        
    def forward(self, x):
        identity = x
        
        # 多尺度特征提取
        y1 = self.pwconv3(self.dwconv3(x))
        y2 = self.pwconv5(self.dwconv5(x))
        
        # 特征融合
        y = y1 + y2
        y = self.norm(y)
        y = self.act(y)
        
        # 通道注意力
        y = y * self.se(y)
        
        return y + identity


# 改进3: 高级自适应路径选择器
class AdaptivePathSelector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 多尺度特征聚合
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局信息
        
        # 多感受野局部信息提取
        self.local_conv3 = nn.Conv2d(dim, dim//2, kernel_size=3, padding=1, groups=dim//2)  # 3x3感受野
        self.local_conv5 = nn.Conv2d(dim, dim//2, kernel_size=5, padding=2, groups=dim//2)  # 5x5感受野
        
        # 通道注意力机制 (SE模块变体)
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )
        
        # 轻量级上下文建模
        self.context_modeling = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1),  # 降维
            nn.BatchNorm2d(dim//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, dim//8, 3, padding=1, groups=dim//8),  # 深度可分离
            nn.BatchNorm2d(dim//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, dim, 1)  # 升维
        )
        
        # 自适应融合网络 - 更高效的设计
        self.fusion_net = nn.Sequential(
            nn.Conv2d(dim*3, dim//4, 1),  # 减少参数量
            nn.LayerNorm([dim//4, 1, 1]),  # 替代BatchNorm，更稳定
            nn.GELU(),  # 更平滑的激活函数
            nn.Dropout(0.1),  # 增加正则化
            nn.Conv2d(dim//4, 2, 1),  # 两条路径的权重
        )
        
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. 全局信息
        global_info = self.global_pool(x)
        
        # 2. 多感受野局部信息
        local_info3 = self.local_conv3(x)
        local_info5 = self.local_conv5(x)
        local_info = torch.cat([local_info3, local_info5], dim=1)  # 拼接不同感受野特征
        local_info = F.adaptive_avg_pool2d(local_info, 1)
        
        # 3. 通道增强
        channel_attn = self.channel_se(x)
        context_info = self.context_modeling(x * channel_attn)  # 通道增强的上下文信息
        context_info = F.adaptive_avg_pool2d(context_info, 1)
        
        # 4. 融合不同来源的特征
        fused_info = torch.cat([global_info, local_info, context_info], dim=1)
        
        # 5. 生成自适应权重
        weights = self.fusion_net(fused_info)  # [B,2,1,1]
        
        # 6. 温度缩放的Softmax
        weights = F.softmax(weights / self.temperature, dim=1)
        
        return weights  # 返回全局和局部路径的权重


# 改进4: 互补特征增强优化
class ComplementaryFeatureEnhancement(nn.Module):
    def __init__(self, dim, enable_cross_layer=True):
        super().__init__()
        # 特征投影
        self.project = nn.Conv2d(dim*2, dim, 1)
        
        # 门控机制
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim//2, 1),
            nn.BatchNorm2d(dim//2),
            nn.SiLU(),
            nn.Conv2d(dim//2, dim, 1),
        )
        
        # 跨层特征处理
        self.enable_cross_layer = enable_cross_layer
        if enable_cross_layer:
            self.cross_gate = nn.Sequential(
                nn.Conv2d(dim*2, dim, 1),
                nn.Sigmoid()
            )
        
        self.norm = nn.LayerNorm(dim)  # 参数形状调整为单一维度
        
    def forward(self, global_feat, local_feat, weights, cross_layer_feat=None):
        b = global_feat.shape[0]
        weights = weights.view(b, 2, 1, 1)
        
        # 加权融合
        weighted_global = global_feat * weights[:, 0:1, :, :]
        weighted_local = local_feat * weights[:, 1:2, :, :]
        
        # 特征相关性增强
        combined = torch.cat([weighted_global, weighted_local], dim=1)
        
        # 门控融合机制
        gate = torch.sigmoid(self.gate_conv(combined))
        fused = self.project(combined)
        gated_fused = fused * gate
        
        # 跨层特征融合(如果有)
        if self.enable_cross_layer and cross_layer_feat is not None:
            # 确保尺寸一致
            if cross_layer_feat.shape[2:] != gated_fused.shape[2:]:
                cross_layer_feat = F.interpolate(
                    cross_layer_feat, 
                    size=gated_fused.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                
            if cross_layer_feat.shape[1] == gated_fused.shape[1]:  # 确保通道数匹配
                cross_attn = self.cross_gate(torch.cat([gated_fused, cross_layer_feat], dim=1))
                gated_fused = gated_fused + cross_layer_feat * cross_attn
        
        # Layer Norm，正确处理维度
        _, c, h, w = gated_fused.shape
        gated_fused = gated_fused.permute(0, 2, 3, 1)  # [B,H,W,C]
        gated_fused = self.norm(gated_fused)
        gated_fused = gated_fused.permute(0, 3, 1, 2)  # [B,C,H,W]
        
        return gated_fused, gated_fused  # 返回当前输出和可用于跨层的特征


# 改进5: 主模块集成 (内存优化版)
class LocalGlobalSynergy(nn.Module):
    def __init__(self, dim, enable_cross_layer=True):
        super().__init__()
        self.global_branch = EnhancedGlobalAttention(dim)
        self.local_branch = EnhancedLocalFeature(dim)
        self.path_selector = AdaptivePathSelector(dim)
        self.feature_enhancement = ComplementaryFeatureEnhancement(dim, enable_cross_layer)
        
        # 存储上一层的特征，用于跨层交互
        self.enable_cross_layer = enable_cross_layer
        
        # 内存优化参数
        self.training_friendly = True  # 训练时启用内存优化
    
    def forward(self, x, prev_layer_feature=None):
        # 低内存模式检查 - 针对训练阶段
        if self.training and self.training_friendly:
            # 检查特征图大小，如果过大则启用内存优化
            b, c, h, w = x.shape
            large_feature = h * w > 10000  # 可根据GPU内存调整阈值
        else:
            large_feature = False
            
        # 获取自适应权重
        path_weights = self.path_selector(x)
        
        global_features = self.global_branch(x)
        local_features = self.local_branch(x)
        
        # 跨层特征交互
        cross_layer_feat = prev_layer_feature if self.enable_cross_layer else None
        
        # 特征互补增强
        output, current_feat = self.feature_enhancement(
            global_features, 
            local_features, 
            path_weights,
            cross_layer_feat
        )
        
        if self.enable_cross_layer:
            return output, current_feat  # 返回输出和当前特征(用于下一层)
        else:
            return output

