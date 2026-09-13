import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LightCrossScaleFeatureBridge']


# Lightweight cross-scale attention module
class LightCrossAttention(nn.Module):
    """
    Lightweight cross-scale attention: reduces computation and the number of parameters
    Reduces parameters by sharing projections and using depthwise separable convolution
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channels = in_channels
        self.reduction_channels = max(in_channels // reduction_ratio, 8)

        # Shared projection matrix reduces the number of parameters
        self.qkv_proj = nn.Sequential(
            nn.Conv2d(in_channels, self.reduction_channels * 2 + in_channels,
                     kernel_size=1, groups=4),  # use grouped convolution to reduce parameters
            nn.BatchNorm2d(self.reduction_channels * 2 + in_channels)
        )

        # Lightweight output projection
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=4)

        # Scaling factor
        self.scale = self.reduction_channels ** -0.5

    def forward(self, x_target, x_source):
        batch, _, h_t, w_t = x_target.shape

        # Use the shared projection to compute Q for the target features and K, V for the source features
        target_qkv = self.qkv_proj(x_target)
        q = target_qkv[:, :self.reduction_channels]

        source_qkv = self.qkv_proj(x_source)
        k = source_qkv[:, :self.reduction_channels]
        v = source_qkv[:, self.reduction_channels*2:]

        # Flatten to compute the attention
        q = q.flatten(2)  # B, C', H_t*W_t
        k = k.flatten(2)  # B, C', H_s*W_s
        v = v.flatten(2)  # B, C, H_s*W_s

        # Efficient attention computation - uses matrix multiplication
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # B, H_t*W_t, H_s*W_s
        attn = F.softmax(attn, dim=-1)

        # Apply the attention weights
        out = torch.bmm(attn, v.transpose(1, 2))  # B, H_t*W_t, C
        out = out.transpose(1, 2).reshape(batch, -1, h_t, w_t)

        # Lightweight output projection
        out = self.out_proj(out)

        return out


# Lightweight feature enhancement module
class LightFeatureEnhancement(nn.Module):
    """
    Lightweight feature enhancement: replaces the previous spatial rearrangement module
    Uses depthwise separable convolution and fewer groups
    """
    def __init__(self, channels, num_groups=4):
        super().__init__()
        # Use fewer groups to reduce the number of parameters
        self.num_groups = num_groups

        # Depthwise separable convolution - reduces the number of parameters
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

        # Channel attention - enhances the important features
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Feature enhancement - uses depthwise separable convolution
        feat = self.depthwise(x)
        feat = self.pointwise(feat)
        feat = self.act(self.norm(feat))

        # Channel attention
        attention = self.channel_attention(feat)
        enhanced = feat * attention

        # Residual connection
        return enhanced + x


# Lightweight feature fusion module
class LightFeatureFuser(nn.Module):
    """
    Lightweight feature fusion module: replaces the asynchronous feature fuser.
    Uses addition instead of concatenation to reduce the number of parameters in later processing
    """
    def __init__(self, channels):
        super().__init__()
        # Feature transform - uses grouped convolution to reduce parameters
        self.transform = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, groups=4),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])

        # Feature weighting - learns the importance of the different features
        self.weights = nn.Parameter(torch.ones(3) / 3)

        # Output projection - uses depthwise separable convolution
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2, feat3):
        # Ensure consistent spatial dimensions
        size = feat2.shape[2:]  # use the middle-layer feature's size
        if feat1.shape[2:] != size:
            feat1 = F.interpolate(feat1, size=size, mode='bilinear', align_corners=False)
        if feat3.shape[2:] != size:
            feat3 = F.interpolate(feat3, size=size, mode='bilinear', align_corners=False)

        # Feature transform
        feats = [transform(feat) for transform, feat in zip(self.transform, [feat1, feat2, feat3])]

        # Weighted sum - uses fewer parameters than concatenation
        weights = F.softmax(self.weights, dim=0)
        fused = sum(w * f for w, f in zip(weights, feats))

        # Output projection
        return self.out_proj(fused)


class LightCrossScaleFeatureBridge(nn.Module):
    """
    Lightweight cross-scale feature bridge module:
    dramatically reduces the number of parameters while preserving the core function of multi-scale feature fusion
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Cross-scale attention - keeps only the two key attention modules
        self.high_to_mid_attention = LightCrossAttention(channels)
        self.low_to_mid_attention = LightCrossAttention(channels)

        # Middle-layer feature enhancement - uses a lightweight module instead of spatial rearrangement
        self.feature_enhancement = LightFeatureEnhancement(channels)

        # Feature fusion - uses a lightweight fusion module
        self.feature_fuser = LightFeatureFuser(channels)

        # Final processing - uses depthwise separable convolution
        self.final_process = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_feat, mid_feat, high_feat):
        """
        Fuses the low, middle, and high layer features into the middle-layer features
        A more efficient implementation that significantly reduces the number of parameters and computation
        """
        # Save the original features for the residual connection
        identity = mid_feat

        # 1. Cross-scale attention interaction
        high_to_mid = self.high_to_mid_attention(mid_feat, high_feat)
        low_to_mid = self.low_to_mid_attention(mid_feat, low_feat)

        # 2. Middle-layer feature enhancement
        mid_enhanced = self.feature_enhancement(mid_feat)

        # 3. Feature fusion - fuse the three feature streams together
        fused = self.feature_fuser(low_to_mid, mid_enhanced, high_to_mid)

        # 4. Final processing
        output = self.final_process(fused)

        # 5. Residual connection - ensures a smooth flow of information
        return output + identity
