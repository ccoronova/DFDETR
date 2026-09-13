import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CrossScaleFeatureBridge']


# Cross-scale attention module - a fully different attention mechanism from AMSF
class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention module: establishes associations between features at different scales.
    Unlike AMSF, this module focuses on cross-scale interaction rather than single-scale feature enhancement
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channels = in_channels
        self.reduction_channels = max(in_channels // reduction_ratio, 8)

        # Query transform - used for the target-layer features
        self.query_conv = nn.Conv2d(in_channels, self.reduction_channels, kernel_size=1)

        # Key/value transform - used for the source-layer features
        self.key_conv = nn.Conv2d(in_channels, self.reduction_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Scaling factor
        self.scale = nn.Parameter(torch.ones(1) * self.reduction_channels ** -0.5)

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x_target, x_source):
        """
        x_target: target features, the object on which attention is applied [B, C, H, W]
        x_source: source features, the source that provides the attention information [B, C, H', W']
        """
        batch, _, h_t, w_t = x_target.shape
        _, _, h_s, w_s = x_source.shape

        # Generate query, key, and value
        q = self.query_conv(x_target).view(batch, self.reduction_channels, -1)  # B, C', H_t*W_t
        k = self.key_conv(x_source).view(batch, self.reduction_channels, -1)    # B, C', H_s*W_s
        v = self.value_conv(x_source).view(batch, self.channels, -1)             # B, C, H_s*W_s

        # Compute attention scores (an attention computation fully different from AMSF)
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale  # B, H_t*W_t, H_s*W_s
        attn = F.softmax(attn, dim=-1)  # normalize the attention weights

        # Apply the attention weights
        out = torch.bmm(v, attn.permute(0, 2, 1))  # B, C, H_t*W_t
        out = out.view(batch, self.channels, h_t, w_t)
        out = self.proj(out)

        return out


# Feature spatial rearrangement module - a component that AMSF does not have
class FeatureSpatialRearrangement(nn.Module):
    """
    Feature spatial rearrangement: rearranges the spatial relationships of feature maps.
    Completely different from AMSF's handling, it emphasizes transformations in the spatial dimension
    """
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.num_groups = num_groups
        assert channels % num_groups == 0, "the number of channels must be divisible by the number of groups"

        # Feature grouping
        group_channels = channels // num_groups
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1, groups=group_channels),
                nn.InstanceNorm2d(group_channels),  # use instance normalization instead of batch normalization (AMSF)
                nn.GELU()  # GELU differentiates from AMSF's ReLU
            )
            for _ in range(num_groups)
        ])

        # Spatial rearrangement mixer - fixes the LayerNorm incompatibility issue
        self.mixer = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),  # use BatchNorm instead of LayerNorm to avoid dimension errors
            nn.ReLU(inplace=True),     # use ReLU instead, to distinguish from AMSF
            nn.Dropout2d(0.1)          # add spatial Dropout
        )

    def forward(self, x):
        B, C, H, W = x.shape
        group_size = C // self.num_groups

        # Channel grouping
        groups = torch.split(x, group_size, dim=1)
        processed_groups = []

        # Group-wise processing
        for i, (group, conv) in enumerate(zip(groups, self.group_convs)):
            # Apply a different transformation to each group
            if i % 4 == 0:  # horizontal flip
                group = torch.flip(group, [3])
            elif i % 4 == 1:  # vertical flip
                group = torch.flip(group, [2])
            elif i % 4 == 2:  # rotation
                group = torch.rot90(group, 1, [2, 3])

            processed = conv(group)

            # Restore the original arrangement
            if i % 4 == 0:
                processed = torch.flip(processed, [3])
            elif i % 4 == 1:
                processed = torch.flip(processed, [2])
            elif i % 4 == 2:
                processed = torch.rot90(processed, -1, [2, 3])

            processed_groups.append(processed)

        # The rearranged features
        rearranged_feat = torch.cat(processed_groups, dim=1)
        mixed_feat = self.mixer(rearranged_feat)

        return mixed_feat + x  # residual connection


# Multi-level feature decomposition module - a decomposition clearly different from AMSF
class MultiLevelFeatureDecomposition(nn.Module):
    """
    Decomposes a feature into multiple components, each capturing information at a different frequency.
    Completely different from AMSF's edge detection and texture analysis approach
    """
    def __init__(self, channels, levels=3):
        super().__init__()
        self.levels = levels

        # Multi-level feature extractors that use different dilation rates to capture different scales
        self.decomposers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2**i, dilation=2**i, groups=channels),
                nn.GroupNorm(channels // 16, channels),  # use group normalization instead of AMSF's batch normalization
                nn.SiLU()  # use the SiLU activation, different from AMSF's GELU
            )
            for i in range(levels)
        ])

        # Frequency channel attention - fixes the LayerNorm incompatibility issue
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * levels, channels, 1),
            nn.BatchNorm2d(channels),  # use batch normalization instead of layer normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, levels, 1),
            nn.Softmax(dim=1)
        )

        # Output fusion
        self.fusion = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # Multi-level feature decomposition
        decomposed_feats = [decomposer(x) for decomposer in self.decomposers]

        # Compute the weights of the different frequency levels
        freq_feats = torch.cat(decomposed_feats, dim=1)
        weights = self.freq_attention(freq_feats)

        # Weighted fusion
        weighted_sum = sum(w * feat for w, feat in zip(
            weights.chunk(self.levels, dim=1),
            decomposed_feats
        ))

        # Residual connection
        return self.fusion(weighted_sum) + x


# Asynchronous feature fuser - in contrast to AMSF's synchronous fusion
class AsynchronousFeatureFuser(nn.Module):
    """
    Asynchronously fuses features from different scales
    Forming a clear contrast with AMSF's synchronous handling
    """
    def __init__(self, channels, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or channels // 2

        # Low-level feature transform - uses spatial attention and channel compression
        self.low_transform = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # Mid-level feature transform - preserves the original resolution
        self.mid_transform = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # High-level feature transform - uses context enhancement
        self.high_transform = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # Asynchronous feature fusion - sequential processing rather than AMSF's parallel fusion
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

        # Gradient modulator - does not exist in AMSF
        self.gradient_modulator = nn.Parameter(torch.ones(3) * 0.33)
        self.normalize = lambda x: x / x.sum()

    def forward(self, low_feat, mid_feat, high_feat):
        # Ensure consistent dimensions
        if low_feat.shape[2:] != mid_feat.shape[2:]:
            low_feat = F.interpolate(low_feat, size=mid_feat.shape[2:], mode='bilinear', align_corners=False)
        if high_feat.shape[2:] != mid_feat.shape[2:]:
            high_feat = F.interpolate(high_feat, size=mid_feat.shape[2:], mode='bilinear', align_corners=False)

        # Feature transform
        weights = self.normalize(self.gradient_modulator)
        low_feat = self.low_transform(low_feat) * weights[0]
        mid_feat = self.mid_transform(mid_feat) * weights[1]
        high_feat = self.high_transform(high_feat) * weights[2]

        # Asynchronous fusion - first fuse low+mid, then fuse with high
        low_mid_feat = self.fuse_low_mid(torch.cat([low_feat, mid_feat], dim=1))
        fused_feat = self.fuse_all(torch.cat([low_mid_feat, high_feat], dim=1))

        return fused_feat


class CrossScaleFeatureBridge(nn.Module):
    """
    Cross-scale feature bridge module: connects features at different scales and fuses them into the middle layer.
    Core difference: this is a completely different architectural concept from AMSF. AMSF focuses on single-scale feature enhancement, while this module focuses on multi-scale feature interaction
    """
    def __init__(self, channels, mode='advanced'):
        super().__init__()
        self.channels = channels
        self.mode = mode

        # Cross-scale attention - connects the high layer to the middle layer
        self.high_to_mid_attention = CrossScaleAttention(channels)

        # Cross-scale attention - connects the low layer to the middle layer
        self.low_to_mid_attention = CrossScaleAttention(channels)

        # Feature spatial rearrangement - used for the middle-layer features
        self.spatial_rearrangement = FeatureSpatialRearrangement(channels)

        # Multi-level feature decomposition - used to process the multi-scale fused features
        self.feature_decomposition = MultiLevelFeatureDecomposition(channels)

        # Asynchronous feature fuser - the final fusion stage
        self.async_fuser = AsynchronousFeatureFuser(channels)

        # Final feature refinement
        self.final_process = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),  # use SiLU to differentiate from AMSF's nonlinear units
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, low_feat, mid_feat, high_feat):
        """
        Fuses the low, middle, and high layer features into the middle-layer features
        low_feat: low-level features [B, C, H_l, W_l]
        mid_feat: middle-level features [B, C, H_m, W_m]
        high_feat: high-level features [B, C, H_h, W_h]
        """
        # Save the original middle-layer features as the residual
        identity = mid_feat

        # 1. Cross-scale attention - high layer to middle layer
        high_to_mid = self.high_to_mid_attention(mid_feat, high_feat)

        # 2. Cross-scale attention - low layer to middle layer
        low_to_mid = self.low_to_mid_attention(mid_feat, low_feat)

        # 3. Feature spatial rearrangement - rearrange the middle-layer features
        mid_rearranged = self.spatial_rearrangement(mid_feat)

        # 4. Asynchronous feature fusion - fuse the three feature layers together
        fused_features = self.async_fuser(low_to_mid, mid_rearranged, high_to_mid)

        # 5. Multi-level feature decomposition - process the fused features
        enhanced_features = self.feature_decomposition(fused_features)

        # 6. Final processing
        output = self.final_process(enhanced_features)

        # Residual connection
        return output + identity
