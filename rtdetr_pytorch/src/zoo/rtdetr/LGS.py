import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LocalGlobalSynergy']
# Multi-granularity adaptive fusion module (MGAF)

# Global context attention (GCA) and local detail extractor (LDE)
# Cross-layer context (CLC), feature fusion enhancer (FFE)
# Improvement 1: Efficient global attention mechanism (memory-optimized)
class EnhancedGlobalAttention(nn.Module):
    def __init__(self, dim, reduction=16):  # Larger reduction lowers memory usage
        super().__init__()
        # Linear projections for dimensionality reduction
        self.query = nn.Conv2d(dim, dim//reduction, 1)
        self.key = nn.Conv2d(dim, dim//reduction, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x):
        # If the input feature map is too large, downsample it first to reduce memory pressure
        h, w = x.shape[2:]
        if h * w > 4096:  # The threshold can be adjusted based on GPU memory
            x_small = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            x_res = x  # Keep the original features for the residual connection
        else:
            x_small = x
            x_res = x

        # Flatten the spatial dimensions
        b, c = x_small.shape[:2]
        h_s, w_s = x_small.shape[2:]

        q = self.query(x_small).view(b, -1, h_s*w_s).permute(0, 2, 1)  # B x HW x C/r
        k = self.key(x_small).view(b, -1, h_s*w_s)  # B x C/r x HW
        v = self.value(x_small).view(b, -1, h_s*w_s).permute(0, 2, 1)  # B x HW x C

        # Apply temperature scaling and normalization
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=1)

        # Use an efficient version of the attention mechanism
        with torch.cuda.amp.autocast(enabled=True):  # Mixed-precision acceleration
            context = torch.bmm(q, k) / self.temperature  # B x HW x HW
            attn = F.softmax(context, dim=-1)

            # Apply sparsification (release context first to save memory)
            del context
            attn = attn * (attn > attn.mean(dim=-1, keepdim=True)).float()

            out = torch.bmm(attn, v)  # B x HW x C

        # Release the attention matrix to save memory
        del attn

        out = out.permute(0, 2, 1).view(b, c, h_s, w_s)

        # If downsampled, upsample back to the original size
        if h_s != h or w_s != w:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return self.gamma * out + x_res


# Improvement 2: Depthwise separable convolution optimization
class EnhancedLocalFeature(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Depthwise separable convolution
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pwconv3 = nn.Conv2d(dim, dim, kernel_size=1)

        self.dwconv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.pwconv5 = nn.Conv2d(dim, dim, kernel_size=1)

        # Add a lightweight SE module
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

        # Multi-scale feature extraction
        y1 = self.pwconv3(self.dwconv3(x))
        y2 = self.pwconv5(self.dwconv5(x))

        # Feature fusion
        y = y1 + y2
        y = self.norm(y)
        y = self.act(y)

        # Channel attention
        y = y * self.se(y)

        return y + identity


# Improvement 3: Advanced adaptive path selector
class AdaptivePathSelector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Multi-scale feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global information

        # Extract local information from multiple receptive fields
        self.local_conv3 = nn.Conv2d(dim, dim//2, kernel_size=3, padding=1, groups=dim//2)  # 3x3 receptive field
        self.local_conv5 = nn.Conv2d(dim, dim//2, kernel_size=5, padding=2, groups=dim//2)  # 5x5 receptive field

        # Channel attention mechanism (SE module variant)
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )

        # Lightweight context modeling
        self.context_modeling = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1),  # Reduce channels
            nn.BatchNorm2d(dim//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, dim//8, 3, padding=1, groups=dim//8),  # Depthwise separable
            nn.BatchNorm2d(dim//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, dim, 1)  # Increase channels
        )

        # Adaptive fusion network - more efficient design
        self.fusion_net = nn.Sequential(
            nn.Conv2d(dim*3, dim//4, 1),  # Reduce parameter count
            nn.LayerNorm([dim//4, 1, 1]),  # Replaces BatchNorm, more stable
            nn.GELU(),  # Smoother activation function
            nn.Dropout(0.1),  # Add regularization
            nn.Conv2d(dim//4, 2, 1),  # Weights for the two paths
        )

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        batch_size = x.shape[0]

        # 1. Global information
        global_info = self.global_pool(x)

        # 2. Multi-receptive-field local information
        local_info3 = self.local_conv3(x)
        local_info5 = self.local_conv5(x)
        local_info = torch.cat([local_info3, local_info5], dim=1)  # Concatenate features from different receptive fields
        local_info = F.adaptive_avg_pool2d(local_info, 1)

        # 3. Channel enhancement
        channel_attn = self.channel_se(x)
        context_info = self.context_modeling(x * channel_attn)  # Channel-enhanced context information
        context_info = F.adaptive_avg_pool2d(context_info, 1)

        # 4. Fuse features from different sources
        fused_info = torch.cat([global_info, local_info, context_info], dim=1)

        # 5. Generate adaptive weights
        weights = self.fusion_net(fused_info)  # [B,2,1,1]

        # 6. Temperature-scaled softmax
        weights = F.softmax(weights / self.temperature, dim=1)

        return weights  # Return the weights for the global and local paths


# Improvement 4: Complementary feature enhancement
class ComplementaryFeatureEnhancement(nn.Module):
    def __init__(self, dim, enable_cross_layer=True):
        super().__init__()
        # Feature projection
        self.project = nn.Conv2d(dim*2, dim, 1)

        # Gating mechanism
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim//2, 1),
            nn.BatchNorm2d(dim//2),
            nn.SiLU(),
            nn.Conv2d(dim//2, dim, 1),
        )

        # Cross-layer feature processing
        self.enable_cross_layer = enable_cross_layer
        if enable_cross_layer:
            self.cross_gate = nn.Sequential(
                nn.Conv2d(dim*2, dim, 1),
                nn.Sigmoid()
            )

        self.norm = nn.LayerNorm(dim)  # Adjust parameter shape to a single dimension

    def forward(self, global_feat, local_feat, weights, cross_layer_feat=None):
        b = global_feat.shape[0]
        weights = weights.view(b, 2, 1, 1)

        # Weighted fusion
        weighted_global = global_feat * weights[:, 0:1, :, :]
        weighted_local = local_feat * weights[:, 1:2, :, :]

        # Correlated feature enhancement
        combined = torch.cat([weighted_global, weighted_local], dim=1)

        # Gated fusion mechanism
        gate = torch.sigmoid(self.gate_conv(combined))
        fused = self.project(combined)
        gated_fused = fused * gate

        # Cross-layer feature fusion (if available)
        if self.enable_cross_layer and cross_layer_feat is not None:
            # Ensure matching spatial size
            if cross_layer_feat.shape[2:] != gated_fused.shape[2:]:
                cross_layer_feat = F.interpolate(
                    cross_layer_feat,
                    size=gated_fused.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            if cross_layer_feat.shape[1] == gated_fused.shape[1]:  # Ensure matching channel count
                cross_attn = self.cross_gate(torch.cat([gated_fused, cross_layer_feat], dim=1))
                gated_fused = gated_fused + cross_layer_feat * cross_attn

        # Layer norm, handling dimensions correctly
        _, c, h, w = gated_fused.shape
        gated_fused = gated_fused.permute(0, 2, 3, 1)  # [B,H,W,C]
        gated_fused = self.norm(gated_fused)
        gated_fused = gated_fused.permute(0, 3, 1, 2)  # [B,C,H,W]

        return gated_fused, gated_fused  # Return the current output and the feature usable for cross-layer interaction


# Improvement 5: Main module integration (memory-optimized)
class LocalGlobalSynergy(nn.Module):
    def __init__(self, dim, enable_cross_layer=True):
        super().__init__()
        self.global_branch = EnhancedGlobalAttention(dim)
        self.local_branch = EnhancedLocalFeature(dim)
        self.path_selector = AdaptivePathSelector(dim)
        self.feature_enhancement = ComplementaryFeatureEnhancement(dim, enable_cross_layer)

        # Store the previous layer's feature for cross-layer interaction
        self.enable_cross_layer = enable_cross_layer

        # Memory optimization parameters
        self.training_friendly = True  # Enable memory optimization during training

    def forward(self, x, prev_layer_feature=None):
        # Low-memory mode check - for the training phase
        if self.training and self.training_friendly:
            # Check feature map size; enable memory optimization if too large
            b, c, h, w = x.shape
            large_feature = h * w > 10000  # Adjust this threshold based on GPU memory
        else:
            large_feature = False

        # Get adaptive weights
        path_weights = self.path_selector(x)

        global_features = self.global_branch(x)
        local_features = self.local_branch(x)

        # Cross-layer feature interaction
        cross_layer_feat = prev_layer_feature if self.enable_cross_layer else None

        # Complementary feature enhancement
        output, current_feat = self.feature_enhancement(
            global_features,
            local_features,
            path_weights,
            cross_layer_feat
        )

        if self.enable_cross_layer:
            return output, current_feat  # Return the output and current feature (for the next layer)
        else:
            return output