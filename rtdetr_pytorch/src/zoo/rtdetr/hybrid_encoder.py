'''by lyuwenyu
'''
import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from onnx.reference.ops.op_sigmoid import Sigmoid, sigmoid
from sympy.codegen import Print
from torch.ao.nn.quantized import BatchNorm2d
from torch.cuda import device
from torch.nn import MaxPool2d
from torch.nn.functional import max_pool2d, conv2d
from torchgen.api.ufunc import kernel_name
from torch.cuda.amp import autocast  # Mixed precision training
from torch.utils.checkpoint import checkpoint  # Gradient checkpointing
from timm.layers import DropPath  # DropPath regularization
from .utils import get_activation
from src.core import register
from timm.layers import trunc_normal_
import numpy as np  # Import the numpy library
from .LGS import LocalGlobalSynergy  # Import the local-global synergy module
from .DADC import DADC  # Direction-aware deformable convolution (DFDETR)
from .SWFD import SWFD  # Stationary wavelet feature decomposition (DFDETR)
__all__ = ['HybridEncoder']

 # The module combines a conv layer, batch norm, and an activation function: it performs a standard convolution, applies batch norm (BatchNorm) to the output, and finally a nonlinear transform via the activation function (ReLU by default, or a specified one).
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

# ================== Modules merged from AMSF.py below ==================

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
        # Perform edge detection using the predefined weights
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
        texture1 = self.conv1(x) - x  # Extract texture variations
        texture2 = self.conv2(x) - x
        return self.act(self.fusion(torch.cat([texture1, texture2], dim=1)))

# Multi-dimensional Feature Refinement and Fusion Network (AMSF)
class AMSF(nn.Module):
    def __init__(self, in_channels, out_channels, scale_num=3,
                 use_edge_enhancer=True, use_texture_aware=True):
        super(AMSF, self).__init__()
        self.scale_num = scale_num
        self.out_channels = out_channels
        # Ablation switches: edge enhancement / texture awareness, each can be turned off independently for ablation studies
        self.use_edge_enhancer = use_edge_enhancer
        self.use_texture_aware = use_texture_aware

        # Edge enhancement module - enhances PCB trace edge features
        if self.use_edge_enhancer:
            self.edge_enhancer = EdgeEnhancer(in_channels)

        # Texture-aware module - detects PCB surface anomalies
        if self.use_texture_aware:
            self.texture_module = TextureAwareModule(in_channels)

    def forward(self, x, level=None):
        y = x
        # Edge enhancement (ablatable)
        if self.use_edge_enhancer:
            y = self.edge_enhancer(y)

        # Texture awareness (ablatable)
        if self.use_texture_aware:
            texture_feat = self.texture_module(y)
            y = y + texture_feat

        return y

# ================== End of AMSF merge ==================

# Resembles a VGG-like structure with two branches: one 3x3 conv and one 1x1 conv, whose outputs are combined by addition. Purpose: the block mainly boosts the network's expressiveness by enhancing feature representation through its branched structure.
class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
# Combines CSPNet (Cross-Stage Partial Network) and RepVggBlock to perform cross-channel feature interaction on the input features. Purpose: two 1x1 conv branches extract different parts of the input features, which are then fused via multiple RepVggBlocks.
class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)  # conv1 extracts one part of the input features, passed through multiple RepVggBlocks.
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)  # conv2 extracts another part of the original input features.
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)
    
# Replace the original RepVggBlock with a lightweight depthwise-separable-convolution block
class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act='silu'):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)  # Reuse the existing depthwise-separable convolution
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.ds_conv(x))

# Improved CSPRepLayer (does not rely on VGG)
class MultiScaleDSLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu",
                 scales=[3, 5]):  # Multi-scale conv kernels (replacing the original 1x1 branch)
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        
        # Multi-scale feature extraction branch (3x3 and 5x5 convs)
        self.branches = nn.ModuleList([
            ConvNormLayer(in_channels, hidden_channels, kernel_size=k, stride=1, padding=k//2, bias=bias, act=act)
            for k in scales
        ])
        
        # Replace the original RepVggBlock with DSConvBlock (a lightweight depthwise-separable-convolution block)
        self.bottlenecks = nn.Sequential(*[
            DSConvBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        
        # SE attention module (dynamic weighting of branches)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//16, hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Output adjustment
        self.output_conv = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)

    def forward(self, x):
        # Multi-scale branch feature extraction
        branch_feats = [branch(x) for branch in self.branches]
        
        # Branch fusion (sum first, then dynamic weighting via SE)
        fused_feat = sum(branch_feats)
        fused_feat = fused_feat * self.se(fused_feat)  # SE attention weighting

        # Lightweight block processing
        x_1 = self.bottlenecks(fused_feat)
        
        # Output adjustment (keeping the number of channels consistent)
        return self.output_conv(x_1)   
# transformer: these modules implement the Transformer encoder layer (self-attention) and the Transformer encoder. Function: perform within-scale interaction on image features, using self-attention to compute correlations between positions to capture global dependencies.
# Intra-scale fusion in the self-attention (Transformer) mechanism
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)  # This embodies intra-scale interaction: multi-head attention learns correlations among positions of the feature map, enabling global feature interaction.

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.activation(x)
  


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[128, 256, 512],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 use_amsf=False,           # Legacy SurfDETR enhancement (edge + texture), disabled by default in DFDETR
                 use_edge_enhancer=True,   # AMSF ablation switch: edge enhancement
                 use_texture_aware=True,   # AMSF ablation switch: texture awareness
                 use_dadc=True,            # DFDETR: direction-aware deformable convolution
                 use_swfd=True,            # DFDETR: stationary wavelet feature decomposition
                 lgs_level_idx=[0,1,2],  # By default, apply LGS only to the deepest features
                ):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.use_amsf = use_amsf
        self.use_edge_enhancer = use_edge_enhancer
        self.use_texture_aware = use_texture_aware
        self.use_dadc = use_dadc
        self.use_swfd = use_swfd

        self.lgs_level_idx = lgs_level_idx  # Record the level indices where LGS should be applied
        # Initialize the AMS-F module
        if self.use_amsf:
            self.amsf = AMSF(
                in_channels=hidden_dim,  # Number of input channels
                out_channels=hidden_dim,  # Number of output channels (adjustable)
                scale_num=len(in_channels),  # Number of multi-scale features
                use_edge_enhancer=use_edge_enhancer,  # Ablation: edge enhancement
                use_texture_aware=use_texture_aware,  # Ablation: texture awareness
            )

        # Initialize the DFDETR modules: DADC and SWFD
        if self.use_dadc:
            self.dadc = DADC(hidden_dim)
        if self.use_swfd:
            self.swfd = SWFD(hidden_dim)
        
    # Apply LGS with varying strength to features at different levels
    
        #Input channel and feature-map preprocessing: this part projects the channel count of each input feature map from its original size (e.g., 512, 1024, 2048) to a fixed hidden dimension of 256 (hidden_dim) via 1x1 convolution.
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
  # encoder Transformer encoder. Function: according to the given use_encoder_idx, a Transformer encoder processes the feature maps at different scales, enabling within-scale feature interaction.
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)
        self.encoder = nn.ModuleList([
    #range generates the corresponding integer sequence based on the length value. Combined with for, the loop repeatedly executes the transformer.
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])
  # top-down fpn
 #Feature pyramid network (FPN) inter-scale fusion: FPN fuses features from high to low level via upsampling and cross-scale concatenation to enhance features.
        #Lateral connection conv layer, used to map higher-level features to the same number of channels.
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        #2, 1; the loop runs at index 1
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion))

 # bottom-up PAN (Path Aggregation Networks): propagates information from low to high level, helping the model better understand the features of large objects.
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        # 2，1
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
        self._reset_parameters()
        # self.learnable_pos_embed = nn.ParameterList()
        for idx in self.use_encoder_idx:
             # Assume eval_spatial_size is known
            stride = self.feat_strides[idx]
            h, w = self.eval_spatial_size[0] // stride, self.eval_spatial_size[1] // stride
            # self.learnable_pos_embed.append(
            #     nn.Parameter(torch.zeros(1, h * w, self.hidden_dim))
            # )    
    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                #Position encoding embedding. Function: to strengthen positional information inside the Transformer, the model builds a sine/cosine based 2D position embedding (sinusoidal position embedding), adds it to the feature maps, and passes it to the Transformer encoder.
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        # print([feat.shape for feat in feats])   # Print the input/output shapes
        assert len(feats) == len(self.in_channels)
#Each input feature is processed by the projection layer in self.input_proj, and the results are stored in the proj_feats list. The channel count becomes 256; everything else stays unchanged.
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
 
        if self.use_amsf:
        #     # proj_feats[2]  = self.amsf(proj_feats[2])             #process only the last layer element
            proj_feats = [self.amsf(feat, level = idx) for idx, feat in enumerate(proj_feats)]   # Iterate over all elements

        if self.use_dadc:
            # DADC: direction-aware deformable convolution (aligns the local trace direction at each scale)
            proj_feats = [self.dadc(feat) for feat in proj_feats]

        if self.use_swfd:
            # SWFD: stationary wavelet feature decomposition (HH-guided multi-band enhancement, preserving resolution)
            proj_feats = [self.swfd(feat) for feat in proj_feats]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]. Flatten the feature map [B, C, H, W] into [B, HxW, C].
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
    # If the model is in training mode, or no spatial size is preset, build the 2D sine-cosine position embedding.
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
    # Retrieve the pos_embed attribute stored on self using the enc_ind index, and move it to ---
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                    # pos_embed = self.learnable_pos_embed[i].to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
            # The encoder processes the features
    # Pass the flattened features and the position embedding to the encoder layer. (b, hxw, c)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
    # Reshape the encoder output back to the original feature map shape.
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print("hybridencoder-1",[x.is_contiguous() for x in proj_feats])
        
        # if self.num_encoder_layers > 0:
        #     for i, enc_ind in enumerate(self.use_encoder_idx):
        #         h, w = proj_feats[enc_ind].shape[2:]
        #         src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
        #             # Use the learnable position encoding directly
                # pos_embed = self.learnable_pos_embed[i].to(src_flatten.device)
        #         memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
        #         proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        #     # ...the rest of the code remains unchanged...
        #FPN fusion
  #High-level features, equivalent to proj_feats[2]
        inner_outs = [proj_feats[-1]]
        # print([feat.shape for feat in proj_feats[-1]])
    # Fuse features from the high level to the low level. range(2, 0, -1) uses indices 2 and 1
        for idx in range(len(self.in_channels) - 1, 0, -1):
    # Get s5. inner_outs always points to the highest-level feature map.
            feat_high = inner_outs[0]
    # Get s4.
            feat_low = proj_feats[idx - 1]
        # Process s5 through the lateral conv layer.
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
    # Upsample the high-level feature maps.
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
    # Fuse the upsampled high-level features and the low-level features (feature maps from different sources) in the feature pyramid network (FPN) block. Cross-scale fusion.
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)  # After each step, insert into the front so the feature map order stays consistent with the original network structure.
             

        outs = [inner_outs[0]]
        # print("1", inner_outs[-1].shape) high/mid/low

        #Cross-scale fusion operation, inside the PAN
    #0,1
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            # print("4",outs[-1].shape)
            feat_high = inner_outs[idx + 1]
            # Downsample feat_low
            downsample_feat = self.downsample_convs[idx](feat_low)
            #Concatenate the downsampled feature map (downsample_feat) with the higher-resolution feature map (feat_high), then pass them to pan_blocks. dim=1 signifies concatenation along the channel dimension; by downsampling and concatenating features of different resolutions, intra-scale interaction is achieved.
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))  # Cross-scale fusion
            #Add the currently processed output feature map (out) to the outs list.
            outs.append(out)   #PAN output
        #
        # if self.use_gpa:
        #     outs = [self.amsf(out) for out in outs]

        return outs
# print([feat.shape for feat in feats]) Print the input/output shapes


