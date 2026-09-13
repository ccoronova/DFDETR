
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DirectionPredictor', 'DADC']


class DirectionPredictor(nn.Module):
    """Predicts the bounded local direction angle logit ``g_theta(x)``.

    Compresses channels with a 1x1 conv, aggregates local context with a 3x3
    conv (both followed by BatchNorm and GELU), then emits a one-channel
    angle logit. Predicting a bounded angle avoids the unstable normalization
    that can occur when a 2-D direction vector approaches zero.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.logit = nn.Conv2d(channels, 1, 1)  # g_theta(x): B x 1 x H x W

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return self.logit(y)


class DADC(nn.Module):
    """Direction-aware deformable convolution with structured offsets."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.k = kernel_size
        k = kernel_size
        num_pos = k * k
        half = (k - 1) // 2

        # Local dominant-direction predictor.
        self.dir_predictor = DirectionPredictor(channels)

        # Globally learned anisotropic scale, one pair per kernel position,
        # shared across spatial locations and feature scales.
        self.alpha = nn.Parameter(torch.zeros(num_pos))  # parallel scale
        self.beta = nn.Parameter(torch.zeros(num_pos))   # orthogonal scale

        # Deformable convolution weights / bias (gamma is the residual gate).
        self.weight = nn.Parameter(torch.empty(channels, channels, k, k))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.zeros(1))  # zero-init residual gate

        # Base 3x3 grid offsets in (col, row) order, kernel positions row-major.
        cols, rows = [], []
        for ky in range(k):
            for kx in range(k):
                cols.append(float(kx - half))
                rows.append(float(ky - half))
        self.register_buffer('base_col', torch.tensor(cols))
        self.register_buffer('base_row', torch.tensor(rows))

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.dir_predictor(x)                       # B x 1 x H x W
        theta = (torch.pi / 2.0) * torch.tanh(g)        # Eq.(1) bounded angle

        # Parallel d and orthogonal d_perp unit vectors (col, row) components.
        d_x, d_y = theta.cos(), theta.sin()
        p_x, p_y = -theta.sin(), theta.cos()

        s_par = 1.0 + torch.tanh(self.alpha)            # Eq.(4): parallel scale
        s_perp = 1.0 + torch.tanh(self.beta)            # Eq.(4): orthogonal scale

        # Precompute the normalized base sampling grid once.
        # grid_sample expects grid[...,0]=x (col), grid[...,1]=y (row).
        col_idx = torch.linspace(0, W - 1, W, device=x.device, dtype=x.dtype)
        row_idx = torch.linspace(0, H - 1, H, device=x.device, dtype=x.dtype)
        grid_c, grid_r = torch.meshgrid(row_idx, col_idx, indexing='ij')
        col_scale = (2.0 / (W - 1)) if W > 1 else 0.0
        row_scale = (2.0 / (H - 1)) if H > 1 else 0.0
        base_grid = torch.stack([
            grid_c * col_scale - 1.0,  # [..., 0] = normalized x (column)
            grid_r * row_scale - 1.0,  # [..., 1] = normalized y (row)
        ], dim=-1)                     # H x W x 2

        z = torch.zeros_like(x)
        for n in range(self.k * self.k):
            rxc = self.base_col[n]   # base column offset of kernel position n
            ryr = self.base_row[n]   # base row offset of kernel position n

            # Eq.(5): sampling position in the local (d, d_perp) frame, in
            # (col, row) components.
            q_col = s_par[n] * rxc * d_x + s_perp[n] * ryr * p_x  # B,1,H,W
            q_row = s_par[n] * rxc * d_y + s_perp[n] * ryr * p_y  # B,1,H,W

            # Eq.(5): deformable offset relative to the base grid, normalized.
            dcol = (q_col - rxc).squeeze(1) * col_scale   # B,H,W
            drow = (q_row - ryr).squeeze(1) * row_scale   # B,H,W

            # Build the sampling grid for this position and gather samples.
            offset_norm = torch.stack([dcol, drow], dim=-1)  # B,H,W,2
            grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1) + offset_norm
            sample = F.grid_sample(x, grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
            ky, kx = divmod(n, self.k)
            w = self.weight[:, :, ky, kx]                 # C_out x C_in
            # z[p0] += W_n * x[p0 + p_n + d_p_n]   (Eq.(6))
            z = z + torch.einsum('oc,bchw->bohw', w, sample)

        z = z + self.bias.view(C, 1, 1)
        y = x + self.gamma * z                            # Eq.(7): residual gate
        return y