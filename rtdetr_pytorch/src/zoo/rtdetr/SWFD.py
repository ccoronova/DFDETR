
import torch
import torch.nn as nn

__all__ = ['SWFD']


class SWFD(nn.Module):
    """Downsampling-free, multi-band feature decomposition with HH-guided
    learnable enhancement."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels

        # Fixed Haar filter bank, channel-wise (groups=C), resolution-preserving.
        self.ll = self._haar_basis(channels, [[0.5, 0.5], [0.5, 0.5]])
        self.lh = self._haar_basis(channels, [[0.5, 0.5], [-0.5, -0.5]])
        self.hl = self._haar_basis(channels, [[0.5, -0.5], [0.5, -0.5]])
        self.hh = self._haar_basis(channels, [[0.5, -0.5], [-0.5, 0.5]])

        # Fuse the 4C-channel subband tensor back to C channels (Eq.(12)).
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Lightweight two-layer branch: HH -> A_HH in [0,1].
        mid = max(channels // reduction, 1)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

        self.delta = nn.Parameter(torch.zeros(1))  # learnable sign (Eq.(12))
        self.gamma = nn.Parameter(torch.zeros(1))  # zero-init residual gate (Eq.(13))

    @staticmethod
    def _haar_basis(channels, kernel):
        """Fixed 2x2 Haar filter applied with dilation 2 / stride 1 / pad 1,
        keeping the spatial size unchanged."""
        conv = nn.Conv2d(channels, channels, kernel_size=2, stride=1,
                         padding=1, dilation=2, groups=channels, bias=False)
        k = torch.tensor(kernel).reshape(1, 1, 2, 2).repeat(channels, 1, 1, 1)
        with torch.no_grad():
            conv.weight.copy_(k)
        conv.weight.requires_grad = False  # fixed basis, not learned
        return conv

    def forward(self, x):
        ll = self.ll(x)  # smooth structure
        lh = self.lh(x)  # horizontal boundaries
        hl = self.hl(x)  # vertical boundaries
        hh = self.hh(x)  # joint high-frequency detail

        # Concatenating all subbands retains the full frequency representation.
        subbands = torch.cat([ll, lh, hl, hh], dim=1)  # B x 4C x H x W
        fused = self.fuse(subbands)                    # B x C x H x W

        a_hh = self.attn(hh)                            # B x C x H x W in [0,1]
        enh = fused * (1.0 + self.delta * a_hh)         # Eq.(12)
        out = x + self.gamma * enh                      # Eq.(13): residual gate
        return out