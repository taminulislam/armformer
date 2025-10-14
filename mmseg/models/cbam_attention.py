import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmseg.registry import MODELS


class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM."""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


@MODELS.register_module()
class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    CBAM combines channel attention and spatial attention to improve
    feature representation in convolutional neural networks.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for channel attention. Default: 16.
        kernel_size (int): Kernel size for spatial attention. Default: 7.
    """

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x = x * channel_att

        spatial_att = self.spatial_attention(x)
        x = x * spatial_att

        return x


@MODELS.register_module()
class CBAMBlock(nn.Module):
    """CBAM attention block that can be inserted into neural networks.

    This block applies CBAM attention after a convolutional layer or
    transformer block to enhance feature representation.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for channel attention. Default: 16.
        kernel_size (int): Kernel size for spatial attention. Default: 7.
        norm_cfg (dict): Normalization config. Default: dict(type='BN').
        act_cfg (dict): Activation config. Default: dict(type='ReLU').
    """

    def __init__(self,
                 in_channels,
                 reduction_ratio=16,
                 kernel_size=7,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(CBAMBlock, self).__init__()
        self.cbam = CBAM(in_channels, reduction_ratio, kernel_size)
        self.norm = build_norm_layer(norm_cfg, in_channels)[1] if norm_cfg else None

    def forward(self, x):
        identity = x

        x = self.cbam(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


def build_cbam_block(channels, reduction_ratio=16, kernel_size=7):
    """Helper function to build CBAM block with default configurations."""
    return CBAMBlock(
        in_channels=channels,
        reduction_ratio=reduction_ratio,
        kernel_size=kernel_size
    )

