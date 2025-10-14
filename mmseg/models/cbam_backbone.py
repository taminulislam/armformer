import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.backbones.mit import MixVisionTransformer
from mmseg.registry import MODELS
from .cbam_attention import CBAMBlock


@MODELS.register_module()
class CBAMMixVisionTransformer(nn.Module):
    """MixVisionTransformer with CBAM attention blocks.

    This backbone extends the standard MixVisionTransformer by adding
    CBAM attention modules after each transformer stage to enhance
    feature representation for better segmentation performance.

    Args:
        backbone_cfg (dict): Configuration for the base MixVisionTransformer.
        cbam_cfg (dict): Configuration for CBAM blocks. Default: None.
        reduction_ratios (list): Reduction ratios for each stage's CBAM.
            Default: [16, 16, 16, 16].
        kernel_sizes (list): Kernel sizes for spatial attention in each stage.
            Default: [7, 7, 7, 7].
    """

    def __init__(self,
                 backbone_cfg,
                 cbam_cfg=None,
                 reduction_ratios=[16, 16, 16, 16],
                 kernel_sizes=[7, 7, 7, 7]):
        super(CBAMMixVisionTransformer, self).__init__()

        self.backbone = MixVisionTransformer(**backbone_cfg)

        self.num_stages = backbone_cfg.get('num_stages', 4)
        embed_dims = backbone_cfg.get('embed_dims', 32)
        num_heads = backbone_cfg.get('num_heads', [1, 2, 5, 8])

        self.out_channels = []
        for i in range(self.num_stages):
            self.out_channels.append(embed_dims * num_heads[i])

        self.cbam_blocks = nn.ModuleList()
        for i in range(self.num_stages):
            cbam_block = CBAMBlock(
                in_channels=self.out_channels[i],
                reduction_ratio=reduction_ratios[i] if i < len(reduction_ratios) else 16,
                kernel_size=kernel_sizes[i] if i < len(kernel_sizes) else 7
            )
            self.cbam_blocks.append(cbam_block)

    def forward(self, x):
        """Forward pass with CBAM attention."""
        outs = self.backbone(x)

        enhanced_outs = []
        for i, out in enumerate(outs):
            enhanced_out = self.cbam_blocks[i](out)
            enhanced_outs.append(enhanced_out)

        return enhanced_outs

    def init_weights(self):
        """Initialize weights."""
        self.backbone.init_weights()

