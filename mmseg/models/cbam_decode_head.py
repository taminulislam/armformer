import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.ham_head import LightHamHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from .cbam_attention import CBAMBlock


@MODELS.register_module()
class CBAMLightHamHead(LightHamHead):
    """LightHamHead with CBAM attention integration.

    This decode head extends the standard LightHamHead by adding
    CBAM attention modules at strategic points to enhance feature
    representation for better segmentation performance.

    Args:
        cbam_positions (list): Positions to insert CBAM blocks.
            Options: ['pre_ham', 'post_ham', 'both']. Default: ['pre_ham'].
        cbam_cfg (dict): Configuration for CBAM blocks.
        ham_channels (int): input channels for Hamburger. Default: 512.
        ham_kwargs (dict): kwargs for Ham. Default: dict().
        **kwargs: Other arguments for LightHamHead.
    """

    def __init__(self,
                 cbam_positions=['pre_ham'],
                 cbam_cfg=None,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 **kwargs):
        super().__init__(ham_channels=ham_channels, ham_kwargs=ham_kwargs, **kwargs)

        self.cbam_positions = cbam_positions
        self.cbam_cfg = cbam_cfg or {}

        self.pre_ham_cbam = None
        self.post_ham_cbam = None

        if 'pre_ham' in cbam_positions or 'both' in cbam_positions:
            self.pre_ham_cbam = CBAMBlock(
                in_channels=self.ham_channels,
                **self.cbam_cfg
            )

        if 'post_ham' in cbam_positions or 'both' in cbam_positions:
            self.post_ham_cbam = CBAMBlock(
                in_channels=self.ham_channels,
                **self.cbam_cfg
            )

    def forward(self, inputs):
        """Forward function with CBAM attention."""
        inputs = self._transform_inputs(inputs)

        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        if self.pre_ham_cbam is not None:
            x = self.pre_ham_cbam(x)

        x = self.hamburger(x)

        if self.post_ham_cbam is not None:
            x = self.post_ham_cbam(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output

