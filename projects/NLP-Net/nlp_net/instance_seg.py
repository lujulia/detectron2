# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry


__all__ = ["INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch"]

INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = ""

@INS_EMBED_BRANCHES_REGISTRY.register()
def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)



class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output
    
def init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class MAD(nn.Module):
    def __init__(self, c1=16, c2=32, classes=19):
        super(MAD, self).__init__()
        self.c1, self.c2 = c1, c2
        self.LMFFNet_Block_2 = nn.Sequential()

        self.mid_layer_1x1 = Conv(128 + 3, c1, 1, 1, padding=0, bn_acti=False)

        self.deep_layer_1x1 = Conv(256 + 3, c2, 1, 1, padding=0, bn_acti=False)

        self.DwConv1 = Conv(self.c1 + self.c2, self.c1 + self.c2, (3, 3), 1, padding=(1, 1),
                            groups=self.c1 + self.c2, bn_acti=True)

        self.PwConv1 = Conv(self.c1 + self.c2, classes, 1, 1, padding=0, bn_acti=False)

        self.DwConv2 = Conv(256 + 3, 256 + 3, (3, 3), 1, padding=(1, 1), groups=256 + 3, bn_acti=True)
        self.PwConv2 = Conv(256 + 3, classes, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2 = x

        x2_size = x2.size()[2:]

        x1_ = self.mid_layer_1x1(x1)
        x2_ = self.deep_layer_1x1(x2)

        x2_ = F.interpolate(x2_, [x2_size[0] * 2, x2_size[1] * 2], mode='bilinear', align_corners=False)

        x1_x2_cat = torch.cat([x1_, x2_], 1)
        x1_x2_cat = self.DwConv1(x1_x2_cat)
        x1_x2_cat = self.PwConv1(x1_x2_cat)
        x1_x2_cat_att = torch.sigmoid(x1_x2_cat)

        o = self.DwConv2(x2)
        o = self.PwConv2(o)
        o = F.interpolate(o, [x2_size[0] * 2, x2_size[1] * 2], mode='bilinear', align_corners=False)

        o = o * x1_x2_cat_att

        o = F.interpolate(o, [x2_size[0] * 8, x2_size[1] * 8], mode='bilinear', align_corners=False)

        return o

@INS_EMBED_BRANCHES_REGISTRY.register()
class PanopticLMFFNetInsEmbedHead(nn.Module):
    """
    A instance embedding head described in paper:`LMFFNet`.
    """
    #@configurable
    def __init__(
        self,
        cfg,
        input_shape: Dict[str, ShapeSpec],
        *,
        center_loss_weight: float = 200.0,
        offset_loss_weight: float = 0.01,
        **kwargs,
    ):
        super().__init__()

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight

        self.center_MAD = MAD(classes=1)
        self.offset_MAD = MAD(classes=2)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, list):
            for feature in module:
                init_weight(feature, nn.init.kaiming_normal_, nn.BatchNorm2d, 1e-3, 0.1, mode='fan_in')

        else:
            init_weight(module, nn.init.kaiming_normal_, nn.BatchNorm2d, 1e-3, 0.1, mode='fan_in')             
    
    def forward( 
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        #x = features[self.in_features[0]]
        #x = self.aspp(x)
        #x = self.predictor(x)
        features = features['FFM-B1'],features['FFM-B2']
        center, offset = self.layers(features)
        if self.training:
            return (
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            return center, offset, {}, {}
        
    def layers(self, features):
        #y = super().layers(features)
        # center
        center = self.center_MAD(features)
        # offset
        offset = self.offset_MAD(features)
        return center, offset

    def center_losses(self, predictions, targets, weights):
        #predictions = F.interpolate(
        #    predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        #)
        loss = self.center_loss(predictions, targets)
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses
    
    def offset_losses(self, predictions, targets, weights):
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses
