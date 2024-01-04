###################################################################################################################
#  LMFFNet: A Well-Balanced Lightweight Network for Fast and Accurate Semantic Segmentation
#  Authors: M Shi, J Shen, Q Yi, J Weng, Z Huang, A Luo, Y Zhou
#  Published in£ºIEEE Transactions on Neural Networks and Learning Systems
#  Date: 2022/06/14
#
##################################################################################################################
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from .loss import L_color, L_spa, L_exp, L_TV

__all__ = ["Conv",
           "conv3x3_resume",
           "BNPReLU",
           "Init_Block",
           "SEM_B",
           "DownSamplingBlock",
           "InputInjection",
           "SENet_Block",
           "PMCA",
           "FFM_A",
           "FFM_B",
           "SEM_B_Block",
           "LMFFNet_backbone" 
           "build_lmffnet_backbone",
                  
]

def Split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


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

class conv3x3_resume(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.conv3x3 = Conv(nIn // 2, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.conv1x1_resume = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.conv3x3(input)
        output = self.conv1x1_resume(output)
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Init_Block(nn.Module):
    def __init__(self):
        super(Init_Block, self).__init__()
        number_f = 32
        self.conv = nn.Conv2d(6,number_f,3,2,1,bias=True) 
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

		#self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		#self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_o = x
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)

        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + r6*(torch.pow(x,2)-x)	
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x)
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        x_new = torch.cat([enhance_image,x_o],1)
        o = self.relu(self.conv(x_new))
        o = self.relu(self.e_conv2(o))
        o = self.relu(self.e_conv3(o))
        return o,enhance_image,r


class SEM_B(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv_left = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                               padding=(1, 1), groups=nIn // 4, bn_acti=True)

        self.dconv_right = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                                padding=(1 * d, 1 * d), groups=nIn // 4, dilation=(d, d), bn_acti=True)


        self.bn_relu_1 = BNPReLU(nIn)

        self.conv3x3_resume = conv3x3_resume(nIn , nIn , (dkSize, dkSize), 1,
                                padding=(1 , 1 ),  bn_acti=True)

    def forward(self, input):

        output = self.conv3x3(input)

        x1, x2 = Split(output)

        letf = self.dconv_left(x1)

        right = self.dconv_right(x2)

        output = torch.cat((letf, right), 1)
        output = self.conv3x3_resume(output)

        return self.bn_relu_1(output + input)


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class SENet_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SENet_Block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class PMCA(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(PMCA, self).__init__()

        self.partition_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv2x2 = Conv(ch_in, ch_in, 2, 1, padding=(0, 0), groups=ch_in, bn_acti=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.SE_Block = SENet_Block(ch_in=ch_in, reduction=reduction)

    def forward(self, x):
        o1 = self.partition_pool(x)

        o1 = self.conv2x2(o1)

        o2 = self.global_pool(x)

        o_sum = o1 + o2
        w = self.SE_Block(o_sum)
        o = w * x

        return o


class FFM_A(nn.Module):
    def __init__(self, ch_in):
        super(FFM_A, self).__init__()
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_in, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2 = x
        o = self.bn_prelu(torch.cat([x1, x2], 1))
        o = self.conv1x1(o)
        return o


class FFM_B(nn.Module):
    def __init__(self, ch_in, ch_pmca):
        super(FFM_B, self).__init__()
        self.PMCA = PMCA(ch_in=ch_pmca, reduction=8)
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_in, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2, x3 = x
        x2 = self.PMCA(x2)
        o = self.bn_prelu(torch.cat([x1, x2, x3], 1))
        o = self.conv1x1(o)
        return o

class SEM_B_Block1(nn.Module):
    def __init__(self, num_channels, num_block, dilation, flag):
        super(SEM_B_Block1, self).__init__()
        self.SEM_B1 = SEM_B(num_channels, d=dilation[0])
        self.SEM_B2 = SEM_B(num_channels, d=dilation[1])
        self.SEM_B3 = SEM_B(num_channels, d=dilation[2])
        
    def forward(self, x):
        x = self.SEM_B1(x)
        x = self.SEM_B2(x)
        o = self.SEM_B3(x)
        return o


class SEM_B_Block2(nn.Module):
    def __init__(self, num_channels, num_block, dilation, flag):
        super(SEM_B_Block2, self).__init__()
        self.SEM_B1 = SEM_B(num_channels, d=dilation[0])
        self.SEM_B2 = SEM_B(num_channels, d=dilation[1])
        self.SEM_B3 = SEM_B(num_channels, d=dilation[2])
        self.SEM_B4 = SEM_B(num_channels, d=dilation[3])
        self.SEM_B5 = SEM_B(num_channels, d=dilation[4])
        self.SEM_B6 = SEM_B(num_channels, d=dilation[5])
        self.SEM_B7 = SEM_B(num_channels, d=dilation[6])
        self.SEM_B8 = SEM_B(num_channels, d=dilation[7])
        

    def forward(self, x):
        x = self.SEM_B1(x)
        x = self.SEM_B2(x)
        x = self.SEM_B3(x)
        x = self.SEM_B4(x)
        x = self.SEM_B5(x)
        x = self.SEM_B6(x)
        x = self.SEM_B7(x)
        o = self.SEM_B8(x)
        return o


@BACKBONE_REGISTRY.register()
class LMFFNetBackbone(Backbone):
    def __init__(self, block_1=3, block_2=8):
        super().__init__()
        self.block_1 = block_1
        self.block_2 = block_2
        self.Init_Block = Init_Block().apply(weights_init)

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.FFM_A = FFM_A(32 + 3)

        self.downsample_1 = DownSamplingBlock(32 + 3, 64)

        self.SEM_B_Block1 = SEM_B_Block1(num_channels=64, num_block=self.block_1, dilation=[2, 2, 2], flag=1)

        self.FFM_B1 = FFM_B(ch_in=128 + 3, ch_pmca=64)

        self.downsample_2 = DownSamplingBlock(128 + 3, 128)

        self.SEM_B_Block2 = SEM_B_Block2(num_channels=128, num_block=self.block_2, dilation=[4, 4, 8, 8, 16, 16, 32, 32], flag=2)

        self.FFM_B2 = FFM_B(ch_in=256 + 3, ch_pmca=128)
	    
        self.L_color = L_color()
        self.L_spa = L_spa()
        self.L_exp = L_exp(16,0.6)
        self.L_TV = L_TV()

    def forward(self, input):
        # Init Block
        out_init_block, enhance_image,r = self.Init_Block(input)
        down_1 = self.down_1(enhance_image)
        input_ffm_a = out_init_block, down_1

        # FFM-A
        out_ffm_a = self.FFM_A(input_ffm_a)

        # SEM-B Block1
        out_downsample_1 = self.downsample_1(out_ffm_a)
        out_sem_block1 = self.SEM_B_Block1(out_downsample_1)

        # FFM-B1
        down_2 = self.down_2(enhance_image)
        input_sem1_pmca1 = out_sem_block1, out_downsample_1, down_2
        out_ffm_b1 = self.FFM_B1(input_sem1_pmca1)

        # SEM-B Block2
        out_downsample_2 = self.downsample_2(out_ffm_b1)
        out_se_block2 = self.SEM_B_Block2(out_downsample_2)

        # FFM-B2
        down_3 = self.down_3(enhance_image)
        input_sem2_pmca2 = out_se_block2, out_downsample_2, down_3
        out_ffm_b2 = self.FFM_B2(input_sem2_pmca2)

        # MAD
        #input_ffmb1_ffmb2 = out_ffm_b1, out_ffm_b2
        #out_mad = self.MAD(input_ffmb1_ffmb2)
        if self.training:
            return {"FFM-B1":out_ffm_b1,"FFM-B2":out_ffm_b2}, self.losses(input,enhance_image,r)#, weights)
        else:
            return {"FFM-B1":out_ffm_b1,"FFM-B2":out_ffm_b2}, {}
    
    def output_shape(self):
        return {"FFM-B1": ShapeSpec(channels=128+3, stride=1),"FFM-B2": ShapeSpec(channels=256+3, stride=1)}
    
    def losses(self, original, enhance, factor):#, weights=None):
        loss_tv = self.L_TV(factor)
        loss_spa = torch.mean(self.L_spa(enhance, original))
        loss_col = torch.mean(self.L_color(enhance))
        loss_exp = torch.mean(self.L_exp(enhance))
        losses = {"loss_ill_enh": 200*loss_tv+loss_spa+5*loss_col+10*loss_exp}
        return losses

@BACKBONE_REGISTRY.register()
def build_lmffnet_backbone(cfg, input_shape=None):
    # fmt: off
    #freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    #out_features        = cfg.MODEL.LMFFNET_BACKBONE.OUT_FEATURES
    #depth               = cfg.MODEL.RESNETS.DEPTH
    #num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    #width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    #bottleneck_channels = num_groups * width_per_group
    #in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    #out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    #stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    #res4_dilation       = cfg.MODEL.RESNETS.RES4_DILATION
    #res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    #deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    #deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    #deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    #res5_multi_grid     = cfg.MODEL.RESNETS.RES5_MULTI_GRID
    return LMFFNetBackbone(block_1=3, block_2=8)#.freeze(freeze_at)


