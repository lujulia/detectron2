# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN
#from detectron2.projects.deeplab import add_deeplab_config


def add_panoptic_lmffnet_config(cfg):
    """
    Add config for Panoptic-LMFFNet.
    """
    # Reuse DeepLab config.
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Used for `poly` learning rate schedule.
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    # Loss type, choose from `cross_entropy`, `hard_pixel_mining`.
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "cross_entropy"
    # DeepLab settings
    #cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]
    #cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    #cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    #cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    #cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    #cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV = False
    # Backbone new configs
    #cfg.MODEL.RESNETS.RES4_DILATION = 1
    #cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 2, 4]
    # ResNet stem type from: `basic`, `deeplab`
    #cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"
    # Target generation parameters.
    cfg.INPUT.GAUSSIAN_SIGMA = 10
    cfg.INPUT.IGNORE_STUFF_IN_OFFSET = True
    cfg.INPUT.SMALL_INSTANCE_AREA = 4096
    cfg.INPUT.SMALL_INSTANCE_WEIGHT = 3
    cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC = False
    # Optimizer type.
    cfg.SOLVER.OPTIMIZER = "SGD"
    # Panoptic-DeepLab semantic segmentation head.
    # We add an extra convolution before predictor.
    cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K = 0.2
    # Panoptic-DeepLab instance segmentation head.
    cfg.MODEL.INS_EMBED_HEAD = CN()
    cfg.MODEL.INS_EMBED_HEAD.NAME = "PanopticLMFFNetInsEmbedHead"
    cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES = ["FFM-B1", "FFM-B2"]
    #cfg.MODEL.INS_EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
    #cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
    #cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS = 256
    #cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS = [6, 12, 18]
    #cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT = 0.1
    # We add an extra convolution before predictor.
    #cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS = 32
    #cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM = 128
    #cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE = 4
    #cfg.MODEL.INS_EMBED_HEAD.NORM = "SyncBN"
    cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT = 200.0
    cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT = 0.01
    # Panoptic-DeepLab post-processing setting.
    cfg.MODEL.PANOPTIC_LMFFNET = CN()
    # Stuff area limit, ignore stuff region below this number.
    cfg.MODEL.PANOPTIC_LMFFNET.STUFF_AREA = 2048
    cfg.MODEL.PANOPTIC_LMFFNET.CENTER_THRESHOLD = 0.1
    cfg.MODEL.PANOPTIC_LMFFNET.NMS_KERNEL = 7
    cfg.MODEL.PANOPTIC_LMFFNET.TOP_K_INSTANCE = 200
    # If set to False, Panoptic-DeepLab will not evaluate instance segmentation.
    cfg.MODEL.PANOPTIC_LMFFNET.PREDICT_INSTANCES = True
    #cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV = False
    # This is the padding parameter for images with various sizes. ASPP layers
    # requires input images to be divisible by the average pooling size and we
    # can use `MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY` to pad all images to
    # a fixed resolution (e.g. 640x640 for COCO) to avoid having a image size
    # that is not divisible by ASPP average pooling size.
    #cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY = -1
    # Only evaluates network speed (ignores post-processing).
    cfg.MODEL.PANOPTIC_LMFFNET.BENCHMARK_NETWORK_SPEED = False
    cfg.DATALOADER.NUM_GPUS = 2
