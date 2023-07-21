# --------------------------------------------------------
# WDA Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from functools import partial
from timm.models import vit_deit_small_patch16_224

from .WDA_transformer import WDATransformer
from .wdassl import WDASSL

vit_models = dict(
    deit_small=vit_deit_small_patch16_224,
)


def build_model(config):
    model_type = config.MODEL.TYPE
    encoder_type = config.MODEL.WDASSL.ENCODER

    if encoder_type == 'WDA':
        enc = partial(
            WDATransformer,
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.WDA.PATCH_SIZE,
            in_chans=config.MODEL.WDA.IN_CHANS,
            embed_dim=config.MODEL.WDA.EMBED_DIM,
            depths=config.MODEL.WDA.DEPTHS,
            num_heads=config.MODEL.WDA.NUM_HEADS,
            window_size=config.MODEL.WDA.WINDOW_SIZE,
            mlp_ratio=config.MODEL.WDA.MLP_RATIO,
            qkv_bias=config.MODEL.WDA.QKV_BIAS,
            qk_scale=config.MODEL.WDA.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.WDA.APE,
            patch_norm=config.MODEL.WDA.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_before_mlp=config.MODEL.WDA.NORM_BEFORE_MLP,
        )
    elif encoder_type.startswith('vit') or encoder_type.startswith('deit'):
        enc = vit_models[encoder_type]
    else:
        raise NotImplementedError(f'--> Unknown encoder_type: {encoder_type}')

    if model_type == 'WDASSL':
        encoder = enc(
            num_classes=0,
            drop_path_rate=config.MODEL.WDASSL.ONLINE_DROP_PATH_RATE,
        )
        encoder_k = enc(
            num_classes=0,
            drop_path_rate=config.MODEL.WDASSL.TARGET_DROP_PATH_RATE,
        )
        model = WDASSL(
            cfg=config,
            encoder=encoder,
            encoder_k=encoder_k,
            contrast_momentum=config.MODEL.WDASSL.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.WDASSL.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.WDASSL.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.WDASSL.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.WDASSL.PRED_NUM_LAYERS,
        )
    elif model_type == 'linear':
        model = enc(
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f'--> Unknown model_type: {model_type}')

    return model
