# TREATMENT: ViTPose-L (classic decoder head) 256x192, COCO-pretrained, fine-tuned on SyRIP.
#
# Model / codec / optimizer family copied from mmpose v1.3.2
#   configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py
# Fine-tuning changes (shared with every backbone, see _base_/syrip_base.py):
#   * load_from = the COCO checkpoint of that config (MAE init_cfg dropped; nothing downloaded)
#   * SyRIP data (1:1 real:syn, 2000 samples/epoch), Val400 selection, 30-epoch step schedule
#   * LR set explicitly (no auto_scale_lr). Official: AdamW 5e-4 @ total batch 512 (8x64),
#     wd 0.1, layer-wise LR decay 0.8 over 24 blocks, grad-clip 1.0, drop-path 0.5.
#     Screening LR = 1e-4 @ batch 64 (same nominal LR as the HRNet control); swept on Val400.
#   * Layer decay (not freezing) keeps the early blocks close to the COCO weights:
#     lr(block i) = lr * 0.8^(24-i); patch/pos embed get 0.8^25; head + last norm get lr.
_base_ = ['./_base_/syrip_base.py']

load_from = ('/scratch/purohitham.v/fidip/models/mmpose/'
             'td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth')

custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.8,
        # kept verbatim from the official config; LayerDecayOptimWrapperConstructor ignores
        # custom_keys and itself sets wd=0 for 1-D params, biases and pos_embed.
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='large',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.5,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=1024,
        out_channels=17,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False))

# official COCO augmentation (identical for every backbone in this study), UDP affine
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),   # shift 0.16 p.3, scale [0.5,1.5], rot truncnorm*80 p.6
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(batch_size=64, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))
