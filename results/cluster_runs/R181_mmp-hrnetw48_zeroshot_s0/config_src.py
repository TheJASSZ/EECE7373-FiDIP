# CONTROL: HRNet-W48 + DARK (DarkPose) 384x288, COCO-pretrained, fine-tuned on SyRIP.
# Same backbone family, input size and decoding as FiDIP's best published row
# (DarkPose+FiDIP HRNet-W48 384x288: Test100 AP 93.6).
#
# Model / codec / test-time settings are copied from mmpose v1.3.2
#   configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py
# Fine-tuning changes (shared with every backbone, see _base_/syrip_base.py):
#   * load_from = the COCO checkpoint of that config (backbone init_cfg dropped; nothing downloaded)
#   * SyRIP data (1:1 real:syn, 2000 samples/epoch), Val400 selection, 30-epoch step schedule
#   * LR set explicitly (no auto_scale_lr). Official: Adam 5e-4 @ total batch 256 (8x32).
#     Screening LR = 1e-4 @ batch 32, the same nominal LR as the ViTPose configs and FiDIP's
#     fine-tuning LR (Adam 1e-4). Swept on Val400 in variant configs.
_base_ = ['./_base_/syrip_base.py']

load_from = ('/scratch/purohitham.v/fidip/models/mmpose/'
             'td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth')

optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4))

# codec: DARK = unbiased MSRA heatmap encoding + DARK (Taylor) decoding with blur kernel 11
codec = dict(
    type='MSRAHeatmap',
    input_size=(288, 384),
    heatmap_size=(72, 96),
    sigma=3,
    unbiased=True)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK',
                        num_blocks=(4, ), num_channels=(64, )),
            stage2=dict(num_modules=1, num_branches=2, block='BASIC',
                        num_blocks=(4, 4), num_channels=(48, 96)),
            stage3=dict(num_modules=4, num_branches=3, block='BASIC',
                        num_blocks=(4, 4, 4), num_channels=(48, 96, 192)),
            stage4=dict(num_modules=3, num_branches=4, block='BASIC',
                        num_blocks=(4, 4, 4, 4),
                        num_channels=(48, 96, 192, 384))),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))

# official COCO augmentation (identical for every backbone in this study)
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),   # shift 0.16 p.3, scale [0.5,1.5], rot truncnorm*80 p.6
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(batch_size=32, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))
