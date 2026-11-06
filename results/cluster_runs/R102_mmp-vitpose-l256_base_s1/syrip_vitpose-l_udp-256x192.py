ann_real200 = '/scratch/purohitham.v/fidip/mmpose_data/splits/train_real200.json'
ann_syn1000 = '/scratch/purohitham.v/fidip/mmpose_data/splits/train_syn1000.json'
ann_test100 = 'splits/test100.json'
ann_val400 = 'splits/val400.json'
backend_args = dict(backend='local')
coco_metainfo = dict(
    from_file=
    '/scratch/purohitham.v/fidip/mmpose/configs/_base_/datasets/coco.py')
codec = dict(
    heatmap_size=(
        48,
        64,
    ),
    input_size=(
        192,
        256,
    ),
    sigma=2,
    type='UDPHeatmap')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper',
    ])
data_mode = 'topdown'
data_root = '/scratch/purohitham.v/fidip/data/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
derived_root = '/scratch/purohitham.v/fidip/mmpose_data/'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_test100 = 'syrip/images/test100/'
img_train = 'syrip/images/train_infant/'
img_val400 = 'syrip/images/val400/'
launcher = 'none'
load_from = '/scratch/purohitham.v/fidip/models/mmpose/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth'
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        arch='large',
        drop_path_rate=0.5,
        img_size=(
            256,
            192,
        ),
        out_type='featmap',
        patch_cfg=dict(padding=2),
        patch_size=16,
        qkv_bias=True,
        type='mmpretrain.VisionTransformer',
        with_cls_token=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            heatmap_size=(
                48,
                64,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='UDPHeatmap'),
        deconv_kernel_sizes=(
            4,
            4,
        ),
        deconv_out_channels=(
            256,
            256,
        ),
        in_channels=1024,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        out_channels=17,
        type='HeatmapHead'),
    test_cfg=dict(flip_mode='heatmap', flip_test=True, shift_heatmap=False),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    constructor='LayerDecayOptimWrapperConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.1),
    paramwise_cfg=dict(
        custom_keys=dict(
            bias=dict(decay_multi=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0)),
        layer_decay_rate=0.8,
        num_layers=24))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[
            24,
            29,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=1)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='splits/test100.json',
        data_mode='topdown',
        data_prefix=dict(img='syrip/images/test100/'),
        data_root='/scratch/purohitham.v/fidip/data/',
        metainfo=dict(
            from_file=
            '/scratch/purohitham.v/fidip/mmpose/configs/_base_/datasets/coco.py'
        ),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/scratch/purohitham.v/fidip/data/splits/test100.json',
    type='CocoMetric')
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        datasets=[
            dict(
                ann_file=
                '/scratch/purohitham.v/fidip/mmpose_data/splits/train_real200.json',
                data_mode='topdown',
                data_prefix=dict(img='syrip/images/train_infant/'),
                data_root='/scratch/purohitham.v/fidip/data/',
                metainfo=dict(
                    from_file=
                    '/scratch/purohitham.v/fidip/mmpose/configs/_base_/datasets/coco.py'
                ),
                pipeline=[],
                type='CocoDataset'),
            dict(
                ann_file=
                '/scratch/purohitham.v/fidip/mmpose_data/splits/train_syn1000.json',
                data_mode='topdown',
                data_prefix=dict(img='syrip/images/train_infant/'),
                data_root='/scratch/purohitham.v/fidip/data/',
                metainfo=dict(
                    from_file=
                    '/scratch/purohitham.v/fidip/mmpose/configs/_base_/datasets/coco.py'
                ),
                pipeline=[],
                type='CocoDataset'),
        ],
        metainfo=dict(
            from_file=
            '/scratch/purohitham.v/fidip/mmpose/configs/_base_/datasets/coco.py'
        ),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(
                encoder=dict(
                    heatmap_size=(
                        48,
                        64,
                    ),
                    input_size=(
                        192,
                        256,
                    ),
                    sigma=2,
                    type='UDPHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        sample_ratio_factor=[
            5.0,
            1.0,
        ],
        type='CombinedDataset'),
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine', use_udp=True),
    dict(
        encoder=dict(
            heatmap_size=(
                48,
                64,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='UDPHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='splits/val400.json',
        data_mode='topdown',
        data_prefix=dict(img='syrip/images/val400/'),
        data_root='/scratch/purohitham.v/fidip/data/',
        metainfo=dict(
            from_file=
            '/scratch/purohitham.v/fidip/mmpose/configs/_base_/datasets/coco.py'
        ),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/scratch/purohitham.v/fidip/data/splits/val400.json',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine', use_udp=True),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/scratch/purohitham.v/fidip/runs/R102_mmp-vitpose-l256_base_s1'
