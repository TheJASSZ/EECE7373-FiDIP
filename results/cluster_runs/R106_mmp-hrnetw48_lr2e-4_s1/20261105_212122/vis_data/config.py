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
        72,
        96,
    ),
    input_size=(
        288,
        384,
    ),
    sigma=3,
    type='MSRAHeatmap',
    unbiased=True)
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
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
load_from = '/scratch/purohitham.v/fidip/models/mmpose/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth'
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        extra=dict(
            stage1=dict(
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_branches=1,
                num_channels=(64, ),
                num_modules=1),
            stage2=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_branches=2,
                num_channels=(
                    48,
                    96,
                ),
                num_modules=1),
            stage3=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_branches=3,
                num_channels=(
                    48,
                    96,
                    192,
                ),
                num_modules=4),
            stage4=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_branches=4,
                num_channels=(
                    48,
                    96,
                    192,
                    384,
                ),
                num_modules=3)),
        in_channels=3,
        type='HRNet'),
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
                72,
                96,
            ),
            input_size=(
                288,
                384,
            ),
            sigma=3,
            type='MSRAHeatmap',
            unbiased=True),
        deconv_out_channels=None,
        in_channels=48,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        out_channels=17,
        type='HeatmapHead'),
    test_cfg=dict(flip_mode='heatmap', flip_test=True, shift_heatmap=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(optimizer=dict(lr=0.0002, type='Adam'))
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
                288,
                384,
            ), type='TopdownAffine'),
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
                288,
                384,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    heatmap_size=(
                        72,
                        96,
                    ),
                    input_size=(
                        288,
                        384,
                    ),
                    sigma=3,
                    type='MSRAHeatmap',
                    unbiased=True),
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
        288,
        384,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            heatmap_size=(
                72,
                96,
            ),
            input_size=(
                288,
                384,
            ),
            sigma=3,
            type='MSRAHeatmap',
            unbiased=True),
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
                288,
                384,
            ), type='TopdownAffine'),
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
        288,
        384,
    ), type='TopdownAffine'),
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
work_dir = '/scratch/purohitham.v/fidip/runs/R106_mmp-hrnetw48_lr2e-4_s1'
