# Shared SyRIP base for the IMPROVE stream (MMPose 1.3.2, mmengine 0.10.x).
#
# * Runtime block = mmpose v1.3.2 configs/_base_/default_runtime.py, with the checkpoint hook set
#   to keep the best Val400 epoch (save_best='coco/AP') and logging every 10 iters.
# * Data: CocoDataset-compatible COCO-17 GT files built by the REPRO stream ($R/data/splits).
#     train = official train_infant (200 real + 1000 synthetic), mixed 1:1 like FiDIP's
#             BalancedBatchSampler (upstream lib/utils/sampler.py): real200 resampled x5 +
#             syn1000 x1 -> 2000 samples / epoch.
#     val   = Val400 (validate500 minus validate100; 400 real images). ALL selection happens here.
#     test  = Test100 (= validate100). Evaluated only for configs pre-registered in
#             results/PREREGISTRATION.md (mmpose_exps/tools/eval_ckpt.py enforces this).
# * Pipelines are model-specific: each model config sets `pipeline=` for train/val/test.
# * GT boxes at val/test (no bbox_file); flip test is set in each model's test_cfg.

default_scope = 'mmpose'

# ---------------- runtime ----------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,            # resumable "last" checkpoint every 5 epochs (runs are < 1 h)
        max_keep_ckpts=1,
        save_best='coco/AP',   # best Val400 AP epoch -> best_coco_AP_epoch_<N>.pth
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
    badcase=dict(
        type='BadCaseAnalysisHook',
        enable=False,
        out_dir='badcase',
        metric_type='loss',
        badcase_thr=5))
custom_hooks = [dict(type='SyncBuffersHook')]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False
backend_args = dict(backend='local')
randomness = dict(seed=1, deterministic=False)  # seed overridden per run (--cfg-options)

# ---------------- data ----------------
data_root = '/scratch/purohitham.v/fidip/data/'
coco_metainfo = dict(
    from_file='/scratch/purohitham.v/fidip/mmpose/configs/_base_/datasets/coco.py')
dataset_type = 'CocoDataset'
data_mode = 'topdown'
# image resolution rule of the REPRO split builder (scripts/make_splits.py):
#   <SYRIP>/images/<split>/<file_name>,  SYRIP = $R/data/syrip
img_train = 'syrip/images/train_infant/'
img_val400 = 'syrip/images/val400/'
img_test100 = 'syrip/images/test100/'
ann_real200 = 'improve_splits/train_real200.json'   # train_infant images with is_synthetic == 0
ann_syn1000 = 'improve_splits/train_syn1000.json'   # train_infant images with is_synthetic == 1
ann_val400 = 'splits/val400.json'
ann_test100 = 'splits/test100.json'

train_dataloader = dict(
    batch_size=32,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=coco_metainfo,
        datasets=[
            dict(type=dataset_type, data_root=data_root, data_mode=data_mode,
                 metainfo=coco_metainfo, ann_file=ann_real200,
                 data_prefix=dict(img=img_train), pipeline=[]),
            dict(type=dataset_type, data_root=data_root, data_mode=data_mode,
                 metainfo=coco_metainfo, ann_file=ann_syn1000,
                 data_prefix=dict(img=img_train), pipeline=[]),
        ],
        sample_ratio_factor=[5.0, 1.0],   # 1:1 real:synthetic, 2000 samples / epoch
        pipeline=[],                      # set by the model config
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=coco_metainfo,
        ann_file=ann_val400,
        data_prefix=dict(img=img_val400),
        test_mode=True,
        pipeline=[],                      # set by the model config
    ))
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=coco_metainfo,
        ann_file=ann_test100,
        data_prefix=dict(img=img_test100),
        test_mode=True,
        pipeline=[],                      # set by the model config
    ))

# CocoMetric = xtcocotools OKS AP with GT areas (use_area=True), score = bbox_score(=1 for GT
# boxes) x mean keypoint score over keypoints > 0.2, OKS-NMS 0.9 (one GT infant per image).
# outfile_prefix is set per evaluation by mmpose_exps/tools/eval_ckpt.py, which dumps
# preds_<split>.json in COCO results format for eval/eval_coco.py.
val_evaluator = dict(type='CocoMetric', ann_file=data_root + ann_val400)
test_evaluator = dict(type='CocoMetric', ann_file=data_root + ann_test100)

# ---------------- shared schedule shape ----------------
# Every backbone uses the same epoch budget and schedule shape: 1-epoch linear warm-up, then the
# official COCO step shape (x0.1 at 170/210 and 200/210 of the budget). Model configs restate
# these explicitly when the budget changes.
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = dict()
test_cfg = dict()
param_scheduler = [
    dict(type='LinearLR', begin=0, end=1, start_factor=0.001, by_epoch=True,
         convert_to_iter_based=True),
    dict(type='MultiStepLR', begin=0, end=30, milestones=[24, 29], gamma=0.1,
         by_epoch=True),
]
