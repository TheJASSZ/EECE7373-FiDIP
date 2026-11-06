# STRETCH: ViTPose-H 256x192 + FiDIP-style domain adaptation (real vs synthetic), via a gradient-
# reversal domain classifier on the backbone's last feature map (mmpose_exps/syrip_da).
# Identical to syrip_vitpose-h_udp-256x192.py (data mix, augmentation, schedule, LR, batch,
# Val400 selection, test-time behaviour) except:
#   * model type DomainAdaptiveTopdown: loss = pose MSE + 0.1 * BCE(D(GRL(f)), is_synthetic),
#     lambda ramped 0 -> 1 with the DANN schedule over the first 630 iterations (10 epochs);
#   * train sub-datasets are SyRIPCocoDataset (carries images[].is_synthetic) and PackPoseInputs
#     forwards 'is_synthetic' to the model. The classifier is unused at test time.
_base_ = ['./syrip_vitpose-h_udp-256x192.py']

custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper', 'syrip_da'],
    allow_failed_imports=False)

model = dict(
    type='DomainAdaptiveTopdown',
    da_cfg=dict(in_channels=1280, hidden=256, loss_weight=0.1, grl_lambda=1.0,
                lambda_warmup_iters=630))

codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)
meta_keys = ('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 'ori_shape',
             'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
             'flip_direction', 'flip_indices', 'raw_ann_info', 'dataset_name', 'is_synthetic')
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs', meta_keys=meta_keys)
]
coco_meta_da = dict(from_file='/scratch/purohitham.v/fidip/mmpose/configs/_base_/datasets/coco.py')
train_dataloader = dict(
    dataset=dict(
        datasets=[
            dict(type='SyRIPCocoDataset', data_root='/scratch/purohitham.v/fidip/data/',
                 data_mode='topdown', metainfo=coco_meta_da,
                 ann_file='/scratch/purohitham.v/fidip/mmpose_data/splits/train_real200.json',
                 data_prefix=dict(img='syrip/images/train_infant/'), pipeline=[]),
            dict(type='SyRIPCocoDataset', data_root='/scratch/purohitham.v/fidip/data/',
                 data_mode='topdown', metainfo=coco_meta_da,
                 ann_file='/scratch/purohitham.v/fidip/mmpose_data/splits/train_syn1000.json',
                 data_prefix=dict(img='syrip/images/train_infant/'), pipeline=[]),
        ],
        pipeline=train_pipeline))
