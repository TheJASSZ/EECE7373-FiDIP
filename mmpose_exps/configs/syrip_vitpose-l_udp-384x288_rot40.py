# ABLATION (b), rotation range: ViTPose-L 384x288 with RandomBBoxTransform(rotate_factor=40).
# Base recipe (syrip_vitpose-l_udp-384x288.py) uses the mmpose default rotate_factor=80: angle =
# truncnorm(-1, 1) * factor, applied with p=0.6. Everything else is unchanged.
_base_ = ['./syrip_vitpose-l_udp-384x288.py']

codec = dict(
    type='UDPHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', rotate_factor=40.0),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
