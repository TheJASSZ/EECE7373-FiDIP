# TREATMENT: ViTPose-H at 384x288 (same input size as the HRNet-W48 control and FiDIP).
# = syrip_vitpose-h_udp-256x192.py with the resolution changes of syrip_vitpose-l_udp-384x288.py
# (img_size (384, 288), UDP codec 288x384 / 72x96 sigma 3, pos_embed resized 16x12 -> 24x18
# offline by mmpose_exps/tools/resize_vit_pos_embed.py; see that config for the rationale).
_base_ = ['./syrip_vitpose-h_udp-256x192.py']

load_from = ('/scratch/purohitham.v/fidip/models/mmpose/derived/'
             'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314_posembed24x18.pth')

codec = dict(
    type='UDPHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)

model = dict(
    backbone=dict(img_size=(384, 288)),
    head=dict(decoder=codec))

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
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
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))
