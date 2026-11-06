# TREATMENT: ViTPose-L at 384x288 (same input size as the HRNet-W48 control and FiDIP).
# Inherits syrip_vitpose-l_udp-256x192.py; only the resolution changes:
#   * backbone img_size (H, W) = (384, 288): 24x18 = 432 patch tokens (16x12 = 192 at 256x192)
#   * UDP codec at input 288x384 / heatmap 72x96. sigma 3 keeps the Gaussian the same size
#     relative to the person as sigma 2 at 256x192; it is also mmpose's choice for its
#     384x288 UDP configs (td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py).
#   * load_from = the 256x192 COCO checkpoint with pos_embed bicubically resized 16x12 -> 24x18
#     by mmpose_exps/tools/resize_vit_pos_embed.py. mmpretrain 1.2.0's own on-load resize
#     (VisionTransformer._prepare_pos_embed) assumes a square grid (int(sqrt(192)) = 13) and
#     fails for 16x12, so the resize is done offline and the derived file's SHA256 is recorded.
_base_ = ['./syrip_vitpose-l_udp-256x192.py']

load_from = ('/scratch/purohitham.v/fidip/models/mmpose/derived/'
             'td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314_posembed24x18.pth')

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
