# TREATMENT: ViTPose-H (classic decoder head) 256x192, COCO-pretrained, fine-tuned on SyRIP.
# Inherits syrip_vitpose-l_udp-256x192.py and applies exactly the L -> H differences of the
# official mmpose v1.3.2 configs (diff of td-hm_ViTPose-{large,huge}_8xb64-210e_coco-256x192.py):
#   arch large -> huge, drop_path 0.5 -> 0.55, layer decay 0.8 -> 0.85 over 24 -> 32 blocks,
#   head in_channels 1024 -> 1280. Same nominal LR (1e-4 @ batch 64) and schedule.
_base_ = ['./syrip_vitpose-l_udp-256x192.py']

load_from = ('/scratch/purohitham.v/fidip/models/mmpose/'
             'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth')

optim_wrapper = dict(paramwise_cfg=dict(num_layers=32, layer_decay_rate=0.85))

model = dict(
    backbone=dict(arch='huge', drop_path_rate=0.55),
    head=dict(in_channels=1280))
