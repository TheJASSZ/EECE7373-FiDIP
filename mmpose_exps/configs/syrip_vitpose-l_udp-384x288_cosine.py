# RECIPE candidate (selected on Val400): cosine LR decay instead of the step schedule.
# Same 1-epoch linear warm-up, then cosine from the base LR to 1% of it over epochs 1..30.
_base_ = ['./syrip_vitpose-l_udp-384x288.py']

param_scheduler = [
    dict(type='LinearLR', begin=0, end=1, start_factor=0.001, by_epoch=True,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', begin=1, end=30, T_max=29, eta_min_ratio=0.01,
         by_epoch=True, convert_to_iter_based=True),
]
