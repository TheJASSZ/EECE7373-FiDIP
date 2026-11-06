# ABLATION (a), value of the synthetic data: ViTPose-L 256x192 trained on the 200 REAL images only.
# Identical to syrip_vitpose-l_udp-256x192.py except the train set: real200 resampled x10, so an
# epoch still has 2000 samples. Iterations, schedule, Val400 selection cadence and augmentation
# are therefore unchanged; only the 1000 synthetic images are removed.
_base_ = ['./syrip_vitpose-l_udp-256x192.py']

train_dataloader = dict(
    dataset=dict(
        datasets=[
            dict(type='CocoDataset', data_root='/scratch/purohitham.v/fidip/data/',
                 data_mode='topdown',
                 metainfo=dict(from_file='/scratch/purohitham.v/fidip/mmpose/configs/_base_/datasets/coco.py'),
                 ann_file='/scratch/purohitham.v/fidip/mmpose_data/splits/train_real200.json',
                 data_prefix=dict(img='syrip/images/train_infant/'), pipeline=[]),
        ],
        sample_ratio_factor=[10.0]))
