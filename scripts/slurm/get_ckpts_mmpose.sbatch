#!/bin/bash
# Download the COCO-pretrained MMPose checkpoints used by the IMPROVE stream into $R/models/mmpose
# and record their SHA256. URLs and published COCO val2017 AP (person detector boxes, AP_H 56) are
# copied verbatim from the mmpose v1.3.2 model-zoo metafiles:
#   configs/body_2d_keypoint/topdown_heatmap/coco/{vitpose_coco,hrnet_dark_coco,hrnet_udp_coco}.yml
#SBATCH --job-name=get_ckpts_mmpose
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
D=/scratch/purohitham.v/fidip/models/mmpose
mkdir -p "$D"
cd "$D"
B=https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco
# name | url | published COCO val AP | config in mmpose v1.3.2
cat > ckpt_sources.tsv <<EOF
name	url	coco_val_AP	mmpose_config
td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth	$B/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth	0.772	configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py
td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth	$B/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth	0.773	configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py
td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth	$B/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth	0.782	configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py
td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth	$B/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth	0.788	configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py
EOF
tail -n +2 ckpt_sources.tsv | while IFS=$'\t' read -r name url ap cfg; do
  if [ ! -s "$name" ]; then
    echo "downloading $name"
    curl -fL --retry 5 --retry-delay 10 -o "$name.part" "$url" && mv "$name.part" "$name"
  fi
done
sha256sum *.pth | tee SHA256SUMS
ls -la
echo "done $(date -Is)"
