#!/bin/bash
# Build the MMPose 1.x prefix env at $R/envs/mmpose and clone the matching mmpose source tree.
# CPU-only job (no GPU needed to install wheels). Idempotent: re-running skips finished steps.
#SBATCH --job-name=build_env_mmpose
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out

set -euo pipefail
R=/scratch/purohitham.v/fidip
ENV=$R/envs/mmpose
MMPOSE_TAG=v1.3.2
echo "host=$(hostname) start=$(date -Is) job=${SLURM_JOB_ID:-none}"

module load miniconda3/25.9.1 cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
# keep caches on scratch, not in the 75 GB home
export CONDA_PKGS_DIRS=$R/envs/.conda_pkgs
export PIP_CACHE_DIR=$R/envs/.pip_cache
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

if [ ! -x "$ENV/bin/python" ]; then
  # conda-forge only: the 'defaults' channel needs an interactive ToS acceptance in miniconda 25.x
  conda create -y -p "$ENV" --override-channels -c conda-forge python=3.10 pip
fi
conda activate "$ENV"
python -V

PIP="python -m pip install --no-input"
$PIP --upgrade "pip<25" "setuptools<70" wheel
# 1) torch first, from the cu121 index (bundles CUDA 12.1 runtime; sm_50..sm_90 kernels)
$PIP torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
# 2) pin numpy<2 (torch 2.1 is built against numpy 1.x) and opencv<4.12 (4.12+ requires numpy>=2)
$PIP "numpy<2" "opencv-python<4.12"
# 3) OpenMMLab stack
$PIP -U openmim
$PIP "mmengine>=0.10.3,<0.11"
$PIP mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
# mmpretrain provides mmpretrain.VisionTransformer, which the mmpose ViTPose configs use as backbone
$PIP "mmpretrain==1.2.0"
# mmpose runtime deps (installed explicitly so chumpy cannot break the resolve)
$PIP json_tricks matplotlib munkres pillow scipy "xtcocotools>=1.12" pycocotools
$PIP --no-build-isolation chumpy || echo "WARN: chumpy failed to build (only needed for SMPL utilities; not used here)"
$PIP --no-deps mmpose==1.3.2
# mmdet is needed even with GT boxes: mmpose 1.3.2 imports it unconditionally in
# mmpose/models/heads/hybrid_heads/rtmo_head.py (`from mmdet.utils import ConfigType, reduce_mean`),
# so `import mmpose.models` fails without it. mmdet 3.3.0 accepts mmcv>=2.0.0rc4,<2.2.0.
$PIP mmdet==3.3.0
# re-assert pins in case anything above pulled newer versions
$PIP "numpy<2" "opencv-python<4.12"

python - <<'EOF'
import numpy, torch, torchvision, mmengine, mmcv, mmpose, mmpretrain, cv2, xtcocotools
print("numpy", numpy.__version__, "torch", torch.__version__, "cuda", torch.version.cuda,
      "tv", torchvision.__version__, "mmengine", mmengine.__version__, "mmcv", mmcv.__version__,
      "mmpose", mmpose.__version__, "mmpretrain", mmpretrain.__version__, "cv2", cv2.__version__)
print("arch list:", torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else "n/a")
from mmcv.ops import nms  # import check of compiled ops (CPU node: no GPU call here)
print("mmcv ops import OK")
EOF

# mmpose source tree (tools/train.py, tools/test.py, base configs, metafiles)
if [ ! -d "$R/mmpose/.git" ]; then
  git clone --depth 1 --branch "$MMPOSE_TAG" https://github.com/open-mmlab/mmpose.git "$R/mmpose"
fi
git -C "$R/mmpose" log -1 --format='mmpose source: %H %d %cd'

# export specs (copied into the repo later)
mkdir -p $R/envs/specs
conda env export -p "$ENV" > $R/envs/specs/mmpose.yml
python -m pip freeze > $R/envs/specs/mmpose-pip-freeze.txt
echo "done $(date -Is)"
