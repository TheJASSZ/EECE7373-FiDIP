#!/bin/bash
# Build the conda prefix env for the upstream FiDIP code: $R/envs/fidip
# Deviation from upstream fidip_env.yml (logged in notebook/repro.md): the yml's pip stage pins
# wheels that do not exist for cp312 (grpcio 1.48.2, mkl-fft/mkl-random/mkl-service), so we install
# the same python/torch/numpy/opencv/pycocotools versions with pip only (torch cu121 wheels).
# pyarrow is pinned to 17.0.0: streamlit's default pyarrow (26.x) requires NumPy >= 2, but upstream needs 1.26.4.
# Usage: sbatch scripts/slurm/build_env_fidip.sbatch
#SBATCH --job-name=build_env_fidip
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
R=/scratch/purohitham.v/fidip
ENV=$R/envs/fidip
REPO=$R/repo
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
export CONDA_PKGS_DIRS=$R/envs/.conda-pkgs PIP_CACHE_DIR=$R/envs/.pip-cache
# ~/.local/lib/python3.12 holds unrelated user packages (transformers, datasets, absl, ...): never let them
# satisfy or shadow this env's requirements. Also stored as a conda env var so `conda activate` sets it.
export PYTHONNOUSERSITE=1
module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -x $ENV/bin/python ]]; then
  conda create -y -p "$ENV" -c conda-forge --override-channels python=3.12.4 pip
fi
conda env config vars set -p "$ENV" PYTHONNOUSERSITE=1
conda activate "$ENV"
python -c "import site, sys; assert not site.ENABLE_USER_SITE, 'user site enabled'; print('user site disabled')"
python -m pip install --upgrade "pip==24.0"

cat > "$R/envs/fidip-constraints.txt" <<'EOF'
torch==2.3.1
torchvision==0.18.1
numpy==1.26.4
EOF
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -c "$R/envs/fidip-constraints.txt" \
  numpy==1.26.4 opencv-python-headless==4.10.0.84 pycocotools==2.0.8 json-tricks==3.17.3 yacs==0.1.8 \
  dsntnn==0.5.3 tensorboard==2.17.0 tensorboardX==2.6.2.2 cython==3.0.10 scipy==1.14.0 matplotlib==3.9.1 \
  pyyaml==6.0.1 easydict==1.13 pillow==10.3.0 gdown==5.2.0 streamlit==1.39.0 pyarrow==17.0.0

python - <<'EOF'
import torch, torchvision, numpy, cv2, pycocotools, json_tricks, yacs, dsntnn, scipy, matplotlib, gdown, streamlit
from torch.utils.tensorboard import SummaryWriter
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())
print('torchvision', torchvision.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)
print('arch list', torch.cuda.get_arch_list())
EOF

# exports go OUTSIDE the synced tree ($R/repo is overwritten by every push); copy them into the repo's envs/
python -m pip freeze --all > "$R/envs/fidip-pip-freeze.txt"
conda env export -p "$ENV" --no-builds | sed '/^prefix:/d' > "$R/envs/fidip.yml"
echo "network check (compute node):"; curl -sS -o /dev/null -w "drive.google.com %{http_code}\n" https://drive.google.com || true
echo "done at $(date)"
