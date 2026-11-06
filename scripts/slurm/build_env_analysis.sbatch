#!/bin/bash
# Build the small CPU-only analysis env at $R/envs/analysis (eval/*, figures, tables).
# Same pins as the fidip env where they overlap (numpy 1.26.4, pycocotools 2.0.8), so metrics
# are computed with identical library versions. Idempotent.
#SBATCH --job-name=build_env_analysis
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
R=/scratch/purohitham.v/fidip
ENV=$R/envs/analysis
echo "host=$(hostname) start=$(date -Is) job=${SLURM_JOB_ID:-none}"
module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=$R/envs/.conda_pkgs PIP_CACHE_DIR=$R/envs/.pip_cache
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" --override-channels -c conda-forge python=3.12 pip
fi
conda activate "$ENV"
python -m pip install --no-input numpy==1.26.4 pycocotools==2.0.8 matplotlib==3.9.2 pandas==2.2.3 \
    scipy==1.14.1 pytest==8.3.3 pillow==10.4.0 opencv-python-headless==4.10.0.84
python -m pip freeze > $R/envs/analysis-pip-freeze.txt
python - <<'PY'
import numpy, pycocotools, matplotlib, pandas, scipy, PIL, cv2
from importlib.metadata import version
print("numpy", numpy.__version__, "pycocotools", version("pycocotools"), "matplotlib", matplotlib.__version__,
      "pandas", pandas.__version__, "scipy", scipy.__version__, "PIL", PIL.__version__, "cv2", cv2.__version__)
PY
echo "end=$(date -Is)"
