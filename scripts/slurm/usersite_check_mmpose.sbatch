#!/bin/bash
# IMPROVE stream: check whether ~/.local user site-packages shadow the mmpose env's packages
# (compare module versions/paths with and without PYTHONNOUSERSITE), then pin
# PYTHONNOUSERSITE=1 into the env's activation vars (conda env config vars). CPU only.
#SBATCH --job-name=im_usersite_check
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -uo pipefail
R=/scratch/purohitham.v/fidip
module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
echo "user site: $(python -c 'import site; print(site.getusersitepackages(), site.ENABLE_USER_SITE)')"
ls -la ~/.local/lib/ 2>&1
ls ~/.local/lib/python3.10/site-packages 2>&1 | head -60
for flag in "" 1; do
  PYTHONNOUSERSITE=$flag python - <<'PY'
import importlib, os
mods = ['numpy', 'torch', 'torchvision', 'mmengine', 'mmcv', 'mmpose', 'mmpretrain', 'mmdet', 'cv2',
        'xtcocotools', 'pycocotools', 'scipy', 'matplotlib', 'json_tricks', 'munkres', 'chumpy', 'PIL',
        'yaml', 'rich', 'addict', 'termcolor', 'yapf', 'packaging', 'shapely', 'terminaltables', 'einops']
print('PYTHONNOUSERSITE =', repr(os.environ.get('PYTHONNOUSERSITE')))
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f'  {m:15s} {str(getattr(mod, "__version__", "?")):14s} {mod.__file__}')
    except Exception as e:
        print(f'  {m:15s} IMPORT FAILED {type(e).__name__}: {e}')
PY
done
conda env config vars set -p $R/envs/mmpose PYTHONNOUSERSITE=1
conda deactivate; conda activate $R/envs/mmpose
conda env config vars list -p $R/envs/mmpose
python -c "import os, site; print('after re-activate: PYTHONNOUSERSITE =', os.environ.get('PYTHONNOUSERSITE'), 'ENABLE_USER_SITE =', site.ENABLE_USER_SITE)"
