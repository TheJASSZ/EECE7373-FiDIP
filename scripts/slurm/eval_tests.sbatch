#!/bin/bash
# Run the evaluator unit tests with the cluster library versions (numpy 1.26.4, pycocotools 2.0.8).
#SBATCH --job-name=eval_tests
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
R=/scratch/purohitham.v/fidip
module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV:-$R/envs/analysis}"
cd $R/repo
echo "host=$(hostname) env=$CONDA_PREFIX git_tree=$(cat $R/repo/.git_tree 2>/dev/null || echo n/a)"
python -c "import numpy, pycocotools; from importlib.metadata import version; print('numpy', numpy.__version__, 'pycocotools', version('pycocotools'))"
python -m pytest eval/tests -q -p no:cacheprovider
