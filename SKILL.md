---
name: explorer-cluster-fidip
description: Operate Northeastern's Explorer SLURM cluster for the EECE 7373 FiDIP / SyRIP infant-pose project -- passwordless SSH, environment activation, submit/monitor/cancel jobs, sync code and results, run conventions. Use whenever asked to run, check, or debug anything "on the cluster" for this project.
---

# FiDIP on Explorer — agent operating manual

## Connection facts
- Host alias **`explorer`** → `login.explorer.northeastern.edu`, user `purohitham.v`, key `~/.ssh/id_ed25519` (passwordless, verified 2026-11-05; login node seen: explorer-02).
- Always call `ssh -o BatchMode=yes -o ConnectTimeout=15 explorer '...'` — BatchMode never prompts, so a broken key fails fast instead of hanging the agent.
- Wrap slow login-node commands in `timeout N` (`du ~`, `module avail` stalled >60 s on 2026-11-05).
- Slurm account: default `cs5130.202630` (no `-A` needed). Data-transfer node: `xfer.discovery.neu.edu` (only needed for very large transfers).
- Local repo: `~/Documents/EECE7373-FiDIP` (Mac has ~7 GB free → never copy datasets/checkpoints to the Mac).

## Paths (`R=/scratch/purohitham.v/fidip`)
| What | Path |
|---|---|
| Code (rsync of local repo) | `$R/repo` |
| Raw downloads + `SHA256SUMS` | `$R/data/downloads` |
| Unzipped raw data | `$R/data/raw/{SyRIP,syrip_for_train}` |
| Prepared dataset (upstream layout) | `$R/data/syrip` |
| Pretrained checkpoints | `$R/models` |
| Conda envs (prefix envs) | `$R/envs/fidip`, `$R/envs/mmpose` |
| Runs | `$R/runs/<RUN_ID>/` |
| Slurm logs | `$R/slurm_logs/%x_%j.out` |
| Persistent copies (backed-up home, 75 GB quota) | `~/fidip_artifacts/` |

Scratch is purged after 28 days and is not backed up → final checkpoints + metrics are copied to `~/fidip_artifacts/`.

## Environments
```bash
module load miniconda3/25.9.1 cuda/12.1.1      # also available: anaconda3/2024.06, cuda/12.3.0, cuda/12.8.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/purohitham.v/fidip/envs/fidip     # upstream FiDIP code (HRNet/DarkPose)
conda activate /scratch/purohitham.v/fidip/envs/mmpose    # improvement experiments (MMPose)
```
Exact package lists are exported to `envs/*.yml` / `envs/*-pip-freeze.txt` in the repo.
**Always `export PYTHONNOUSERSITE=1`.** Otherwise `~/.local/lib/python3.x` (transformers, datasets, absl …) leaks into every conda env of the same Python version. The fidip env sets it as a conda env var; set it in every sbatch script as well.

## Partitions (from `sinfo`, 2026-11-05)
| Partition | Time limit | GPUs | Use it for |
|---|---|---|---|
| `gpu` | 8 h | h200 (4 nodes × 8), a100, v100-sxm2, v100-pcie, t4 | training runs |
| `sharing` | 1 h | a100, h100, l40, a6000, a5000, v100, quadro | smoke tests, evaluations, demos |
| `short` (default) | 2 d | 1 v100-sxm2 node + CPU nodes | CPU jobs (analysis, data prep) |

Request a GPU with `--gres=gpu:h200:1` (or `a100:1`, `v100-sxm2:1`). Templates: `scripts/slurm/*.sbatch`.

**Per-user QOS caps** (`sacctmgr show qos`, 2026-11-05). All of the project's streams share user `purohitham.v`.

| Partition | Max running | Max submitted | Max GPUs | Wall time |
|---|---|---|---|---|
| `gpu` | 4 | 8 | 4 | 8 h |
| `sharing` | 2 | 4 | 2 | 1 h |

Over the limit, jobs sit in `PENDING (QOSMaxJobsPerUserLimit)`.

**Slot policy (D14):**
- Repro (`rp_` job prefix): gpu ≤ 2 running / ≤ 4 submitted; sharing ≤ 1 / ≤ 2.
- Improve (`im_`): same as repro.
- Analysis (`an_`): CPU partitions only.

**Making the most of the slots:**
- Pack several trainings into one GPU job (concurrently if memory allows, otherwise back-to-back).
- Use `sharing` L40S/L40/A6000 (44–48 GB) for evals and smoke tests; they are often idle.
- **MMPose / mmcv 2.1.0 has no sm_90 kernels.** Run MMPose jobs on A100/L40S/A6000, not H100/H200.
- Count a stream's jobs with: `squeue -u $USER -h -o "%j %T %P" | grep '^rp_'`.

## Common operations
```bash
# sync code to the cluster: tar-over-ssh (macOS openrsync is incompatible with the cluster's rsync 3.2.5).
# Never deletes remote files, because compiled libs and other build products live only in the cluster copy.
bash scripts/sync_to_cluster.sh
# pull small run artifacts (run.json, metrics/preds JSON, logs, PNGs; never checkpoints) -> results/cluster_runs/
bash scripts/sync_results.sh
# submit
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch scripts/slurm/<job>.sbatch'
# chain short jobs
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && j1=$(sbatch --parsable a.sbatch) && sbatch --dependency=afterok:$j1 b.sbatch'
# poll
ssh explorer 'squeue -u $USER --format="%.10i %.10P %.32j %.2t %.10M %.6D %R"'
# tail a log
ssh explorer 'tail -50 /scratch/purohitham.v/fidip/slurm_logs/<name>_<JOBID>.out'
# after it ends
ssh explorer 'sacct -j <JOBID> --format=JobID,JobName%32,State,ExitCode,Elapsed,NodeList' ; ssh explorer 'seff <JOBID>'
# pull small results (metrics/preds/figures only)
bash scripts/sync_results.sh
```

## Run conventions
- `RUN_ID = R<NNN>_<model>_<variant>_s<seed>` (e.g. `R010_fidip-w48_retrain_s1`).
- Each run dir holds `config.*`, `train.log`, `run.json` (git SHA, job ID, node, GPU, seed, start/end, checkpoint SHA256), `preds_{val400,test100}.json`, `metrics_{val400,test100}.json`.
- Every run gets a row in `results/registry.csv` and in `MANIFEST.md`.
- Model selection uses **Val400 only**. **Test100** is evaluated only for configs listed in `results/PREREGISTRATION.md`.
- All reported metrics come from `eval/eval_coco.py`.

## Active runs
Snapshot as of 2026-11-06 07:33 EST. Refresh with `ssh explorer 'squeue -u $USER'`.

| Job ID | Name | Partition / GPU | Purpose | State |
|---|---|---|---|---|
| 11671810 | rp_R021-R023 | gpu / h200 | plain fine-tune control (PRE_EPOCH 0, λ 0.0), 3 seeds (repro-3) | running |
| 11671549 | im_pack_confA_s2-s3 | gpu / h200 (moved from a100, D22) | HRNet R131 + R132 (arm A) | pending |
| 11671553 | im_pack_confA_s4-s5 | gpu / h200 (moved from a100, D22) | HRNet R133 + R134 (arm A) | pending |
| 11671548 | im_pack_confH_s2-s4 | gpu / h200 (moved from a100, D22) | ViT-H-256 R136–R138, back-to-back (arm B) | pending |
| 11671556 | im_pack_confAB_s6s5 | sharing / l40s | HRNet R135 + ViT-H R139 | running |
| 11671737 | im_R140 | sharing / l40s | ViT-H-256 R140 (arm B, s6) | running |

**Next:** once all ten of R131–R140 have finalized, run one `final_test_eval_mmpose.sbatch` over R101, R105 and R131–R140, then `ensemble_mmpose.sbatch` (R141), then the analysis refresh and contrasts C1–C4 (MANIFEST §10).

## Guardrails
- No training or evaluation on login nodes. Use `sbatch`/`srun`. **Login nodes kill heavy processes:** an `unzip` of the 313 MB SyRIP.zip was `Killed` (2026-11-05). Only `curl`/`wget` downloads and light file operations are safe there.
- **Confirm with the human before `scancel`** on any job that has run for more than 30 minutes.
- **Run jobs from a per-job code snapshot** (D19). Copy `$R/repo` to `$R/snapshots/$SLURM_JOB_ID` at job start, run from there, and record `git rev-parse HEAD^{tree}` in `run.json`. Concurrent syncs used to rewrite `$R/repo` files under running jobs: bash reads scripts incrementally, so jobs truncated or misbehaved. `scripts/sync_to_cluster.sh` now unpacks into a private staging dir and copies it over with the cluster's `rsync`. rsync renames each file into place atomically, so running jobs keep reading the old inode.
- Never tune on Test100 (validate100).
- Never put big data in `/home` or `/projects`.
- Never print or commit `.env`.
- Smoke-test every new sbatch script on `sharing` before submitting long jobs.
- Save a checkpoint every epoch so a job can resume after a timeout (`--dependency` chains).
