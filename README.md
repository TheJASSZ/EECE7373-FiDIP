# Reproducing and Improving FiDIP for Infant Pose Estimation on SyRIP

**Vishnu Purohitham** · EECE 7373 Machine Learning with Small Data, Northeastern University, Fall 2026 · Report: [`report/FiDIP_Report.pdf`](report/FiDIP_Report.pdf)

This repository reproduces FiDIP, the Fine-tuned Domain-adapted Infant Pose model of Huang, Fu, Liu & Ostadabbas (IEEE FG 2021, [arXiv 2010.06100](https://arxiv.org/abs/2010.06100); [official code](https://github.com/ostadabbas/Infant-Pose-Estimation)), on the [SyRIP dataset](https://coe.northeastern.edu/Research/AClab/SyRIP/), and then tries to beat it on the same split.

All work ran on Northeastern's Explorer cluster (Slurm), driven by an AI coding agent (Claude Code), with the student approving and steering. Three rules held throughout:
- **One evaluator.** Every number comes from a single audited evaluator, [`eval/eval_coco.py`](eval/eval_coco.py).
- **No test-set selection.** Models are selected only on **Val400**, the 397 validate500 images that neither belong to nor duplicate Test100. **Test100** is evaluated once per pre-registered configuration.
- **Live documentation.** The agent kept [`SKILL.md`](SKILL.md), [`AGENT.md`](AGENT.md) and [`MANIFEST.md`](MANIFEST.md) current while it worked.

## Results summary

SyRIP Test100, GT person boxes, flip test. Metrics are `eval_coco.py` under the `paper` protocol; the `fixed` protocol is in brackets (see below).

| Model | Published AP | Our AP on Test100 | Source |
|---|---|---|---|
| DarkPose + FiDIP, HRNet-W48 384×288, released checkpoint (R001) | 93.6 | **93.6** [95.0]; AP50/AP75 98.5/98.5; AR 94.6 (exact match) | `results/tables/reproduction.csv` |
| Pose-MobileNet + FiDIP, released checkpoint (R002) | 79.2 | 79.1 | `results/tables/reproduction.csv` |
| DarkPose HRNet-W48, COCO only, no infant training (R003) | 88.5 | 88.5; AP50/AP75 98.5/98.5; AR 90.0 (exactly the paper's row) | `$R/runs/R003_darkpose-w48-coco_released/metrics_upstream_test100.txt` |
| Our FiDIP retrain, λ = 5e-4, 3 seeds, Val400-selected epoch (R010–R012) | 93.6 | 93.6 ± 0.4; paired ΔAP vs. R001 −0.05 [−0.67, +0.65] | `results/tables/bootstrap_summary.csv` |
| Same recipe without domain adaptation, λ = 0.0 (R013–R015) | 92.7 (Finetune) | 93.6 ± 0.1 (Val400 90.8 ± 0.2); FiDIP − FT ΔAP −0.04 [−0.34, +0.20], so the +0.9 DA gain **does not reproduce** | `results/analysis_eval/R013–R015`, `results/tables/bootstrap_retrain_fidip_vs_ft_test100.json` |
| Plain fine-tune, no classifier pre-stage, λ = 0.0 (R021–R023, repro-3) | 92.7 (Finetune) | 93.7 ± 0.2 (Val400 90.9 ± 0.2); FiDIP − plain FT ΔAP −0.15 [−0.43, +0.15] | `results/analysis_eval/R021–R023`, `results/tables/bootstrap_*` |
| Improvement: ViTPose-H 256×192, pre-registered arm B (R136–R140, 5 fresh seeds) | — | see report §5.4 and `results/tables/main_comparison.csv` | `results/PREREGISTRATION.md` (D20) |

The report's §5.4 states the published baseline, our result and the exact evaluation conditions in one place (the highlighted box).

### Findings that change the numbers

| Finding | Effect | Decision |
|---|---|---|
| validate500 and the training file ship a constant placeholder `area` | Val400 AP of R001 would read 98.9 instead of 90.4 | recompute `area = w·h` (D8) |
| Test100 annotation id 0 is never matched by pycocotools | AR capped at 0.99; `fixed` − `paper` = +1.4 AP for R001 | report both protocols; a claim must hold under both (D10, D13) |
| Test100 images 20, 42, 46, 50 are byte copies of training images; test7 is a near-copy | Test100-clean (95 images): −0.8 AP for R001 | keep Test100 as shipped; report Test100-clean (D16) |
| Three validate500 images duplicate Test100 images | would leak into model selection | removed, so Val400 has 397 images (D15) |
| The README's `syrip_for_train.zip` puts MINI-RGBD labels on all 1000 synthetic images; its `test11.jpg` is a copy of `test24.jpg` | −1.2 Val400 AP when training on those labels (R020); −1.5 AP from the test11 copy | use `SyRIP.zip` `200R_1000S` labels and images (D17) |
| The released training loop selects `model_best` on Test100; the README command trains from random weights with λ = 0 | the published protocol is not the README command | pass `TRAIN.CHECKPOINT` and `TRAIN.LAMBDA`; select on Val400 (D1, D5) |

Sources for these figures: `report/generated/numbers.tex` (mirrored in `results/tables/provenance.csv`) and `report/generated/manual_numbers.tex` (mirrored in `results/tables/provenance_manual.csv`).

## Quickstart (STUDENT_GUIDE Parts 1–6)

The complete, tested command sequence is **[MANIFEST.md §10 "Reproduce everything"](MANIFEST.md)**. The outline below follows the guide's parts. It uses the cluster root `R=/scratch/purohitham.v/fidip`, and the cluster is reached as `ssh -o BatchMode=yes explorer '...'` (see [`SKILL.md`](SKILL.md)).

**Part 0: access.** You need passwordless SSH to Explorer. Never run compute on the login node.

**Part 1: environment.**
- Push the code (tar over ssh; macOS `openrsync` cannot talk to the cluster's rsync): `bash scripts/sync_to_cluster.sh`, or `bash scripts/push_mmpose.sh`, which also records git provenance.
- Build the FiDIP env: `sbatch scripts/slurm/build_env_fidip.sbatch`. It is pip-only (py3.12, torch 2.3.1 cu121); upstream `fidip_env.yml` has unbuildable pins.
- Build the MMPose env: `sbatch scripts/slurm/build_env_mmpose.sbatch` (py3.10, torch 2.1.2, mmpose 1.3.2).
- Build the analysis env: `sbatch scripts/slurm/build_env_analysis.sbatch`.
- Check the GPU: `python gpu_test.py`, run in `third_party/Infant-Pose-Estimation/` inside a GPU job.
- Set `PYTHONNOUSERSITE=1` in every job.

**Part 2: data.**
- `sbatch scripts/slurm/get_data.sbatch` downloads, SHA256-verifies and unzips on a compute node, then builds the upstream layout and the split files.
- Then check `$R/data/SPLITS_READY` and `$R/data/splits/VERIFY.txt`.
- The main SyRIP zips need **no password**.

**Part 3: pretrained models.** `sbatch scripts/slurm/get_models.sbatch` fetches `hrnet_fidip.pth`, `mobile_fidip.pth` and COCO `w48_384x288.pth` with a strict-load check. `sbatch scripts/slurm/get_ckpts_mmpose.sbatch` fetches the MMPose COCO checkpoints.

**Part 4: train.** The FiDIP retrain is the upstream README command plus the overrides it needs: `TRAIN.CHECKPOINT`, `TRAIN.LAMBDA`, `TRAIN.PRE_EPOCH`, `DATASET.TEST_SET` (Val400, so selection never sees Test100) and `SEED`. Note that λ must be a float:
```bash
python tools/train_adaptive_model_hrnet.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  TRAIN.CHECKPOINT models/coco/w48_384x288.pth TRAIN.LAMBDA 0.0005 TRAIN.PRE_EPOCH 10 DATASET.TEST_SET val400 SEED 1
```
On the cluster this is wrapped by `scripts/slurm/train_fidip_pack.sbatch`, which packs 3 seeds on one H200; the exact lines are in MANIFEST §10.
MMPose runs: `bash scripts/launch_mmpose.sh <RUN_ID> mmpose_exps/configs/<config>.py <seed> "" -p sharing --gres=gpu:l40s:1 -t 00:58:00` (the MMPose arms ran on L40S, as recorded per run).
Over- and underfitting are monitored with the per-epoch `epochs.csv` (train loss, Val400 loss and AP); curves are in `results/figures/curves/`.

**Part 5: test.**
- Released checkpoints: `sbatch scripts/slurm/eval_released.sbatch`.
- Pre-registered MMPose runs, one shot each, gated by `results/PREREGISTRATION.md`: `scripts/slurm/final_test_eval_mmpose.sbatch`.
- Score any prediction file:
  ```bash
  python eval/eval_coco.py --gt $R/data/splits/test100.json --pred runs/<RUN_ID>/preds_test100.json --out metrics_test100.json
  python eval/bootstrap_ci.py paired --gt $R/data/splits/test100.json --a A.json --b B.json --out ci.json
  ```

**Part 6: demo** (fidip env):
```bash
python tools/demo.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml --model $R/models/hrnet_fidip.pth \
  --image <img|dir> --output <out> [--conf-threshold 0.3] [--output-size 512,512] [--video in.mp4]
FIDIP_MODEL=$R/models/hrnet_fidip.pth streamlit run tools/streamlit_demo.py
```

**Tables, figures and report.**
```bash
bash scripts/pull_analysis.sh && python3 scripts/collect_results.py   # -> results/tables/*, report/generated/*
bash report/build.sh                                                  # -> report/FiDIP_Report.pdf (needs tectonic)
python -m pytest eval/tests -q                                        # evaluator and analysis unit tests
python3 report/check_bib.py report/refs.bib                           # re-verify the bibliography DOIs
```

### Where this repo differs from the STUDENT_GUIDE and upstream
- The guide's `fidip_simple.yml`, `gpu_test.py`, `tools/demo.py` and `tools/streamlit_demo.py` are **not in upstream**. We wrote `gpu_test.py` and both demos from the guide's documented interface. The env is pip-only instead of `fidip_simple.yml`.
- Run literally, the upstream README training command trains from random weights with λ = 0. **It needs `TRAIN.CHECKPOINT` and `TRAIN.LAMBDA`**, plus `TRAIN.PRE_EPOCH 10` for the classifier pre-stage and `DATASET.TEST_SET val400` so that `model_best` is not chosen on Test100.
- The main SyRIP zips downloaded **without a password**; only Expanded SyRIP is protected. The adult `train_pre_infant` set ships inside `syrip_for_train.zip`.
- Upstream patches P1–P11 are separate commits, listed in MANIFEST §6.

## Repository map

| Path | Contents |
|---|---|
| `SKILL.md`, `AGENT.md`, `MANIFEST.md` | cluster how-to · running lab notebook (decisions D1–D22) · inventory, provenance, reproduce-everything commands (§10) |
| `CLAUDE.md` | project rules every agent loads (no Test100 selection, one evaluator, logging, secrets) |
| `report/` | LaTeX source (`report.tex`, `sections/`, TikZ `figures/`), `refs.bib`, `generated/` (script-written numbers and tables), `build.sh`, `FiDIP_Report.pdf` |
| `eval/` | `eval_coco.py` (the evaluator), `bootstrap_ci.py`, `ensemble_preds.py`, `taxonomy.py`, `analyze.py`, `visualize.py`, `tests/` |
| `scripts/` | data prep (`get_data.sh`, `make_splits.py`, `near_dups.py`), sync, launchers, `collect_results.py`, `slurm/*.sbatch` |
| `mmpose_exps/` | MMPose configs (arms A/B, screening, ablations) and tools |
| `third_party/Infant-Pose-Estimation/` | vendored upstream FiDIP code at `a76b4f4`, with small separately committed patches, the demos and `gpu_test.py` |
| `results/` | `registry.csv` (one row per run), `PREREGISTRATION.md`, `tables/`, `figures/` |
| `splits/` | Val400/Test100 id lists and the verification report |
| `envs/` | exported environment specs |
| `notebook/` | per-stream logs (repro, improve, analysis, report) |
| `docs/REQUIREMENTS_CHECKLIST.md` | rubric and guide requirements traced to where they are met |

## Documentation and integrity
- Report: [`report/FiDIP_Report.pdf`](report/FiDIP_Report.pdf). The rubric-to-section map is in its appendix.
- Agent operation: [`SKILL.md`](SKILL.md) · lab notebook: [`AGENT.md`](AGENT.md) · provenance: [`MANIFEST.md`](MANIFEST.md) · pre-registration: [`results/PREREGISTRATION.md`](results/PREREGISTRATION.md).
- **AI-tool use.** Claude Code, with the Claude Opus 5.5 model, was used as a tool, orchestrating parallel worker agents. The student set goals, approved the plan, steered and reviewed. The report's §9 describes what the agent did, what the human did, and how outputs were verified.
- **Integrity.** No other student's repository was opened or used.
- **License.** Upstream FiDIP code is for non-commercial use only (see its README). SyRIP is distributed by the Augmented Cognition Lab, Northeastern University.
