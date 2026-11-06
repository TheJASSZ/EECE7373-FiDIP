# MANIFEST.md — inventory and provenance

This file lists what exists, where it lives, and **exactly which run produced each reported number**.
Cluster root: `R=/scratch/purohitham.v/fidip`. Persistent copies: `explorer:~/fidip_artifacts/`.

## 1. Code
| Item | Location | Version / commit |
|---|---|---|
| This project repo | `~/Documents/EECE7373-FiDIP` (git), mirrored to `$R/repo` | see `git log` |
| Upstream FiDIP code | `third_party/Infant-Pose-Estimation` (vendored from the GitHub tarball API; git commit `cf33818` = unmodified import) | github.com/ostadabbas/Infant-Pose-Estimation @ `a76b4f418107b8d3548d9779e94702e2508fbef1` (2024-07-05) |
| Our patches to upstream | separate commits touching `third_party/` | listed in §6 |

## 2. Environments
| Name | Path | Python / PyTorch / CUDA | Spec file |
|---|---|---|---|
| fidip | `$R/envs/fidip` | py 3.12.4 / torch 2.3.1+cu121 / cuDNN 8902 / numpy 1.26.4 / pyarrow 17.0.0 (pip-only route, D6; built by `scripts/slurm/build_env_fidip.sbatch`). Conda env var `PYTHONNOUSERSITE=1`. | `envs/fidip.yml`, `envs/fidip-pip-freeze.txt` |
| mmpose | `$R/envs/mmpose` | py 3.10 / torch 2.1.2+cu121 / mmcv 2.1.0 / mmengine 0.10.7 / mmpose 1.3.2 / mmpretrain 1.2.0 / mmdet 3.3.0 (imported unconditionally by mmpose) / numpy 1.26.4. mmpose source: `$R/mmpose`, tag v1.3.2 (5408bc76). mmcv has **no sm_90 kernels**, so MMPose runs use A100/L40S. `PYTHONNOUSERSITE=1`. | `envs/mmpose.yml`, `envs/mmpose-pip-freeze.txt` |

## 3. Datasets
| Name | Source URL | Size | SHA256 | Cluster path |
|---|---|---|---|---|
| SyRIP (original) | https://coe.northeastern.edu/Research/AClab/SyRIP/SyRIP.zip | 327,912,103 B | `1d13e816375988be9bbdab8bf5be6562a5b9bfa83684d57ccbfec9dd619e1692` | `$R/data/raw/SyRIP` |
| SyRIP for FiDIP training | https://coe.northeastern.edu/Research/AClab/SyRIP/syrip_for_train.zip | 637,172,858 B | `1f267878732fb81776c5c7f153bad45d652cadf5bfd83761971390b64d6cee1b` | `$R/data/raw/syrip_for_train` |
| SyRIP README | https://coe.northeastern.edu/Research/AClab/SyRIP/README.md | 4,256 B | `2b7ebd8353917fd5fb1625208054c063036e3b6b1ebcef8a75b0add6e1a99b93` | `$R/data/downloads/README.md` |

Downloaded 2026-11-05 19:52–19:53 EST to `$R/data/downloads/`. The checksums are in `$R/data/downloads/SHA256SUMS`.

**Raw-tree check (repro worker, 21:00 EST):** all 7666 zip members match by name, size and CRC32, with no extra files. 120 of 120 sampled members also match by md5 after decompression.

**Prepared layout:** `$R/data/syrip`, built by `scripts/get_data.sh` and `scripts/make_splits.py`. The image for split X is at `$R/data/syrip/images/X/<file_name>`. Upstream `DATASET.ROOT = $R/data/syrip`, with `TEST_SET` ∈ {`test100`, `val400`, `validate500`, `validate_infant`} (patch P10).

**Which source each split uses (D17):**
- **Training labels:** `SyRIP.zip` `annotations/200R_1000S`.
- **Not used:** `syrip_for_train.zip`'s `person_keypoints_train_infant.json`. Its image bytes are identical to SyRIP.zip, but all **1000 synthetic entries carry MINI-RGBD labels**, off by a median of 191 px.
- **Test images:** come from `SyRIP.zip`. syrip_for_train's `validate_infant/test11.jpg` is a byte copy of `test24.jpg`.
- **Unannotated images:** `SyRIP.zip` `images/train_infant/` also holds 300 unannotated real images, 199 of them byte-identical to validate500 images. Only images listed in the JSON are used.

### Splits (final 2026-11-05 21:02:55 EST, marker `$R/data/SPLITS_READY`, report `$R/data/splits/VERIFY.txt`)
| Split | Definition | # images | File + SHA256 | Used for |
|---|---|---|---|---|
| Train | SyRIP.zip `200R_1000S` (200 real + 1000 synthetic). `area` recomputed as w×h, because the shipped file has the constant 14232.77 (D8). | 1200 | `$R/data/splits/train_infant.json` `3bdcce0b5a59f0d97e0782221f8d33a47653666cb4c59b8a68d70b1901dbe903` | training only |
| **Val400** | validate500 \ validate100, **minus 3 images that duplicate Test100 images**: 217 (md5 = test56), 222 (md5 = test7), 207 (near-duplicate of test47: dHash 1, pose-OKS 0.87). Rule fixed in advance: exact md5, or dHash ≤ 10 and pose-OKS ≥ 0.85. Area = w×h (D8). | **397** | `$R/data/splits/val400.json` `8ed74bd53ef8cd6981523c9ba9b386ef3c9c77bce8dd7e376045c12e0da23a4b`; exclusions in `splits/val400_exclude_near_dups.txt` | model / epoch / hparam selection only (D1, D15) |
| **Test100** | official validate100 = the paper's Test100, **byte copy as shipped** (ids as shipped, so min annotation id 0: D10) | 100 | `$R/data/splits/test100.json` `8b299a879a95913483498e3f2115a826ac23ebc5262c3efb8d65f63718e3194c` | final reporting only (released, reproduction, pre-registered) |
| Test100-clean | Test100 minus {7, 20, 42, 46, 50}. These 5 images also appear in the official train set: 4 byte-identical (20↔73, 42↔315, 46↔272, 50↔340) and 1 near-identical (7↔47, pose-OKS 0.96). | 95 | derived from test100.json by the evaluator | sensitivity check only (D16) |
| validate500 | official validate500, area = w×h | 500 | `$R/data/splits/validate500.json` `d129e2f50d9611eeee7f7675187049f3c4a3d65569efd1d305ed99520676f85e` | released-checkpoint box-size analysis only. Never reported as a clean test set, because Val400 ⊂ validate500. |
| train_pre_infant | shipped in syrip_for_train.zip: adult real (COCO) + synthetic (SURREAL) | 4057 (2029 real + 2028 synthetic) | `$R/data/raw/syrip_for_train/syrip/annotations/person_keypoints_train_pre_infant.json` | FiDIP domain-classifier pre-stage (PRE_EPOCH 10) |

## 4. Pretrained checkpoints (downloaded)
| File | Source | SHA256 | Path |
|---|---|---|---|
| `hrnet_fidip.pth`: released DarkPose+FiDIP HRNet-W48 | FiDIP_models Google Drive folder `108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC`, file id `19tBMoVS8wTza7VqVfPc6KBQ_POPAEIqb` | `bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07` | `$R/models/hrnet_fidip.pth` |
| `mobile_fidip.pth`: released Pose-MobileNet+FiDIP | same folder, file id `1aHaBwO3XBYtFXI7cDMRIzMWRffIJzrQU` | `bd0b7548b1f1158f0a8a97d022305547343c835e7d4027727a978d2d7a4cbca1` | `$R/models/mobile_fidip.pth` |
| `w48_384x288.pth`: COCO DarkPose HRNet-W48, the FiDIP init | TGA_models Google Drive folder `14kAA1zXuKODYgrRiQmKnVcipbY7RedVV` (file `coco/w48_384×288.pth`, id `1TKvk0R_qrWV3UeSTawUYM5Burp-lOmss`) | `c730334eef7c47f051f694949ad2f213526723e46a7781178deb0fc90a93635c` | `$R/models/coco/w48_384x288.pth` |
| MMPose HRNet-W48 DARK 384×288 (COCO AP 77.2) | `https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth` | `39c3c3816ec3f6cd180e27fb79435d9cbd878acd37317bbbd942a97ddbb36ec8` | `$R/models/mmpose/` |
| MMPose HRNet-W48 UDP 384×288 (COCO AP 77.3) | `…/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth` | `70d7ab0196e40b61e4508f08e547576dee9ec90f069ec5a8520f6126b8b3b4a1` | `$R/models/mmpose/` |
| MMPose ViTPose-L 256×192 (COCO AP 78.2) | `…/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth` | `53609f55473dd4d26aa2e509911fc092a3e4d438e9eb75126be6f905b693c475` | `$R/models/mmpose/` |
| MMPose ViTPose-H 256×192 (COCO AP 78.8) | `…/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth` | `e32adcd41ab0b0ef0b5bf3d167ddae7cdbd45fcf45e7f6a834815ef04d641f2b` | `$R/models/mmpose/` |
| ViTPose-L with pos-embed resized to 24×18 (for 384×288) | derived by `mmpose_exps/tools/resize_vit_pos_embed.py` | `627f26a2c948b301dc012114145f834b9dddf41ce3aa639f7cb509727eb6ab6f` | `$R/models/mmpose/derived/` |
| ViTPose-H with pos-embed resized to 24×18 | derived, same script | `f8c4b2df396ec054a7857f497289913176cd00e919fd992266a97e0f72a0e712` | `$R/models/mmpose/derived/` |

All three Drive checkpoints load with `strict=True`; none has missing/unexpected keys or a `module.` prefix.

## 5. Trained checkpoints (ours)
- **Per-run checkpoint:** every run's selected checkpoint and its SHA256 are recorded in `$R/runs/<RUN_ID>/run.json`. That file also holds the git head and tree, the dirty flag, the Slurm job, node, GPU and seed.
  - FiDIP retrains keep both `ckpt_best_val400.pth` and `ckpt_final.pth`.
  - MMPose runs keep `best_coco_AP_epoch_<e>.pth`.
- **Persistent copies:** `explorer:~/fidip_artifacts/runs/` (home, backed up), SHA-verified. For FiDIP R010–R015 that is 5.3 GB.
- **Final table:** generated from the run.json files when results are final (§8).

## 6. Configs and patches
**Upstream patches.** Each one is a separate commit touching `third_party/Infant-Pose-Estimation` and marked `[EECE7373 patch Pn]`.

| Patch | Commit | Change | Why |
|---|---|---|---|
| P1 | 7168b1e | compiled-nms imports made optional | only the pure-Python OKS-NMS is used, so `make`/nvcc is unnecessary |
| P3 | 4f3f869 | `torch.load(map_location, weights_only=False)`; strict loading at test time | stops a silent `strict=False` mis-load |
| P4 | 7ec379a | build the train_pre loader only if `PRE_EPOCH > 0` | the shipped code always needed its JSON |
| P5 | 67fde20 | `is '*'` → `== '*'` | SyntaxWarning, Python 3.12 |
| P6 | 763a36f | `mkdir(parents=True, exist_ok=True)` | nested OUTPUT_DIR |
| P7 | 25cb420 | loss `.npy` files go into the run dir; no `add_graph` | the cwd was being polluted; the slow CPU trace is removed |
| P8 | 44dcb09 | per-epoch `epochs.csv` (train MSE, Val400 loss, AP/AR) | over/underfitting monitoring (STUDENT_GUIDE Part 4.5) |
| P10 | fe6d81c | any split name (test100, val400, validate500), plus eval of labelled `test*` splits | upstream hard-codes split names |
| P11 | 596fb7b | `SEED` option | repeatable seeds |
| demo | 7e68cfe | added `tools/demo.py`, `tools/streamlit_demo.py`, `gpu_test.py` | STUDENT_GUIDE Parts 1.3 and 6; these files are not in upstream |

**Configs:**
- **FiDIP:** the upstream infant yaml `experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml`, with command-line overrides from `scripts/train_fidip.sh`: `TRAIN.CHECKPOINT` COCO init, `TRAIN.PRE_EPOCH 10` (0 for R021–R023), `TRAIN.LAMBDA`, `DATASET.TEST_SET val400`, `SEED`.
- **MMPose:**
  - Base: `mmpose_exps/configs/_base_/syrip_base.py`.
  - Arm A: `syrip_hrnet-w48_dark-384x288.py`. Arm B: `syrip_vitpose-h_udp-256x192.py`.
  - Screening: `syrip_vitpose-{l,h}_udp-{256x192,384x288}.py`.
  - Ablation variants: `_realonly`, `_rot180`, `_cosine`, `_da`.
- **Derived train subsets** (1:1 real:synthetic mixing), in `$R/mmpose_data/splits/`: real200 `081b5664…c879` and syn1000 `ee0c5b3b…9210`, both from `train_infant.json` (`3bdcce0b…`).

## 7. Runs
See `results/registry.csv` (one row per run: run ID, Slurm job ID, config, seed, GPU, git SHA, Val400 AP, Test100 AP, checkpoint).

## 8. Reported numbers → provenance
- Every number in the report enters through `report/generated/numbers.tex` (`\setres{key}{value}`) and `report/generated/tab_*.tex`. Both are written by `scripts/collect_results.py` from `eval/eval_coco.py` and `eval/bootstrap_ci.py` outputs. Published baselines (cited) are typed directly. A few values without a generated key are transcribed into `report/generated/manual_numbers.tex`, each with its source file or log entry noted (mirrored in `results/tables/provenance_manual.csv`). The final computed keys and the H1 verdict come from `scripts/fill_final_keys.py` (`results/tables/provenance_final_keys.csv`).
- `results/tables/provenance.csv` maps each report key to its value, RUN_ID(s), metrics file, protocol (`paper`/`fixed`) and split. The table below is generated from it once results are final.

| Report key | Value | RUN_ID(s) | Metrics file | Protocol / split |
|---|---|---|---|---|
| _generated at finalization_ | | | | |

## 9. Report build
| What | Command |
|---|---|
| Build the PDF (`report/FiDIP_Report.pdf`) | `bash report/build.sh` (Tectonic) |
| Re-verify citations (31 entries; the 22 DOIs are checked against Crossref) | `python3 report/check_bib.py` |
| Regenerate results tables, numbers and provenance | `python3 scripts/collect_results.py` |

## 10. Reproduce everything (from the local repo root; cluster = `ssh explorer`)
```bash
# 0. push code (tar-over-ssh with atomic per-file rename) and record git provenance on the cluster
bash scripts/push_mmpose.sh
# 1. data, environment, checkpoints  (then verify $R/data/SPLITS_READY and $R/data/splits/VERIFY.txt)
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch scripts/slurm/get_data.sbatch && sbatch scripts/slurm/build_env_fidip.sbatch && sbatch scripts/slurm/get_models.sbatch'
# 2. released checkpoints: R001 (HRNet), R002 (MobileNet) on test100 / val400 / validate500 / validate_infant
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch scripts/slurm/eval_released.sbatch'
# 3. FiDIP retrains, 3 seeds per pack on one H200 (spec = "RUN_ID SEED LAMBDA [EVAL_SETS] [yacs opts]")
#    NOTE: lambda must be written as a float (0.0, not 0), or yacs raises a type mismatch
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch -p gpu --gres=gpu:h200:1 -t 02:00:00 -J rp_R010-R012 scripts/slurm/train_fidip_pack.sbatch "R010_fidip-w48_lambda5e-4_s1 1 0.0005" "R011_fidip-w48_lambda5e-4_s2 2 0.0005" "R012_fidip-w48_lambda5e-4_s3 3 0.0005"'
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch -p gpu --gres=gpu:h200:1 -t 02:00:00 -J rp_R013-R015 scripts/slurm/train_fidip_pack.sbatch "R013_fidip-w48_lambda0_s1 1 0.0" "R014_fidip-w48_lambda0_s2 2 0.0" "R015_fidip-w48_lambda0_s3 3 0.0"'
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch -p gpu --gres=gpu:h200:1 -t 01:30:00 -J rp_R021-R023 scripts/slurm/train_fidip_pack.sbatch "R021_fidip-w48_ft-noprestage_s1 1 0.0 test100,val400 TRAIN.PRE_EPOCH 0" "R022_fidip-w48_ft-noprestage_s2 2 0.0 test100,val400 TRAIN.PRE_EPOCH 0" "R023_fidip-w48_ft-noprestage_s3 3 0.0 test100,val400 TRAIN.PRE_EPOCH 0"'
# 4. MMPose: env, checkpoints, data (once)
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch scripts/slurm/build_env_mmpose.sbatch && sbatch scripts/slurm/get_ckpts_mmpose.sbatch && sbatch scripts/slurm/prep_ckpts_mmpose.sbatch && sbatch scripts/slurm/prep_data_mmpose.sbatch'
#    zero-shot COCO models R181–R185 (Val400 only)
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch scripts/slurm/zeroshot_mmpose.sbatch'
#    screening and ablations (seed 1, Val400 only): R101 HRNet, R102 ViT-L 256, R103 ViT-L 384, R104 ViT-H 384, R105 ViT-H 256,
#    R106–R109 LR variants, R120/R122 real-only, R121 rot180. Configs in mmpose_exps/configs/ (names in notebook/improve.md)
#    confirmatory arms (pre-registered imp-1/imp-2; fresh seeds 2–6):
#      arm A: R131–R135  mmpose_exps/configs/syrip_hrnet-w48_dark-384x288.py   seeds 2..6
#      arm B: R136–R140  mmpose_exps/configs/syrip_vitpose-h_udp-256x192.py    seeds 2..6
# 3b. λ sweep and recipe variants (Val400 only): R016–R018 λ ∈ {1e-4, 2e-3, 1e-2}; R019 paper recipe (TRAIN.LR 1e-3 END_EPOCH 100 BATCH 64);
#     R020 README-literal labels. Same train_fidip_pack.sbatch spec format as step 3, with EVAL_SETS=val400
# one MMPose run = launch_mmpose.sh RUN_ID CONFIG SEED "CFG_OPTS" [sbatch args]
bash scripts/launch_mmpose.sh R136_mmp-vitpose-h256_conf_s2 mmpose_exps/configs/syrip_vitpose-h_udp-256x192.py 2 "" -p sharing --gres=gpu:l40s:1 -t 00:58:00
# 5. pre-registered one-shot Test100 eval (gated by results/PREREGISTRATION.md, checkpoint SHA256 verified), then the ensemble
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && RUN_IDS="R101_mmp-hrnetw48_base_s1 R105_mmp-vitpose-h256_base_s1 R131_… R140_…" sbatch --export=ALL scripts/slurm/final_test_eval_mmpose.sbatch'
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch scripts/slurm/ensemble_mmpose.sbatch'
# 6. analysis: evaluate every gated run, contrasts C1–C4, pull results, regenerate the report tables and numbers
ssh explorer 'cd /scratch/purohitham.v/fidip/repo && sbatch scripts/slurm/analysis.sbatch script scripts/analysis/job_refresh.sh --no-released && sbatch scripts/slurm/analysis.sbatch script scripts/analysis/run_contrasts.sh results/contrasts/final_C1_C4.json'
bash scripts/pull_analysis.sh && python3 scripts/collect_results.py && bash report/build.sh
# demo (STUDENT_GUIDE Part 6), in the fidip env
python tools/demo.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml --model $R/models/hrnet_fidip.pth --image <img|dir> --output <out> [--gt-json $R/data/splits/test100.json] [--conf-threshold 0.3] [--output-size 512,512] [--video in.mp4]
FIDIP_MODEL=$R/models/hrnet_fidip.pth streamlit run tools/streamlit_demo.py
```
