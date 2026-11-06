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
| fidip | `$R/envs/fidip` | py 3.12.4 / torch 2.3.1+cu121 / numpy 1.26.4 (pip-only route, D6). Conda env var `PYTHONNOUSERSITE=1`. | `envs/fidip.yml`, `envs/fidip-pip-freeze.txt` |
| mmpose | `$R/envs/mmpose` | _TBD_ | `envs/mmpose.yml`, `envs/mmpose-pip-freeze.txt` |

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
| `hrnet_fidip.pth` (released DarkPose+FiDIP HRNet-W48) | FiDIP_models Google Drive folder `108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC` | `bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07` | `$R/models/hrnet_fidip.pth` |
| `mobile_fidip.pth` (released Pose-MobileNet+FiDIP) | same folder | `bd0b7548b1f1158f0a8a97d022305547343c835e7d4027727a978d2d7a4cbca1` | `$R/models/mobile_fidip.pth` |
| `w48_384x288.pth` (COCO DarkPose HRNet-W48, FiDIP init) | TGA_models Google Drive folder `14kAA1zXuKODYgrRiQmKnVcipbY7RedVV` | `c730334eef7c47f051f694949ad2f213526723e46a7781178deb0fc90a93635c` | `$R/models/coco/w48_384x288.pth` |
| MMPose COCO checkpoints (ViTPose-L/H, HRNet-W48 DARK 384×288) | download.openmmlab.com (URLs in `notebook/improve.md`) | _TBD (improve worker)_ | `$R/models/mmpose/` |

All three Drive checkpoints load with `strict=True`; none has a `module.` prefix (repro worker, 21:0x).

## 5. Trained checkpoints (ours)
| RUN_ID | Checkpoint path | SHA256 | Epoch / selection rule |
|---|---|---|---|
| _none yet_ | | | |

## 6. Configs and patches
| File | Purpose |
|---|---|
| _TBD_ | |

## 7. Runs
See `results/registry.csv` (one row per run: run ID, Slurm job ID, config, seed, GPU, git SHA, Val400 AP, Test100 AP, checkpoint).

## 8. Reported numbers → provenance
- Every number in the report enters through `report/generated/numbers.tex` (`\setres{key}{value}`) and `report/generated/tab_*.tex`. Both are written by `scripts/collect_results.py` from `eval/eval_coco.py` and `eval/bootstrap_ci.py` outputs. Published baselines are the only hand-typed numbers, and they are cited.
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
