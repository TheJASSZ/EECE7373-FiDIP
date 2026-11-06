# MANIFEST.md — inventory and provenance

This file lists what exists, where it lives, and **exactly which run produced each reported number**.
Cluster root: `R=/scratch/purohitham.v/fidip`. Persistent copies: `explorer:~/fidip_artifacts/`.

## 1. Code
| Item | Location | Version / commit |
|---|---|---|
| This project repo | `~/Documents/EECE7373-FiDIP` (git), mirrored to `$R/repo` | see `git log` |
| Upstream FiDIP code | `third_party/Infant-Pose-Estimation` (vendored) | github.com/ostadabbas/Infant-Pose-Estimation @ _TBD_ |
| Our patches to upstream | separate commits touching `third_party/` | listed in §6 |

## 2. Environments
| Name | Path | Python / PyTorch / CUDA | Spec file |
|---|---|---|---|
| fidip | `$R/envs/fidip` | _TBD_ | `envs/fidip.yml`, `envs/fidip-pip-freeze.txt` |
| mmpose | `$R/envs/mmpose` | _TBD_ | `envs/mmpose.yml`, `envs/mmpose-pip-freeze.txt` |

## 3. Datasets
| Name | Source URL | Size | SHA256 | Cluster path |
|---|---|---|---|---|
| SyRIP (original) | https://coe.northeastern.edu/Research/AClab/SyRIP/SyRIP.zip | 313 MB | _TBD_ | `$R/data/raw/SyRIP` |
| SyRIP for FiDIP training | https://coe.northeastern.edu/Research/AClab/SyRIP/syrip_for_train.zip | 608 MB | _TBD_ | `$R/data/raw/syrip_for_train` |
| Prepared layout used by code | built from the above by `scripts/get_data.sh` | — | — | `$R/data/syrip` |

### Splits
| Split | Definition | # images | ID list (in repo) | Used for |
|---|---|---|---|---|
| Train | official `train_infant` (200 real + 1000 synthetic) | 1200 | _TBD_ | training only |
| **Val400** | validate500 \ validate100 (checked disjoint by filename + image hash) | 400 | `splits/val400_image_ids.txt` | model/epoch/hparam selection only |
| **Test100** | official validate100 (= paper's Test100) | 100 | `splits/test100_image_ids.txt` | final, pre-registered reporting only |

## 4. Pretrained checkpoints (downloaded)
| File | Source | SHA256 | Path |
|---|---|---|---|
| _TBD_ | | | |

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
| Report table / figure | Number | RUN_ID | Metrics file |
|---|---|---|---|
| _TBD_ | | | |
