# MANIFEST.md — inventory and provenance

This file lists what exists, where it lives, and **exactly which run produced each reported number**.
Cluster root: `R=/scratch/purohitham.v/fidip`. Persistent copies: `explorer:~/fidip_artifacts/`.

## 1. Code
| Item | Location | Version / commit |
|---|---|---|
| This project repo | `~/Documents/EECE7373-FiDIP` (git), mirrored to `$R/repo` | see `git log` |
| Upstream FiDIP code | `third_party/Infant-Pose-Estimation` (vendored from the GitHub tarball API; git commit `91a40f8` = unmodified import) | github.com/ostadabbas/Infant-Pose-Estimation @ `a76b4f418107b8d3548d9779e94702e2508fbef1` (2024-07-05) |
| Our patches to upstream | separate commits touching `third_party/` | listed in §6 |

## 2. Environments
| Name | Path | Python / PyTorch / CUDA | Spec file |
|---|---|---|---|
| fidip | `$R/envs/fidip` | _TBD_ | `envs/fidip.yml`, `envs/fidip-pip-freeze.txt` |
| mmpose | `$R/envs/mmpose` | _TBD_ | `envs/mmpose.yml`, `envs/mmpose-pip-freeze.txt` |

## 3. Datasets
| Name | Source URL | Size | SHA256 | Cluster path |
|---|---|---|---|---|
| SyRIP (original) | https://coe.northeastern.edu/Research/AClab/SyRIP/SyRIP.zip | 327,912,103 B | `1d13e816375988be9bbdab8bf5be6562a5b9bfa83684d57ccbfec9dd619e1692` | `$R/data/raw/SyRIP` |
| SyRIP for FiDIP training | https://coe.northeastern.edu/Research/AClab/SyRIP/syrip_for_train.zip | 637,172,858 B | `1f267878732fb81776c5c7f153bad45d652cadf5bfd83761971390b64d6cee1b` | `$R/data/raw/syrip_for_train` |
| SyRIP README | https://coe.northeastern.edu/Research/AClab/SyRIP/README.md | 4,256 B | `2b7ebd8353917fd5fb1625208054c063036e3b6b1ebcef8a75b0add6e1a99b93` | `$R/data/downloads/README.md` |

Downloaded 2026-11-05 19:52–19:53 EST to `$R/data/downloads/`. The checksums are in `$R/data/downloads/SHA256SUMS`.
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
