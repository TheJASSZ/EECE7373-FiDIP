# notebook/improve.md — live log for the improve stream

Newest entries at the bottom. Timestamps in EST.

### 2026-11-05 20:05 EST — Start of IMPROVE stream
- Read CLAUDE.md, SKILL.md, AGENT.md, MANIFEST.md, results/PREREGISTRATION.md (no rows yet: Test100 is off-limits for my runs).
- Cluster state at start: `$R/envs`, `$R/models`, `$R/runs` empty; `$R/data/SPLITS_READY` absent (REPRO's SyRIP unzip on the login node was killed: `703120 Killed unzip -q -o SyRIP.zip`).
- Plan while data is pending: (1) build `$R/envs/mmpose` inside a `short` CPU job (python 3.10, torch 2.1.2/cu121, mmengine 0.10.x, mmcv 2.1.0, mmpose 1.3.2, mmpretrain for the ViT backbone, numpy<2); (2) clone mmpose v1.3.2 to `$R/mmpose`; (3) download COCO checkpoints in a job; (4) write configs.

### 2026-11-05 20:14 EST — Jobs launched, literature findings folded in
- **Env build:** `sbatch scripts/slurm/build_env_mmpose.sbatch` → job **11659031** (short, c0587). Compute nodes have internet (torch wheel downloaded from download.pytorch.org at ~50 MB/s). conda env from **conda-forge only**: miniconda 25.x refuses the `defaults` channel non-interactively until its ToS is accepted.
  - Pins: `numpy<2`, because torch 2.1 is built against numpy 1.x. `opencv-python<4.12`, because 4.12+ requires numpy>=2.
  - mmpose is installed `--no-deps`, with its runtime deps listed explicitly, so chumpy can't break the resolve. chumpy gets `--no-build-isolation`; failure is non-fatal (SMPL-only).
- **Checkpoints:** `sbatch scripts/slurm/get_ckpts_mmpose.sbatch` → job **11659053** (short). URLs copied from the mmpose **v1.3.2** metafiles (`vitpose_coco.yml`, `hrnet_dark_coco.yml`, `hrnet_udp_coco.yml`). Same 4 files the coordinator's literature scan lists.
- **Coordinator literature scan (received ~20:15). Choices taken from it:**
  - **Control** = HRNet-W48 **DARK** 384x288 (matches FiDIP's DarkPose W48). Skip ViTPose-B, RTMPose, Sapiens: extra pretraining data would raise a fairness question.
  - **Resolution matters:** baseline AP50 = AP75 = 98.5, so the headroom is at strict OKS 0.80-0.95. Screen ViTPose-L at 256x192 *and* 384x288, and ViTPose-H at 384x288. For 384x288, `load_from` the 256x192 COCO checkpoint; mmpretrain resizes `pos_embed`.
  - **Real:synthetic mix 1:1 as in FiDIP.** I verified this in the vendored upstream code: `tools/train_adaptive_model_hrnet.py:190` uses `BalancedBatchSampler(train_dataset, syn_labels)` (`lib/utils/sampler.py`). It oversamples the 200 real images up to 1000 and alternates real/syn, giving a 2000-sample epoch. MMPose equivalent: `ConcatDataset([RepeatDataset(real200, times=5), syn1000])` with shuffle.
  - Upstream infant recipe, verified in `experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml`: `END_EPOCH: 20`, Adam `LR: 0.0001`, `LR_STEP: [40, 200]` (so the LR is constant over the 20 epochs), batch 20, `ROT_FACTOR 45`, `SCALE_FACTOR 0.35`, flip test, GT boxes, `BLUR_KERNEL 11`.
  - The MMPose default `RandomBBoxTransform` (truncnorm×80°, p=0.6; scale [0.5,1.5]) is roughly FiDIP's aug. Rotation ablation: ±40 / ±80 (default) / ±180.
  - Recipe knobs to select on **Val400**: step vs cosine, EMA, epoch budget. LR is set explicitly per batch; no `auto_scale_lr`.
  - If a run takes <30 min, use **5 seeds** for the two finalists. Keep per-run checkpoints and pred dumps for a later (separately pre-registered) ensemble.
  - Eval artifacts D8/D10: val400/validate500 GT areas were recomputed as w×h (the shipped area is a constant placeholder). Test100 annotation id 0 is unmatched in pycocotools/xtcocotools (AR ≤ 0.99). `eval/eval_coco.py` reports both protocols.

### 2026-11-05 20:28 EST — Env built, checkpoints verified, a pos_embed trap found, configs written
- **Env (job 11659031, COMPLETED, 9m51s, c0587):**
  - python 3.10, numpy 1.26.4, torch 2.1.2+cu121 (CUDA 12.1), torchvision 0.16.2+cu121.
  - mmengine 0.10.7, mmcv 2.1.0 (openmmlab cu121/torch2.1 wheel), mmpretrain 1.2.0, mmpose 1.3.2, opencv-python 4.11.0. chumpy built fine with `--no-build-isolation`.
  - mmpose source: `$R/mmpose` = tag v1.3.2, commit `5408bc76f5b848cf925a0d1857899011d8c5b497` (shallow clone).
  - Specs exported to `$R/envs/specs/` (copied to `envs/` in the repo below).
- **Checkpoints (job 11659053, COMPLETED, 4m17s)**, in `$R/models/mmpose/`. Each SHA256 prefix matches the 8-hex hash in its OpenMMLab file name.
  | file | SHA256 | published COCO val2017 AP (det. boxes) |
  |---|---|---|
  | td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth | 39c3c3816ec3f6cd180e27fb79435d9cbd878acd37317bbbd942a97ddbb36ec8 | 0.772 |
  | td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth | 70d7ab0196e40b61e4508f08e547576dee9ec90f069ec5a8520f6126b8b3b4a1 | 0.773 |
  | td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth | 53609f55473dd4d26aa2e509911fc092a3e4d438e9eb75126be6f905b693c475 | 0.782 |
  | td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth | e32adcd41ab0b0ef0b5bf3d167ddae7cdbd45fcf45e7f6a834815ef04d641f2b | 0.788 |
- **Trap: mmpretrain cannot resize ViTPose's pos_embed for 384x288.** This contradicts the literature note that "mmpretrain resizes pos_embed automatically".
  - mmpretrain 1.2.0 `VisionTransformer._prepare_pos_embed` infers the source grid as `to_2tuple(int(np.sqrt(L)))`. For the 16x12 ViTPose grid (L=192) that gives 13x13, and `resize_pos_embed` then fails its `L == src_h*src_w` assertion.
  - Fix: an offline bicubic resize with the true grid, 16x12 → 24x18 (`mmpose_exps/tools/resize_vit_pos_embed.py`, job 11659247). The checkpoint pos_embed is (1,193,C): one leading cls slot, which mmpretrain strips on load because `with_cls_token=False`. The slot is kept unchanged, so (1,433,C) loads the same way. Derived files in `$R/models/mmpose/derived/`:
    - `..ViTPose-large..._posembed24x18.pth` SHA256 `627f26a2c948b301dc012114145f834b9dddf41ce3aa639f7cb509727eb6ab6f`
    - `..ViTPose-huge..._posembed24x18.pth` SHA256 `f8c4b2df396ec054a7857f497289913176cd00e919fd992266a97e0f72a0e712`
- **Configs** (`mmpose_exps/configs/`):
  - `_base_/syrip_base.py`: runtime, data, evaluator and the shared schedule.
  - Control: `syrip_hrnet-w48_dark-384x288.py`.
  - Treatments: `syrip_vitpose-{l,h}_udp-{256x192,384x288}.py`.
  - Design decisions:
    - **Train mix:** mmpose `CombinedDataset` with `sample_ratio_factor=[5,1]` over real200 / syn1000. Its resampling maps each epoch index into the real set with stride ≈0.2, so every real image appears about 5x per epoch: 2000 samples per epoch, 1:1 (FiDIP's BalancedBatchSampler). Children supply only the pipeline, so one dataset base serves every backbone.
    - **Same for every backbone:** epochs (30), schedule shape (1-epoch linear warm-up, then x0.1 at 170/210 and 200/210 of the budget = epochs 24 and 29), augmentation (mmpose COCO defaults), GT boxes, flip test, Val400 `save_best='coco/AP'` checked every epoch.
    - **Per family:** optimizer (HRNet Adam; ViT AdamW, wd 0.1, layer decay 0.8 L / 0.85 H, grad-clip 1.0, drop-path 0.5 / 0.55), batch (official per-GPU: 32 and 64), codec (DARK vs UDP).
    - **Screening LR = 1e-4 nominal for both families.** This is FiDIP's fine-tuning LR, and it sits between the linear-scaled official LRs (5e-4 x 32/256 = 6.25e-5 for HRNet, 5e-4 x 64/512 = 6.25e-5 for ViT) and FiDIP scaled to batch 32 (1.6e-4). It gets swept on Val400 for the control and the best ViT alike.
    - 384x288 ViT: UDP sigma 3 at 72x96 heatmaps, which keeps the Gaussian the same size relative to the person as sigma 2 at 256x192. This is also mmpose's own 384x288 UDP setting.
    - Backbone `init_cfg` is dropped (no MAE/ImageNet download); `load_from` loads the full COCO model.
- **Data layout seen (no `SPLITS_READY` yet):** `$R/data/splits/{train_infant,val400,test100,validate500}.json` exist. `train_infant`: 1200 images, 1000 `is_synthetic`. `$R/data/syrip/images/validate_infant` links to the syrip_for_train copy (100 images = Test100 only), so **Val400 images (test100..test499.jpg) exist only in `raw/SyRIP/images/validate_infant`** (500). Waiting for REPRO's marker and notebook before fixing `img_val`.
- **Tooling:**
  - `mmpose_exps/tools/{verify_env,eval_ckpt,finalize_run,make_train_subsets,resize_vit_pos_embed}.py`.
  - `scripts/slurm/{build_env,get_ckpts,prep_ckpts,verify_env,train,eval_val400,final_test_eval}_mmpose.sbatch`.
  - `scripts/push_mmpose.sh`: syncs, then writes `$R/repo/GIT_INFO_mmpose.json` (HEAD, HEAD^{tree}, dirty flag) for run.json, per the coordinator's commit-rewrite note.
  - `eval_ckpt.py` and `final_test_eval_mmpose.sbatch` refuse Test100 unless the RUN_ID has a row in `results/PREREGISTRATION.md`. The final script also checks the checkpoint SHA256 against run.json and refuses a second Test100 eval.
- **GPU verify jobs** (after prep 11659247): **11659249** (gpu, H200) and **11659250** (sharing, A100).

### 2026-11-05 20:34 EST — First GPU verify: two failures, fixes
- **Verify jobs 11659249 (H200, d4053, 49 s) and 11659250 (A100-SXM4-80GB, d3204, 31 s)** both COMPLETED, but every config check failed with `ModuleNotFoundError: No module named 'mmdet'`.
  - **Cause:** mmpose 1.3.2 imports mmdet unconditionally in `mmpose/models/heads/hybrid_heads/rtmo_head.py:9` (`from mmdet.utils import ConfigType, reduce_mean`), so `import mmpose.models` needs it even with GT boxes. The literature note "mmdet isn't needed" holds for inference APIs, not for model building.
  - **Fix:** `mmdet==3.3.0` (accepts mmcv>=2.0.0rc4,<2.2.0) added to `build_env_mmpose.sbatch`. Re-run job **11659316**. Verify re-queued: **11659317** (H200) and **11659318** (A100).
  - Benign pip warning seen in the rebuild: `openxlab 0.1.3 requires setuptools~=60.2.0, but you have setuptools 69.5.1`. openxlab comes in with openmim and is unused here.
- **H200: mmcv 2.1.0 CUDA ops have no sm_90 kernels.** `mmcv.ops.nms` → `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
  - torch itself is fine: arch list `sm_50 … sm_90`. On the A100 the same ops pass (`nms_keep [0, 2]`).
  - **Decision:** the top-down heatmap path (HRNet / mmpretrain ViT, HeatmapHead, KeypointMSELoss, DARK/UDP codecs in numpy/cv2, mmcv CPU transforms) calls **no mmcv CUDA op**. A call on the H200 would raise loudly rather than fail silently, so a clean H200 train + eval smoke run is sufficient evidence that H200 results are valid.
  - **Why H200 at all:** A100 capacity is thin. `sinfo` at 20:30: all 11 `gpu` A100s allocated, 6 of the `sharing` A100 nodes `down*`, 167 jobs pending on `gpu`. H200: 32 on `gpu` + 8 on `sharing`.
  - Plan: train on H200 and record the GPU in every run.json. If anything mmcv-CUDA shows up, fall back to A100 or build mmcv from source with `TORCH_CUDA_ARCH_LIST="8.0;9.0"`.
- Added `scripts/slurm/zeroshot_mmpose.sbatch` (Val400 eval of the raw COCO checkpoints) and `scripts/slurm/prep_data_mmpose.sbatch` + `mmpose_exps/tools/check_data.py` (real200/syn1000 subsets + image-existence check).

### 2026-11-05 20:44 EST — Verify results (A100, L40S, L40S+TF32); batch 32 for every backbone; TF32 on
- **mmdet rebuild (job 11659316, COMPLETED, 2m51s):** `mmdet==3.3.0` installed; numpy 1.26.4 and opencv 4.11.0.86 pins held.
- **Verify jobs:** A100-SXM4-80GB sm_80 (11659318, d1028); L40S sm_89 FP32 (11659395, d4047); L40S with `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (11659409, d4042). H200 re-verify 11659317 is still queued (Priority).
  - mmcv CUDA ops pass on A100 and L40S (`nms_keep [0,2]`, roi_align OK); they fail on H200 (above).
  - Checkpoint loading: **no missing keys for any config**. The only unexpected key is `backbone.cls_token` for ViTPose, which is expected with `with_cls_token=False` and is mmpose's own behaviour.
  - Train step (fwd+bwd+AdamW) in s/step, with peak GB:
    | config | batch | A100 FP32 | L40S FP32 | L40S TF32 |
    |---|---|---|---|---|
    | HRNet-W48 DARK 384x288 | 32 | 0.28 (14.5) | 0.31 (14.4) | 0.31 (14.4) |
    | ViTPose-L 256x192 | 64 | 1.37 (22.7) | 0.80 (22.7) | 0.49 (22.7) |
    | ViTPose-L 384x288 | 64 | 3.14 (47.6) | OOM | OOM |
    | ViTPose-H 256x192 | 64 | 2.82 (40.8) | 1.60 (40.8) | 0.97 (40.8) |
    | ViTPose-H 384x288 | 64 | OOM | OOM | OOM |
    | ViTPose-H 384x288 | 32 | 3.28 (44.6) | OOM | OOM |
- **Decision: batch 32 for every backbone.** Officially it is 64 per GPU for ViTPose and 32 for HRNet.
  - It gives identical iterations: 63 per epoch, 1890 over 30 epochs, close to FiDIP's 2000.
  - Everything except ViT-H-384 then fits the 44-48 GB cards that are free on `sharing` (L40S/L40/A6000, 1 h limit). The `gpu` partition's A100/H200 are congested: the verify estimate was a 40-80 min wait.
  - ViT-H-384 at batch 32 needs an 80 GB+ card (A100/H200 on `gpu`).
- **Decision: TF32 matmuls on** (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`) for every IMPROVE train and eval job, recorded in run.json.
  - PyTorch 2.1 already runs cuDNN convs in TF32 by default, so HRNet is unaffected (0.31 s/step either way) and ViT matmuls now get the same treatment (1.6x faster on L40S, more on A100).
  - FP32 accumulation is kept. Inference is FP32/TF32 (no AMP).
- **Data still pending:** REPRO's `make_splits.py` (jobs 11659256/299/351/400) fails `assert ov['val400_x_test100_by_id'] == ov['val400_x_test100_by_file_name'] == ov['val400_x_test100_by_md5'] == 0`. Some Val400/Test100 overlap, likely md5 duplicates (REPRO has a `dups.py`). If so, `val400.json` may still change.
  - I will not start screening until `SPLITS_READY`. Two-epoch **smoke** runs (pipeline check only, no selection) go ahead on the current files.

### 2026-11-05 20:49 EST — Reply to orchestrator: $R/data/raw was not touched by IMPROVE; $R/data is now read-only for me
- **Q: did an IMPROVE job or command move or modify files under `$R/data/raw` (e.g. the ~20:08 flatten of `raw/SyRIP/SyRIP`)? A: No.**
  - Every IMPROVE command that touched `$R/data` was read-only: `ls`, `readlink`, `os.path.exists` checks and JSON reads of `$R/data/splits/*.json`.
  - At 20:08 my only jobs were the env build (11659031, writes `$R/envs/mmpose` and `$R/mmpose`) and the checkpoint download (11659053, writes `$R/models/mmpose`).
- **One write I did make into `$R/data` (not raw), now undone.** Job **11659492** (`prep_data_mmpose`, 20:46, 29 s) wrote my derived train subsets to `$R/data/improve_splits/{train_real200,train_syn1000,MANIFEST}.json`.
  - I deleted that directory at about 20:50; `$R/data` again holds only `downloads raw splits syrip`.
  - The subsets now go to **`$R/mmpose_data/splits/`**. The base config uses absolute `ann_file` paths for them; every other path still reads `$R/data/splits/*.json` and the REPRO image roots `$R/data/syrip/images/<split>/` (read-only).
  - For the record, the check in 11659492 against the then-current splits: real200 = 200 images (0 syn), syn1000 = 1000 (all syn), val400 = 400, test100 = 100; **0 missing images** in `syrip/images/{train_infant,val400,test100}`.
- **Cancelled smoke jobs:** 11659495 (R190) and 11659496 (R191) had been running 46 s and 62 s. I cancelled them because their configs pointed at the old subset path, and deleted their partial run dirs. They will be relaunched under the same RUN_IDs.
