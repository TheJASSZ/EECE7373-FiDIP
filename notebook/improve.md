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

### 2026-11-05 20:52 EST — Slurm QOS limits (per user; every stream shares purohitham.v)
- From `sacctmgr show qos`:
  | partition | running jobs | submitted jobs | GPUs | walltime |
  |---|---|---|---|---|
  | `sharing` | 2 | 4 | 2 | 1 h |
  | `gpu` | 4 | 8 | 4 | 8 h |
- The R191 smoke run (11659558) is pending with `QOSMaxJobsPerUserLimit` while R190 (11659557, L40S d4047) and the H100 verify (11659526, d4041) run.
- Consequence: at most about 6 concurrent GPU jobs for the whole project, so I'll keep about 2-4 IMPROVE jobs in flight. Heads-up sent to the orchestrator.

### 2026-11-05 20:54 EST — GPU budget policy D14 adopted; scancel blocked; SKILL-ready GPU notes
- **D14 (orchestrator) IMPROVE share:**
  - `gpu`: ≤2 running, ≤4 submitted. `sharing`: ≤1 running, ≤2 submitted.
  - Job names prefixed `im_`. Several fine-tunes are packed into one GPU job where memory allows.
  - Target A100/L40S/A6000 explicitly. REPRO's R001/R002 evals and R010-R015 have priority.
- **Transient overshoot, reported honestly.** At adoption IMPROVE had 2 running on `sharing` (R190 smoke 11659557, H100 verify 11659526) and one pending (R191 smoke 11659558), plus one pending H200 verify on `gpu` (11659317).
  - I tried to `scancel` the two pending, never-started jobs (11659317, 11659558; both <30 min old). **The Claude Code permission classifier denied it ("Interfere With Workloads").** I did not work around that.
  - Both are short (~1-4 min each) and useful: R191 is the ViT-L-384 pipeline smoke, and 11659317 documents sm_90. They will run as-is. No new `sharing` submissions until they drain.
- **SKILL-relevant GPU facts (for SKILL.md):**
  - mmcv 2.1.0 (cu121/torch2.1 wheel) has **no sm_90 kernels**: `mmcv.ops.nms` on H100/H200 → `RuntimeError: CUDA error: no kernel image is available for execution on the device`. torch 2.1.2+cu121 itself supports sm_90 (`get_arch_list` ends at `sm_90`).
  - The ops are verified OK on **A100 (sm_80)** and **L40S (sm_89)**; A6000 (sm_86) gets the same wheel kernels. So for the MMPose env use `--gres=gpu:a100:1` (gpu partition) or `--gres=gpu:l40s:1` / `gpu:a6000:1` (sharing); avoid `h100`/`h200`.
  - Measured memory (fp32/TF32, batch 32): HRNet-W48 384x288 14.4 GB; ViTPose-L 384x288 ~24 GB; ViTPose-H 384x288 44.6 GB. The last does not fit a 48 GB L40S (44.4 GB usable), so it needs an A100-80GB.
  - Per-user QOS: `sharing` 2 running / 4 submitted / 2 GPUs / 1 h; `gpu` 4 / 8 / 4 GPUs / 8 h. Check with `sacctmgr show qos`.
  - mmpose 1.3.2 needs `mmdet` installed even for GT-box top-down (unconditional import in `mmpose/models/heads/hybrid_heads/rtmo_head.py:9`). mmdet 3.3.0 works with mmcv 2.1.0.
  - ViTPose at 384x288: mmpretrain 1.2.0 cannot auto-resize the non-square 16x12 `pos_embed` (`int(sqrt(192))=13` → assertion). The procedure lives in committed `mmpose_exps/tools/resize_vit_pos_embed.py`, run by `scripts/slurm/prep_ckpts_mmpose.sbatch` (job 11659247): bicubic, `align_corners=False`, 16x12 → 24x18, cls slot kept.
    - L: SHA256 `627f26a2c948b301dc012114145f834b9dddf41ce3aa639f7cb509727eb6ab6f`
    - H: SHA256 `f8c4b2df396ec054a7857f497289913176cd00e919fd992266a97e0f72a0e712`
    - A sidecar `<file>.json` next to each derived checkpoint records source SHA, grids, mode and destination SHA.

### 2026-11-05 20:59 EST — Smoke tests passed (2 epochs each, `sharing` L40S): full pipeline validated
- **R190_mmp-hrnetw48_smoke_s1**: job 11659557, d4047 L40S, 4m15s.
  - Train 0.37-0.40 s/iter at batch 32 (data 0.03-0.06 s), 14.8 GB; Val400 pass about 4 s.
  - MMPose Val400 AP 0.8917 (ep 1), 0.8985 (ep 2); selected `best_coco_AP_epoch_2.pth` (SHA256 e578ff9a…).
  - **eval_coco.py Val400 (paper = fixed):** AP 89.8, AP50 98.9, AP75 95.8, AR 91.2, AP@OKS .80/.85/.90/.95 = 92.9/87.4/77.0/53.1 (AP_80_95 77.6). Mean OKS 0.933. Identical to MMPose's printed 0.8985 (Val400 has no id-0 GT).
- **R191_mmp-vitpose-l384_smoke_s1**: job 11659558, d4042 L40S, 4m12s.
  - Train 0.67 s/iter at batch 32, 25.4 GB; the resized-pos_embed checkpoint loads and trains (Val400 AP 0.899 after one epoch).
  - The logged `lr` is the layer-0 group (base_lr x 0.8^25), as intended.
  - MMPose Val400 AP 0.8995 (ep 1), 0.9119 (ep 2).
  - **eval_coco.py Val400:** AP 91.2, AP50 98.9, AP75 96.7, AR 92.8, AP@.80/.85/.90/.95 = 93.4/91.2/81.4/58.1 (AP_80_95 81.0). Mean OKS 0.940.
- **What the smokes verify:**
  - CombinedDataset 1:1 mix, GT-box Val400 eval with flip, `save_best`, resolved config dump.
  - `eval_ckpt.py` pred dump (COCO results with `image_id`, `keypoints`, `score`, `bbox`), then `eval_coco.py`.
  - `finalize_run.py`: run.json with job/node/GPU/seed/TF32/git head+tree+dirty/init+selected ckpt SHA256; registry row.
- Smoke rows are appended to `results/registry.csv` marked "pipeline check only". They are not candidates, and their Val400 predates SPLITS_READY.
- **H100 verify (11659526, d4041, sm_90, TF32):** mmcv ops fail (`no kernel image`, as on H200); all model train steps OK.
  - HRNet b32 0.175 s; ViT-L-256 b64 0.195 s; ViT-L-384 b64 0.49 s (47.7 GB); ViT-H-256 b64 0.37 s; ViT-H-384 b32 0.49 s (44.6 GB); ViT-H-384 b64 OOM on 80 GB.
  - Not used for runs (D14).
- **Pipeline changes since:**
  - Per-run body factored into `scripts/run_one_mmpose.sh`, called by `train_mmpose.sbatch` (one run) and the new `train_pack_mmpose.sbatch` (several runs on one GPU, staggered starts, `nvidia-smi` per-process memory log every 60 s at `$R/logs/pack/<job>_gpu.csv`, `pack_info.json` per run).
  - All job names carry the `im_` prefix.
  - Finalized runs drop the resumable `epoch_*.pth` (optimizer state: 765 MB HRNet, ~3.7 GB ViT-L) and keep `best_*.pth`.
  - run.json gains `throughput` (median s/iter, data s/iter, max memory).

### 2026-11-05 21:05 EST — SPLITS_READY (21:02); Val400 = 397 images; pack smoke shows packing works
- **REPRO's final splits** (`$R/data/splits/VERIFY.txt`):
  - `train_infant` 1200 (200 real / 1000 syn), sha256 `3bdcce0b…`. **Unchanged**, so my derived real200/syn1000 subsets (`$R/mmpose_data/splits`, built 20:58 from this exact file) remain valid.
  - `test100` 100, sha256 `8b299a87…`, unchanged.
  - **`val400` now 397 images**, sha256 `8ed74bd5…`. REPRO dropped 207 (near-duplicate of a Test100 image) and 217 and 222 (byte-identical to Test100 ids 56 and 7). Val400×Test100 overlap by id, file name and md5 is now 0. The name "Val400" is kept and the file is `val400.json`.
  - Image rule unchanged: `$R/data/syrip/images/<split>/<file_name>`. `test489.jpg` is PNG bytes; cv2 decodes it fine.
  - Official leak noted by REPRO: 4 Test100 images are byte-identical to official train images (test,train) = (20,73),(42,315),(46,272),(50,340), plus near-dup (7,47). They are kept for comparability with the paper, which trained on the same split.
- **Pack smoke (job 11659697, L40S d4047, 2m46s, FAILED by design of timing):** R192 HRNet + R193 ViT-L-256, 1 epoch each.
  - Both trained concurrently: GPU util 93%, per-process peak 16.8 GB (HRNet) and 14.8 GB (ViT-L-256). Packed s/iter: HRNet 0.53 (vs 0.37 alone), ViT-L-256 0.46.
  - Both then failed at the Val400 eval with `FileNotFoundError: …/images/val400/test207.jpg`: REPRO was rebuilding the Val400 tree (207 dropped) at that moment. This confirms the decision to wait for SPLITS_READY. R192/R193 are discarded (no run.json; not registered).

### 2026-11-05 21:11 EST — Coordinator notes applied (final splits, PYTHONNOUSERSITE); screening launched
- **Derived subsets rebuilt from the final splits:** `prep_data_mmpose` job **11659874** (short) re-runs `make_train_subsets.py $R/data/splits/train_infant.json $R/mmpose_data/splits` and `check_data.py`.
  - The source sha256 is the same `3bdcce0b…` as at 20:58, but the coordinator flagged it as stale, so it is rebuilt and the new SHA256s get logged. Only `$R/data/splits/train_infant.json` (SyRIP.zip 200R_1000S) is used; never syrip_for_train's train JSON.
- **PYTHONNOUSERSITE:**
  - `export PYTHONNOUSERSITE=1` added to every `*_mmpose.sbatch` that runs python and recorded in run.json.
  - Job **11659867** (`usersite_check_mmpose.sbatch`) lists `~/.local/lib`, prints every key package's version and `__file__` with and without the flag, then runs `conda env config vars set -p $R/envs/mmpose PYTHONNOUSERSITE=1`.
- **Launched (D14 budget: gpu 2 running / ≤4 submitted, sharing 1 running / 2 submitted), all after 11659874 and 11659867:**
  | job | partition / GPU | contents |
  |---|---|---|
  | **11659880** `im_zeroshot_mmpose` | sharing L40S | Val400 zero-shot of the 5 COCO models: R181 HRNet-W48, R182 ViT-L-256, R183 ViT-L-384 (resized pos_embed, never trained at 384), R184 ViT-H-256, R185 ViT-H-384 |
  | **11659881** `im_pack_screenA` | gpu A100-80GB | R104_mmp-vitpose-h384_base_s1 (~45 GB) + R103_mmp-vitpose-l384_base_s1 (~24 GB) |
  | **11659885** `im_pack_screenB` | gpu A100-80GB | R105_mmp-vitpose-h256_base_s1 + R106_mmp-hrnetw48_lr2e-4_s1 + R107_mmp-hrnetw48_lr5e-5_s1 (control LR sweep, independent of which ViT wins) |
  | **11659888** `im_pack_screenC` | sharing L40S, after zero-shot | R101_mmp-hrnetw48_base_s1 + R102_mmp-vitpose-l256_base_s1 |
- Base recipe for all R10x: 30 epochs x 63 iters (batch 32), 1-epoch warm-up + step x0.1 at epochs 24/29, LR 1e-4 nominal, mmpose COCO augmentation, Val400 `save_best` every epoch, seed 1, TF32.

### 2026-11-06 01:05 EST — Resumed after usage pause: results so far, OOM failures, reruns
- **Usersite check (11659867):** `~/.local/lib/python3.10/site-packages` does not exist, so nothing shadows the env. All packages (numpy 1.26.4 … mmdet 3.3.0) load from `$R/envs/mmpose`. chumpy's import fails on numpy 1.26 (expected; unused). `conda env config vars set PYTHONNOUSERSITE=1` applied.
- **Subsets rebuilt from the final splits (11659874):** real200 sha256 `081b5664c938397f8a2086991da1991ee31d855031ac216da566d8766207c879`, syn1000 `ee0c5b3bfbe080a6300197c04b460e98662adb0f4799ae1231dcbbe4cd209210`. Identical to the 20:58 build, as expected from the unchanged source.
- **Documented failure: OOM on a 40 GB A100.** `--gres=gpu:a100:1` on `gpu` can land on **d1026 = A100-SXM4-40GB**; only d1028/d1029 have 80 GB (node feature `a100@80g`).
  - `im_pack_screenA` **11659881** (R104 ViT-H-384 + R103 ViT-L-384) failed with OOM after 5 min.
  - `im_pack_screenB` **11659885** (20 min): R105 ViT-H-256 `torch.cuda.OutOfMemoryError` and R107 `RuntimeError: CUDA error: out of memory`; R106 completed.
  - Failed dirs moved to `$R/runs/_failed/<RUN_ID>_job<id>` (logs kept, .pth removed).
  - **Fix:** `--constraint=a100@80g` for A100 packs. (SKILL note: `gpu:a100` ≠ 80 GB.)
- **Val400 so far** (eval_coco `metrics`, 397 images, paper = fixed protocol):
  | run | AP | AP50 | AP75 | AR | AP.80-.95 | sel. ep | job / GPU |
  |---|---|---|---|---|---|---|---|
  | R181 HRNet-W48 zero-shot | 86.46 | 98.92 | 92.75 | 88.31 | 71.64 | - | 11659880 L40S |
  | R182 ViT-L-256 zero-shot | 88.21 | 98.86 | 94.77 | 90.40 | 74.79 | - | 11659880 |
  | R183 ViT-L-384 zero-shot (resized pos_embed) | 86.30 | 97.81 | 93.66 | 89.45 | 70.87 | - | 11659880 |
  | R184 ViT-H-256 zero-shot | 88.81 | 98.88 | 94.63 | 90.91 | 75.84 | - | 11659880 |
  | R185 ViT-H-384 zero-shot (resized pos_embed) | 87.44 | 98.85 | 94.53 | 90.38 | 73.03 | - | 11659880 |
  | **R101 HRNet-W48 base (control)** | **91.52** | 98.85 | 95.72 | 92.75 | 81.57 | 29 | 11659888 L40S |
  | R106 HRNet-W48 lr 2e-4 | 91.43 | 98.83 | 95.79 | 92.59 | 81.85 | 28 | 11659885 A100-40 |
  | **R102 ViT-L-256 base** | **92.81** | 98.84 | 96.73 | 94.13 | 84.82 | 16 | 11659888 L40S |
- The Val400 curves plateau by ~epoch 10: HRNet 91.3-91.5, ViT-L-256 92.4-92.6. The 30-epoch budget is ample, and Val400 `save_best` picks within the plateau. The released FiDIP (R001, REPRO) scores 90.4 on Val400 for reference.
- **Reruns (fresh dirs):**
  | job | partition / GPU | contents |
  |---|---|---|
  | `im_pack_screenA2` **11664861** | gpu A100-80 | R104 + R105 (~68 GB) |
  | `im_pack_screenB2` **11664862** | gpu A100-80 | R103 + R107 (~40 GB) |
  | `im_pack_lrL256` **11664863** | sharing L40S | ViT LR sweep on the current best, matching the control's sweep: R108 ViT-L-256 lr 2e-4, R109 lr 5e-5 |
