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
