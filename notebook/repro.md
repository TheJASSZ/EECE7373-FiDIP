# notebook/repro.md — live log for the repro stream

Newest entries at the bottom. Timestamps in EST.

### 2026-11-05 20:05 EST — start; read CLAUDE.md, SKILL.md, AGENT.md, upstream syrip.py
- Upstream `lib/dataset/syrip.py` has a trap the audit didn't list. `_get_ann_file_keypoint()` and `evaluate()` test `'test' in image_set`. A split literally named `test100` would load `image_info_test100.json` and **skip evaluation** (returns `{'Null': 0}`, AP 0). Patch P10 must handle it (see the patch entry).
- `unzip -l` on the zips (central directory only, cheap):
  - SyRIP.zip: `SyRIP/{README.md, annotations/{200R,200R_1000S,1000S,validate100,validate500}, images/{train_infant (1500 files), validate_infant (500 files)}}`. The 500-image folder is spelled `validate_infant` in the zip; the README's `validate_infnat` is a typo.
  - syrip_for_train.zip: `syrip/annotations/person_keypoints_{train_infant,train_pre_infant,validate_infant}.json`, `syrip/images/{train_infant (1500), train_pre_infant (4057), validate_infant (100)}`.
  - **train_pre_infant IS shipped** in syrip_for_train.zip, although the upstream README says "available on request". So PRE_EPOCH can use the real set; no substitute is needed.

### 2026-11-05 20:07–20:15 EST — unzip in a Slurm CPU job
- Wrote `scripts/get_data.sh` (stages download/verify/unzip/layout/splits) and `scripts/slurm/get_data.sbatch` (partition `short`, 4 CPU, 8 GB).
- Job 11659034: cancelled while still PENDING (seconds old). I changed the extraction path first.
- Job **11659050** (c0594): `verify` OK (all 3 SHA256 match `SHA256SUMS`). Unzip took 16 s + 46 s. Then my count check failed: `SyRIP.zip: 2006 files in zip, 2629 files on disk` → `ERROR: count mismatch`.
  - Cause: at 20:08:00, before my job started, someone flattened the killed login-node partial extract (`raw/SyRIP/SyRIP/*` → `raw/SyRIP/`; 623 stale files). The analysis stream already reads `raw/SyRIP/annotations/*`. My job had extracted a complete copy into `raw/SyRIP/SyRIP/`.
  - Fix (metadata-only, login node): checked that `raw/SyRIP/SyRIP` has exactly 2006 files, then `rm -rf raw/SyRIP/{annotations,images}` (stale partial), moved `raw/SyRIP/SyRIP/{README.md,annotations,images}` up one level, and removed the empty dir. Result: `raw/SyRIP` has 2006 files = zip; `raw/syrip_for_train` has 5660 = zip. Touched `raw/UNZIP_DONE`.
  - `get_data.sh` now extracts SyRIP.zip with `-d $RAW` (single-nested), so a from-scratch run reproduces exactly this layout.
- **Final raw layout:** `$R/data/raw/SyRIP/{README.md,annotations,images}` and `$R/data/raw/syrip_for_train/syrip/{annotations,images}`.
- Infra note: `python3` reading ~1000 JPEG headers on the login node dropped the ssh session with exit 255, twice; a plain `echo` over the same ssh worked. Moved all python data checks to `srun -p short` (job 11659171 and later).

### 2026-11-05 20:15–20:25 EST — annotation audit: two train JSONs, one defective
- **validate100 ⊂ validate500:** validate100 image ids are exactly {0..99}; validate500 ids are {0..499}. File names `test<id>.jpg` and width/height are identical, and validate100's image order equals validate500's first 100. So Val400 = validate500 ids {100..499}.
  - The `syrip_for_train` copy of validate100 is identical to SyRIP.zip's in ids, ann ids, keypoints, bbox and area. It adds an `images[].posture` field (supine 43, prone 41, sitting 11, standing 5), which may be useful for failure analysis.
- **Areas** (matches D8): validate100 area == w*h on 100/100; validate500 has the constant 2443298.48; train 200R_1000S has the constant 14232.77.
- **Two different train annotation files exist** for the same 1200 ids and file names:
  - (a) SyRIP.zip `annotations/200R_1000S/person_keypoints_train_infant.json` (official).
  - (b) syrip_for_train.zip `annotations/person_keypoints_train_infant.json`.
  - Real 200: identical. Synthetic 1000: **all differ** (per-image max keypoint diff p50 = 191 px, max 1165 px; every bbox differs). (b) also zeroes eyes/ears (v=0) on ~95% of synthetic images.
  - Sampled md5 (7 train + 3 val images): the image bytes are identical in both zips.
  - Checked against the real JPEG headers (`srun` job 11659171): (a) has width/height correct for 1000/1000 synthetic images and 987 boxes inside the image. (b) has the wrong size on 46 images (it says 640×480 for 1920×1080 renders) and only 858 boxes inside the image.
  - **Cause:** every synthetic entry in (b) has `original_file_name` like `10/syn_00360.png` and `image_source: "MINI_dataset"`. **These are MINI-RGBD labels attached to SyRIP synthetic file names.** MINI-RGBD has no face keypoints, which explains the zeroed eyes and ears.
  - **Decision:** train on (a) SyRIP.zip `200R_1000S` with SyRIP.zip images, everywhere (upstream retrain and framework-neutral `train_infant.json`). (b) is documented as defective and NOT used. This is also the "official SyRIP train" of the protocol. A visual overlay check follows once cv2 is installed.
- Coordinator decisions received and applied in `make_splits.py`:
  - D8: area := w*h for val400, validate500 and neutral train. test100 stays as shipped.
  - D10: ids exactly as shipped, asserted against the source JSONs; `test100` first ann id 0 kept.
  - No co-author trailer in commits.
  - run.json records HEAD + tree SHA + a dirty flag.

### 2026-11-05 20:26–21:03 EST — splits: three failed attempts, then all assertions pass
- **Job 11659256** failed on my over-literal D10 assertion. "First annotation has id 0" is wrong: validate100's annotation list starts with image 27. Changed it to "min annotation id == 0".
- **Job 11659299** failed: `val400 ... wh_mismatch_examples: [(489, 'test489.jpg', (61278, 35522), (680, 679))]`.
  - `file` shows test489.jpg is really a **PNG** (680×679 RGBA) named .jpg. cv2/PIL decode by content, so the loaders are fine.
  - Fix: the header reader now handles PNG IHDR, and non-JPEG files are reported.
- **Job 11659351** failed: in the upstream 100-image folder (syrip_for_train.zip `images/validate_infant`), `test11.jpg` is 2048×2048 while the GT says 1536×2048, and it is md5-identical to `test24.jpg`.
  - Checked by hand: `test11 SyRIP=0161fbedd7ce for_train=c642c6573896`, `test24 = c642c6573896` in both zips.
  - **So upstream's documented layout (`TEST_SET validate_infant`) scores image 11 against the wrong picture.** SyRIP.zip's copy is correct.
  - Made the upstream-folder check informational. R001/R002 report both `test100` (correct images) and `validate_infant` (as shipped).
- **Job 11659400** failed on `val400_x_test100_by_md5 == 0`. An md5 scan of all 500 validation images plus train (srun job) found:
  - Val/Test exact duplicates: val217 = test56 and val222 = test7, with identical keypoints and slightly different boxes. Val345 = val349 inside Val400.
  - **Official train/Test100 leak:** test 20/42/46/50 are byte-identical to annotated train 73/315/272/340.
- **Near-duplicate scan** (`scripts/near_dups.py`; jobs 11659552/63/92/656): flip-aware 64-bit dHash plus a GT pose-OKS in image-normalized coordinates, with contact sheets checked by eye.
  - At dHash 4–9 most pairs are *different frames of the same video*: same infant and background, different pose (e.g. test51/val239, test72/val355, test81/val180 with pose-OKS 0.60). These are correlated but not duplicates.
  - Rule fixed before looking at OKS values: exclude from Val400 any image that is exact-md5 to a Test100 image, or has dHash ≤ 10 and pose-OKS ≥ 0.85 with one. This gives **{207 (test47, d=1, OKS 0.87), 217, 222}**. The closest kept pair is test51/val239 at OKS 0.849.
  - Test100 ↔ train near-exact copies: (7↔train47, OKS 0.96) plus the 4 exact ones.
- **Job 11659717** (21:02): ALL ASSERTIONS PASSED. Touched `$R/data/SPLITS_READY` at 21:02:55 and messaged the orchestrator. Also found: 199 of the 300 *unannotated* real images in SyRIP.zip `images/train_infant/` are byte-identical to validate500 images, so train images must come only from the JSON.
- Orchestrator decisions: D15 (Val400 = 397, name kept), D16 (Test100 as shipped is primary; Test100-clean = minus {7, 20, 42, 46, 50} as a sensitivity subset), D17 (train on 200R_1000S), D7 revised (PRE_EPOCH 10 with the real train_pre set).

**Final data paths (for all streams):**

| What | Path |
|---|---|
| Framework-neutral GT | `$R/data/splits/{train_infant,test100,val400,validate500}.json` (sha256 3bdcce0b…, 8b299a87…, 8ed74bd5…, d129e2f5…) |
| Image rule | `$R/data/syrip/images/<split>/<file_name>` |
| Upstream DATASET.ROOT | `$R/data/syrip` (annotations/ → symlinks; `TEST_SET` ∈ {test100, val400, validate500, validate_infant}) |
| Report | `$R/data/splits/VERIFY.{txt,json}` (copy in repo `splits/VERIFY.txt`) |
| ID lists | repo `splits/{val400,test100}_image_ids.txt`, `splits/val400_exclude_near_dups.txt` |

| Split | images | anns | multi-person | synthetic | area (D8) |
|---|---|---|---|---|---|
| train_infant | 1200 | 1200 | 0 | 1000 (200 real) | w*h (source had constant 14232.8) |
| test100 | 100 | 100 | 0 | 0 | as shipped = w*h |
| val400 | **397** | 397 | 0 | 0 | w*h (source had constant 2443298.5) |
| validate500 | 500 | 500 | 0 | 0 | w*h |
| train_pre_infant (syrip_for_train) | 4057 | 4057 | 0 | 2028 (2029 real) | as shipped |

### 2026-11-05 20:35–21:00 EST — data integrity of $R/data/raw (orchestrator request)
- The 20:08 flattening of `raw/SyRIP/SyRIP` was **not** my job: job 11659034 never ran, and job 11659050 started at 20:10:26. My only intervention was the metadata-only fix at ~20:15 (see above).
- `scripts/verify_raw.py` (job 11659573, 29 s) checked every zip member against the disk:
  - `SyRIP.zip`: 2006 members, missing 0, size mismatch 0, CRC32 mismatch 0, extra on disk 0; md5(decompressed member) == md5(disk) for 60/60 sampled.
  - `syrip_for_train.zip`: 5660 members, all 0; 60/60 md5.
  - → `RAW TREES VERIFIED`. It is the `verifyraw` stage in `get_data.sh`.

### 2026-11-05 20:31–20:55 EST — environment `$R/envs/fidip`
- Job 11659313 (12.5 min): `conda create -p $R/envs/fidip -c conda-forge --override-channels python=3.12.4`, then pip-only installs:
  - torch 2.3.1 / torchvision 0.18.1 from the cu121 index;
  - numpy 1.26.4, opencv-python-headless 4.10.0.84, pycocotools 2.0.8, json-tricks 3.17.3, yacs 0.1.8, dsntnn 0.5.3, tensorboard 2.17, tensorboardX 2.6.2.2, cython 3.0.10, scipy 1.14.0, matplotlib 3.9.1, gdown 5.2.0, streamlit 1.39.0.
  - Deviation from `fidip_env.yml` (grpcio 1.48.2 / mkl-* have no cp312 wheels).
  - Result: torch 2.3.1+cu121, CUDA 12.1, cuDNN 8902. The compute node reaches drive.google.com (HTTP 302).
- **Problem:** pip warned `transformers 4.57.3 requires safetensors…`, but I installed neither. `~/.local/lib/python3.12/site-packages` (user's other projects: transformers, datasets, peft, accelerate, absl…) leaks into every py3.12 env. The freeze listed them, and `absl-py` (a tensorboard dependency) was being satisfied from `~/.local` only.
  - **Fix:** `PYTHONNOUSERSITE=1` in the build and all job scripts, plus `conda env config vars set PYTHONNOUSERSITE=1`.
  - Rebuild job 11659491 (54 s) installed `absl-py` into the env and re-exported a clean freeze (90 lines, no foreign packages) → `envs/fidip-pip-freeze.txt`, `envs/fidip.yml`.

### 2026-11-05 20:35–20:50 EST — upstream patches (one commit each, `[EECE7373 patch Pn]` markers in the code)
| Patch | Files | Why |
|---|---|---|
| P1 | lib/nms/nms.py | compiled nms imports optional (only oks_nms is used) → no `make` |
| P3 | tools/test_adaptive_model.py, tools/train_adaptive_model_hrnet.py | `torch.load(map_location='cpu', weights_only=False)`; strict test load that strips a `module.` prefix if present (upstream strict=False could silently load nothing) |
| P4 | tools/train_adaptive_model_hrnet.py | build the train_pre loader only if PRE_EPOCH > 0 |
| P5 | lib/models/{pose_hrnet,adaptive_pose_hrnet}.py | `is '*'` → `== '*'` |
| P6 | lib/utils/utils.py | `mkdir(parents=True, exist_ok=True)` for per-run OUTPUT_DIRs |
| P7 | tools/train_adaptive_model_hrnet.py | loss/acc .npy into final_output_dir; `add_graph` dropped |
| P8 | lib/core/function.py, tools/train_adaptive_model_hrnet.py | `epochs.csv` per epoch: train MSE (pose loss without −λ·BCE), loss_p, domain loss/acc, val loss/acc, AP/AR, lr, is_best, wall time |
| P10 | lib/dataset/syrip.py | any split name (val400/test100/...) via json file_name under images/<split>/; evaluate labelled 'test*' splits |
| P11 | lib/config/default.py, tools/train_adaptive_model_hrnet.py | `SEED` (python/numpy/torch/cuda; -1 = unseeded as upstream) |

Recipe quirks left as-is for faithfulness (to discuss in the report):
- `model_p.train()` during the PRE_EPOCH classifier stage, so pose-net BN statistics adapt to the adult COCO+SURREAL data. Only `optimizer_d` steps.
- The Step II backward also accumulates −λ·∂BCE into `model_d`'s grads, which are only zeroed after the next Step I update.
- `lr_scheduler.step()` at epoch start (LR_STEP 40 > 20 epochs, so it has no effect).
- In-place flip of `db_rec['center']` (inherited HRNet bug; matters for the 5× oversampled real images within an epoch).
- `idx <= 108` freezes the stem, layer1, transition1, stage2, transition2 and `stage3.0.branches.0.0.conv1`.
- Unshuffled BalancedBatchSampler: real indices are oversampled 200 → 1000 with python `random`, once.

### 2026-11-05 20:56–21:00 EST — checkpoints (job 11659612, gdown, Drive not blocked)
| File | SHA256 | Drive id / path |
|---|---|---|
| `$R/models/hrnet_fidip.pth` (255 MB) | bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07 | 19tBMoVS8wTza7VqVfPc6KBQ_POPAEIqb / FiDIP_models/hrnet_fidip.pth |
| `$R/models/mobile_fidip.pth` (16.7 MB) | bd0b7548b1f1158f0a8a97d022305547343c835e7d4027727a978d2d7a4cbca1 | 1aHaBwO3XBYtFXI7cDMRIzMWRffIJzrQU / FiDIP_models/mobile_fidip.pth |
| `$R/models/coco/w48_384x288.pth` (255 MB) | c730334eef7c47f051f694949ad2f213526723e46a7781178deb0fc90a93635c | 1TKvk0R_qrWV3UeSTawUYM5Burp-lOmss / TGA_models/coco/w48_384×288.pth (Unicode × on Drive; ASCII name locally) |

- Not fetched: FiDIP_models/coco/posemobile.pth, and 10 other TGA models (listing in `$R/models/drive_listing.json`).
- Strict-load check (`scripts/inspect_ckpt.py`, job 11659693, CPU): all three are plain OrderedDicts with no `module.` prefix and **0 missing / 0 unexpected / 0 shape mismatches**.
  - hrnet_fidip and coco w48: 1754 tensors, HRNet 63.60M params.
  - mobile_fidip: 332 tensors, 4.10M params.
  - So upstream's strict=False was not hiding anything for these files.

### 2026-11-05 21:07–21:10 EST — R001/R002 released-checkpoint eval (job 11659813, sharing, L40S d4042, 1 min 47 s)
- Command (per split): `python tools/test_adaptive_model.py --cfg <infant yaml> TEST.MODEL_FILE <ckpt> TEST.USE_GT_BBOX True WORKERS 8 DEBUG.DEBUG False DATASET.ROOT $R/data/syrip/ DATASET.TEST_SET <split> OUTPUT_DIR/LOG_DIR $R/runs/<RUN>/upstream_eval/<split>/...`, driven by `scripts/eval_upstream.sh`. Logs show `=> loaded 1754 tensors strictly` (HRNet) and `332` (MobileNet).
- Upstream-printed results (pycocotools, flip test, DARK):

| Run | Split | AP | AP50 | AP75 | AR | AR50 | AR75 | Paper |
|---|---|---|---|---|---|---|---|---|
| R001 HRNet-W48 FiDIP | **test100** | **93.6** | 98.5 | 98.5 | 94.6 | 99.0 | 99.0 | 93.6 / 98.5 / 98.5 / 94.6 / 99.0 / 99.0 (exact) |
| R001 | validate_infant (README layout, test11 = test24 copy) | 92.1 | 97.1 | 97.1 | 93.6 | 98.0 | 98.0 | |
| R001 | val400 (397) | 90.4 | 98.8 | 95.8 | 91.9 | 99.5 | 96.5 | |
| R001 | validate500 (own boxes, D8 area) | 90.9 | 98.7 | 96.7 | 92.3 | 99.4 | 97.0 | |
| R002 MobileNetV2 FiDIP | **test100** | **79.1** | 99.0 | 89.3 | 84.1 | 99.0 | 92.0 | 79.2 / 99.0 / 89.4 / 84.1 / 99.0 / 92.0 |
| R002 | validate_infant | 77.7 | 97.6 | 88.0 | 83.1 | 98.0 | 91.0 | |
| R002 | val400 | 68.2 | 95.3 | 77.4 | 76.9 | 97.0 | 84.4 | |

- **The paper's numbers were evidently computed on the correct test images.** The README's own layout loses about 1.5 AP (93.6 → 92.1) through the duplicated test11.jpg. MobileNet's 0.1 AP gap is within version noise (cv2/torch builds).
- Outputs: `$R/runs/R00{1,2}_*/preds_{test100,val400,validate_infant[,validate500]}.json`, `metrics_upstream_*.txt`, `run.json`.
- Glitch: `run.json` lacked `time_end`. The final `--stamp end` line did not take effect in the job, but a stub of the same script tail works. I backfilled it from the slurm log, and the stamp is now written after every split.

### 2026-11-06 01:05–01:20 EST — smoke test, retrain submissions, hypotheses
- Checkpoint rules stated up front in `results/PREREGISTRATION.md` (rows repro-1/2, commit 3a06c5f): R010–R015 report BOTH the best-Val400 epoch and final epoch 20 on Test100; R016–R020 Val400 only.
- **Smoke R009_fidip-w48_smoke_s1** (job 11664893, gpu H200 d4053, 1 min 48 s; duplicate L40S job 11664960 cancelled while pending). Spec: λ 5e-4, PRE_EPOCH 1, END_EPOCH 1, Val400 eval only.
  - Pre-stage, then train at 0.12 s/iteration at batch 20; epochs.csv OK.
  - In-training Val400 AP 0.8811 equals the re-evaluation of `model_best.pth` (0.881).
  - **Peak GPU memory 11.8 GB per process**, so 3 seeds (~36 GB) are packed on one H200.
  - `loss_p` goes ≤ 0 at times: −λ·BCE (5e-4 × ~0.7 ≈ 3.5e-4) is the same order as the heatmap MSE (~4e-4). So the adversarial term is not negligible at λ = 5e-4, and P8 logs MSE separately.
- Submitted (D14: ≤ 2 running on gpu, followers chained with `--dependency=afterany`):
  - 11665025 `rp_R010-R012` (λ 5e-4, s1–s3)
  - 11665026 `rp_R013-R015` (λ 0, s1–s3)
  - 11665029 `rp_R016-R018` (λ sweep)
  - 11665030 `rp_R019-R020`
- **Hypotheses (logged before results):**
  - R010–R012 vs R013–R015: the paper reports +0.9 AP for domain adaptation (93.6 vs 92.7). With a 3-seed sd likely ~0.3–0.5 AP, we may or may not resolve it.
  - λ sweep R016–R018 (1e-4, 2e-3, 1e-2): small λ should approach the λ = 0 result. At large λ (2e-3, 1e-2) the −λ·BCE term dominates the ~4e-4 MSE and should hurt pose accuracy. Expect a flat or slightly peaked curve around 5e-4.
  - R019 paper recipe (Adam lr 1e-3, 100 epochs, batch 64, λ 5e-4; LR_STEP [40, 200] kept from the yaml, so ×0.1 at epoch 40): 10× higher LR on a partly frozen pretrained net may overfit 1200 images or destabilize. The longer schedule plus LR drop may compensate. This tests whether the paper recipe, not the yaml, explains 93.6.
  - R020 README-literal labels (syrip_for_train json: MINI-RGBD labels on SyRIP synthetic images): expect a clear drop, because the 1000 synthetic samples carry wrong keypoints. This quantifies what following the README literally costs.
- **Demo** job 11664937 (sharing L40S): all `tools/demo.py` modes passed.
  - 8 Test100 overlays with GT boxes, 2 per posture (ids 0, 3, 1, 6, 2, 5, 38, 69); mean conf 0.82–0.90.
  - Detector path: test0 mean conf 0.46 with the detector box vs 0.87 with the GT box. Box quality matters, a point for the report.
  - Single image and an 8-frame video also worked.
  - The Streamlit AppTest failed only because FIDIP_MODEL was not exported for that step. Fixed; rerunning as job 11665018.

### 2026-11-06 01:20–01:30 EST — demo rerun: second env bug (pyarrow vs numpy)
- Demo job 11665018 rerun:
  - `gpu_test.py` with warm-up: L40S, capability 8.9, 4096² fp32 matmul in 3.1 ms (45 TFLOP/s), GPU OK.
  - All `demo.py` modes OK again.
  - The Streamlit AppTest then failed with `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`. streamlit 1.39 pulled pyarrow 26.0.1, while upstream needs numpy 1.26.4.
  - Fix: pin `pyarrow==17.0.0` in `build_env_fidip.sbatch` and rebuild (job 11665101). Upstream code never imports pyarrow, and the retrain jobs had not started yet.
  - Demo rerun queued as job 11665103 (`afterok:11665101`). My first chained submission used `afterok` on the failed job, which can never be satisfied; I cancelled it immediately.
- Overlays copied to `results/figures/demo/{test100_gt,detector}/` (2 MB) and checked by eye (test38: crouching infant, skeleton aligned, left = blue, right = orange).
- Likely root cause of the missing `time_end` in R001/R002 run.json (21:10 entry): the old `sync_to_cluster.sh` extracted the tar *in place*, and bash reads scripts incrementally. A sync by another stream during job 11659813 could truncate and rewrite `eval_upstream.sh` under the running bash, which then hits EOF before the last line. The orchestrator's new sync (commit 6caddd9, staged unpack + atomic rename) removes this. My scripts also stamp the end time after every split now.

### 2026-11-06 01:30–01:40 EST — demo verified (STUDENT_GUIDE Part 6), retrains waiting on H200
- Job 11665103 (sharing L40S, 1 min 48 s, env with pyarrow 17):
  - `gpu_test.py` OK.
  - All `tools/demo.py` modes OK.
  - **Streamlit AppTest OK:** `table rows 17, mean conf 0.866, above-threshold at 0.30 = 17 / 17, at 0.95 = 6 / 17`. The slider changes the result.
  - The live-server check first failed: `curl localhost:8599` went through the compute-node HTTP proxy (squid on xfer-00, ERR_ACCESS_DENIED). Fixed with `curl --noproxy '*'`.
  - Job 11665259 (CPU): `health: ok`, `<title>Streamlit</title>`.
  - How to verify: `scripts/test_streamlit_demo.py` (AppTest drives the server-side-path input and the slider, since AppTest cannot simulate file uploads), plus a background `streamlit run --server.headless true` and `curl --noproxy '*' localhost:8599/_stcore/health`.
- Env exports had been silently reverted: a push from another stream overwrote `$R/repo/envs/fidip-pip-freeze.txt` (mtime 20:47 = the local copy) after my rebuild wrote it. Exports now go to `$R/envs/` (outside the synced tree) and are copied into the repo; the freeze shows pyarrow 17.0.0. Lesson: never write outputs into `$R/repo`.
- R010–R012 (11665025) and R013–R015 (11665026) are still PENDING (Priority; H200 contention; estimates slide 01:24 → 01:34, and 02:41 for the second).
  - `--test-only` probes: H200 with a smaller CPU/mem request → 02:07; A100-80GB (`--constraint=a100@80g`) → 02:42; sharing H100 → 03:43; sharing A100 → 02:46.
  - Kept as submitted.

### 2026-11-06 06:37–06:58 EST — R010–R012 results; R013–R015 failure and resubmission
- **R010–R012** (λ 5e-4, s1–s3; job 11665025, H200 d4054, 05:27 start, 18 min for the 3-run pack).
  - Peak GPU memory **19.8 GB per process** when packed (vs 11.8 GB in the 1-epoch single smoke).
  - Best Val400 epochs 20 / 19 / 18. Val400 AP and val loss still improve at epoch 20 (R010 val loss 4.14e-4 → 3.79e-4 by epoch 8, train MSE 4.5e-4 → 3.3e-4): **no overfitting within 20 epochs**.

| Run | Val400 AP (best) | Test100 AP best-ckpt, paper protocol (upstream print) | Test100 final | Test100 best, fixed protocol |
|---|---|---|---|---|
| R010 s1 | 90.63 | 93.85 (93.9) | 93.85 | 95.33 |
| R011 s2 | 90.38 | 93.74 (93.7) | 93.41 | 95.42 |
| R012 s3 | 90.53 | 93.07 (93.1) | 92.88 | 94.58 |
| **mean ± sd** | **90.51 ± 0.13** | **93.55 ± 0.42** | **93.38 ± 0.49** | **95.11 ± 0.46** |

  - eval_coco (paper protocol) equals the upstream print to rounding.
  - **The official recipe (+ COCO init, PRE_EPOCH 10 on train_pre, λ 5e-4, epochs selected on Val400) reproduces the paper's 93.6** (3-seed mean 93.55). The released checkpoint gives 93.6.
- **R013–R015 attempt 1 FAILED** (job 11665026, 1 min): `ValueError: Type mismatch (<class 'float'> vs. <class 'int'>) with values (0.0 vs. 0) for config key: TRAIN.LAMBDA`.
  - Cause: I passed `0` on the command line, and yacs refuses int-for-float.
  - The smoke only tested λ 5e-4, so this path was never exercised.
  - Dirs moved to `$R/runs/_failed/R01{3,4,5}_*_attempt1_job11665026`. Registry row status=failed.
  - **Resubmitted** with `TRAIN.LAMBDA 0.0` as job 11670943, on the same hardware as R010–R012 (H200) to avoid a hardware confound in the λ ablation. `--test-only` estimates: H200 07:28, A100-80GB 08:04, V100 12:15.
  - Lesson: pass floats as `0.0`, and smoke-test every distinct command line.
- R016–R018 (11665029) and R019–R020 (11665030) started on H200 d4053 at ~06:40. R020 uses `DATASET.TRAIN_SET train_infant_readme`, and R019 floats are written as `0.001`.

### 2026-11-06 06:58 EST — λ sweep outcome (R016–R018, s1, Val400 only; job 11665029, H200 d4053, 17 min 49 s, 19.7 GB/process)
| λ | run | best Val400 AP (epoch) | final-epoch Val400 AP | final train MSE | final val loss |
|---|---|---|---|---|---|
| 1e-4 | R016 | **91.02** (16; tie with epoch 20) | 91.02 | 2.56e-4 | 3.74e-4 |
| 5e-4 | R010–R012 | 90.51 ± 0.13 (3 seeds) | 90.39 ± 0.30 | ~3.3e-4 | ~3.8e-4 |
| 2e-3 | R017 | 89.67 (20) | 89.67 | 3.69e-4 | 4.21e-4 |
| 1e-2 | R018 | 85.78 (5) | 83.84 (oscillates 82–85) | 6.09e-4 | 5.99e-4 |
- **Outcome vs hypothesis:** as predicted, large λ hurts. At λ ≥ 2e-3 the −λ·BCE term outweighs the ~3e-4 heatmap MSE: train MSE rises with λ, and at 1e-2 Val400 AP peaks at epoch 5 and then degrades.
- Not predicted: **λ = 1e-4 beats the official 5e-4 by 0.5 AP** (single seed, about 4× the 3-seed sd of 0.13). This needs λ = 0 (R013–R015, pending) and more seeds before any claim. Test100 was not touched for the sweep.
- `eval/eval_coco.py` on the released preds (job 11671085, `scripts/eval_coco_repro.sh`) equals the upstream prints.
  - R001: Test100 paper **93.60** / fixed 94.99; Val400 90.36; validate500 90.92 / 91.15; validate_infant 92.15 / 93.64.
  - R002: Test100 79.14 / 79.84; Val400 68.18; validate_infant 77.74 / 78.62.
  - Our λ = 5e-4 retrains match the released model on Val400 (90.51 ± 0.13 vs 90.36).
- 07:01 EST: R013–R015 resubmission (11670943) started on H200 d4053 (same GPU type as R010–R012).

### 2026-11-06 07:05 EST — R020 outcome (README-literal training labels; job 11665030 packed with R019, H200 d4053)
- Same as R010 s1, except `DATASET.TRAIN_SET train_infant_readme`: syrip_for_train.zip's train json, whose 1000 synthetic entries carry MINI-RGBD labels.
- Val400 AP per epoch 85.1 → 89.3 (best, epoch 12) → 89.2 (final); upstream-printed best 89.3 / final 89.2.
- **vs R010 s1 (official labels, same seed): −1.3 AP; vs the 3-seed mean: −1.2 AP (about 9 sd).**
- Hypothesis confirmed: following the README literally costs about 1.2 AP on Val400. The damage is bounded because the 200 real images (oversampled to half of every batch) keep correct labels. Val400 only, as preregistered.
