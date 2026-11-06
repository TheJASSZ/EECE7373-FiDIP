# notebook/analysis.md — live log for the analysis stream

Newest entries at the bottom. Timestamps in EST.

### 2026-11-05 20:10 EST: GT audit before writing the evaluator
I read the upstream evaluation path: `lib/dataset/syrip.py::evaluate`, `lib/nms/nms.py::oks_nms`, `lib/core/function.py::validate_adaptive`.
- Instance score = mean of keypoint maxvals > `IN_VIS_THRE` (0.2) × box score. The box score is 1 for GT boxes (`JointsDataset`: `score = db_rec.get('score', 1)`).
- Then OKS-NMS at 0.9. The NMS area is `prod(scale*200)`, the padded crop area.
- Then pycocotools COCOeval (keypoints, default sigmas, GT `area`).

I inspected the raw annotation JSONs on the cluster (`$R/data/raw/SyRIP/annotations/*`) with light reads on the login node.

**Finding 1: validate500 `area` is a constant placeholder.**
- validate100 (= Test100): `area == bbox_w*bbox_h` exactly for all 100 annotations, with float boxes.
- validate500: `area = 2443298.48` on all 500 annotations. That is sqrt(area) of about 1563 px for every infant, so OKS would be far too lenient.
- Train `200R_1000S` also has a constant `area` (14232.77).
- Without a fix, Val400 AP would be close to 100 for every model and useless for selection.
- I sent this to the orchestrator, who adopted it as D8: repro writes val400/validate500/train with `area = bbox_w*bbox_h`.
- `eval_coco.py` will refuse a GT whose areas are all identical unless `--area bbox` is passed.

**Finding 2: validate100 is a subset of validate500, but not an identical copy.**
- Image ids and filenames are the same for all 100 shared images.
- Boxes differ on all 100. IoU(v100, v500) has median 0.87 and min 0.70. v100 boxes are a median 12% larger by area (range 0.90 to 1.42).
- v500 boxes are mostly integers (475/500). Box area / tight-keypoint-box area has median 1.95 for v100 and 1.71 for v500.
- So Val400 GT boxes are tighter than Test100 boxes. That gives smaller OKS normalisers (slightly stricter OKS) and tighter top-down crops. The AP gap will be measured once the released-model predictions exist (see the to-do below).
- Keypoints are identical on 98/100 images. The 2 exceptions, both apparently re-annotated in v100 (Euclidean px, v100 vs v500):
  - **test83.jpg** (650×979, sqrt(area) about 659). l_hip 64.0 px, l_knee 59.5, r_hip 20.4, l_elb 18.9, r_elb 18.2, r_ank 17.5, l_sho 13.3. l_sho/l_hip/l_knee are v=2 in v100 but v=1 in v500, which is the only visibility disagreement.
  - **test84.jpg** (1536×2051, sqrt(area) about 1556). r_hip 91.6, l_knee 87.2, l_hip 83.4, r_ank 70.3, r_knee 59.6, l_sho 54.7 px.
- In OKS terms these are modest: about 6% of sqrt(area) for test84's hips, giving per-keypoint OKS of about 0.96. They show that Test100 labels were refined after validate500 was released.

**Finding 3 (important): annotation id 0 is a pycocotools artifact that is in the published numbers.**
- In validate100, annotation ids equal image ids, so `test0.jpg` has annotation id 0. Train has no id 0. Val400 (ids 100–499) has no id 0.
- COCOeval's `evaluateImg` stores `dtMatches = gt['id']`. `accumulate` then treats 0 as "unmatched".
- As a result, the image-0 GT can never be a true positive and its detection always counts as a false positive. That caps AR at 0.99 and costs about 1 AP.
- Verified with a 10-image toy GT and perfect predictions (pycocotools 2.0.8, script `scratchpad/probe/annid0.py`): ids from 0 give AP 0.811 / AR 0.900; ids from 1 give 1.000 / 1.000.
- This matches the README table, where AR50 is at most 99.0 for every model on Test100.
- Decision (sent to the orchestrator): the split JSONs keep ids as shipped. `eval_coco.py` reports two protocols:
  - `paper` (primary): ids as given. Comparable with the published 93.6 and with upstream's printed AP.
  - `fixed`: ids shifted off 0.
- The two are identical on Val400.

**To do (requested by the orchestrator):**
- Quantify the v100-vs-v500 GT effect on the 100 shared images with the released FiDIP model. The 2×2 needs predictions made with v500 boxes on those 100 images, i.e. preds on validate500.
- Over/underfitting curves from `epochs.csv`.
- An attempts table (`results/tables/attempts.csv` + LaTeX).

### 2026-11-05 20:19 EST: `eval/eval_coco.py` v1.0 built and committed (c6eb701), plus the id-0 artifact pinned down
**What I built.** `eval/eval_coco.py`, 24 pytest tests (`eval/tests/`), and `eval/README.md`. The pipeline:
- GT checks, including a guard that refuses constant areas.
- Robust prediction loading across 12 encodings.
- Upstream rescoring, `mean(conf>0.2) × 1`.
- A verbatim port of upstream OKS-NMS 0.9.
- pycocotools COCOeval under both protocols.
- Per-threshold AP/AR and `AP_80_95`.
- Per-image OKS and per-keypoint error/PCK normalised by sqrt(bbox w·h).

**How I verified it** (local venv with pycocotools 2.0.8 and numpy 2.4.4; all 24 pass in 0.8 s):
- GT-as-prediction gives AP = AR = 1.
- Monotone AP/OKS/PCK degradation over 7 noise levels with common random numbers.
- The 12 prediction encodings give bit-identical metrics.
- Rescoring invariances (same rule, junk file scores, monotone transforms, confidence scaling).
- Per-image OKS equals COCOeval's `E.ious`.
- Exact per-keypoint error and PCK on a constructed 0.15·sqrt(area) shift.
- OKS-NMS keep lists equal upstream `nms.py`'s `oks_nms` on 60 random instance sets. Upstream's functions are exec'd from the vendored file via `ast`, which avoids its compiled `.so` imports.

**The id-0 artifact, for the report** (decision D10):
- Toy reproduction (`scratchpad/probe/annid0.py`): a 10-image GT with perfect predictions gives AP 0.811 / AR 0.900 when annotation ids start at 0, and 1.000 / 1.000 when they start at 1.
- Responsible lines, pycocotools 2.0.8 `cocoeval.py`:
  - `evaluateImg` (def at l.235) initialises `dtm = np.zeros((T,D))`.
  - l.295 stores `dtm[tind,dind] = gt[m]['id']`, which is 0 for test0.
  - `accumulate` (def at l.315), l.369, concatenates `dtMatches`.
  - l.375–376: `tps = np.logical_and(dtm, ~dtIg)`, `fps = np.logical_and(~dtm, ~dtIg)`. So a match to GT id 0 counts as an FP, and the GT stays unmatched.
- pycocotools 2.0.11 (in `$R/envs/mmpose`) has the identical code (l.269, l.295, l.375).
- **Ceiling on Test100 under `paper`:**
  - AR (and AR50, AR75) ≤ 0.99.
  - For a perfect model, every per-threshold AP equals **(r + 0.99·(100 − r))/101**, where r is the score rank of test0's detection among the 100. For r = 1 it is 0.99·100/101.
  - So AP50 ∈ [0.9802, 0.9901], and AP for a perfect model lies in the same band. This is unit-tested for r ∈ {1, 2, 49, 100}.
  - The published AP50 = AP75 = 98.5 needs r ≈ 44–54. So on the 99 matchable images, DarkPose+FiDIP (and Finetune 92.7, and pretrained DarkPose 88.5) are already perfect at OKS 0.5 and 0.75. All their headroom sits at OKS ≥ 0.8. From the published numbers, FiDIP's AP_80_95 ≈ 86.3.
- **MMPose is different:** MMPose 1.3.2 `CocoMetric` uses `xtcocotools` 1.14.3. Its `cocoeval.py` initialises `gtm/dtm = -1` (l.428–429) and tests `dtm >= 0` (l.537–538, citing cocoapi PR #332).
  - So **MMPose's own printed Test100 AP is the `fixed` protocol**, about 1 AP above upstream-comparable numbers for identical predictions.
  - Only `eval_coco.py` numbers may be compared across frameworks. Message sent to the orchestrator.
  - Val400 has no id 0, so MMPose's printed Val400 AP is unaffected by this; other xtcocotools differences may remain.

**Surprises.**
- A clean mock "Test100" with noise σ = 0.025·sqrt(area) already reproduces the published pattern exactly: AP50 = AP75 = 98.2 and AR50 = AR75 = 99.0 under `paper`, against 100/100 under `fixed`.
- xtcocotools has silently fixed what pycocotools still gets wrong.

### 2026-11-05 20:44 EST: statistics, ensembling, figures, tables and the cluster pipeline (commits 7745e6c, cfc7c00, 802f9bb, bee2d8a)
**What I built**
- `eval/bootstrap_ci.py`: modes `paired`, `multiseed` (with optional `--hierarchical`), `contrasts` (Holm) and `single`. The fast path reads COCOeval's own per-image matches (`evalImgs`) and re-implements only `accumulate()`, vectorised.
- `eval/ensemble_preds.py`: mean or confidence-weighted coordinate averaging, computed as x0 + Σw(xi − x0)/Σw so an ensemble of identical files is bit-exact. Multi-person images are matched by GT box or by keypoint centroid, never by GT keypoints.
- `eval/taxonomy.py`: L/R swap per pair plus the whole-body mirror, in-plane torso angle, a front/back/profile view proxy, occlusion count and scale.
- `eval/analyze.py`, with subcommands `models`, `splits`, `curves` and `gtbox` (figures and tables listed in `eval/README.md`).
- `eval/visualize.py`: gallery, failures, compare.
- `eval/plot_style.py`: the shared figure style.
- `scripts/analysis_pipeline.py`, `scripts/slurm/analysis.sbatch`, `scripts/pull_analysis.sh`, `scripts/collect_results.py`.

**How I verified it**
- 58 pytest tests, passing locally (numpy 2.4.4) and on the cluster in `$R/envs/analysis` (job 11659381: numpy 1.26.4, pycocotools 2.0.8, the same as the fidip env; 57 passed then, before the gate test was added).
- Bootstrap fast path:
  - Bit-identical to COCOeval on the full set for the paper protocol with ids from 1 and from 0, and for the fixed protocol. The mock includes multi-person images, missing predictions, near-duplicate detections and score ties.
  - Equal to 1e-12 to COCOeval run on 8 explicitly duplicated (resampled) datasets.
  - Every bootstrap output records `self_check.max_abs_diff_vs_cocoeval` (0.0 so far).
  - Speed: B = 10000 paired on 100 images takes about 2 s.
- Holm adjustment checked against a hand computation. With identical files, Δ = 0, CI = [0, 0] and p = 1. A clear improvement gives p < 0.01, and with common noise every image wins.
- Taxonomy, on constructed poses: rotation bins at ±30/90/100/150/180°; the view sign is rotation-invariant and flips under mirroring; a wrist swap is detected alone; a full mirror flags all 8 pairs; small jitter and an unresolvable (coincident) GT pair are not flagged.
- Ensembling: identical inputs give identical outputs and metrics. Independent noise improves mean OKS. Both multi-person matchers keep persons apart.
- Figures: I rendered every one on a realistic mock (rotated, mirrored and occluded anatomical pose; 3 models, one with 3 seeds) and looked at each PNG. Fixes from that inspection:
  - colliding titles and labels;
  - in-plot annotations moved into legends;
  - a 2×3 taxonomy grid;
  - square, grey-padded crops;
  - neutral colours for splits, so slot colours keep meaning "model".
- Palette checked with the dataviz validator on a white surface:
  - 8 slots pass the adjacent-pair checks: worst CVD ΔE 9.1 (protan), normal-vision 19.6.
  - Only slots 1–3 pass all-pairs (9.2 / 24.0), so all-pairs figures colour at most 3 models and grey the rest.
  - Slots 3–5 have < 3:1 contrast, so every figure has a legend or direct labels.
- `collect_results.py` tested on the mock registry, pre-registration, configs and bootstrap. Fixes from that test:
  - Macro names keep their camel case (`\PaperFiDIPAP`).
  - Duplicate macros are de-duplicated.
  - The paper's implied AP_.80:.95 prints as 86.25. It is exactly 345/4, and 1-decimal round-half-even would print 86.2, not the 86.3 quoted earlier.
  - The bootstrap summary was renamed to avoid a glob collision with its own output.
- Test100 gate (`analysis_pipeline.test_allowed`, unit-tested): Test100 predictions are evaluated only for released or `_retrain` runs and pre-registered RUN_IDs. The λ sweep and MMPose screening are withheld.

**New input from repro.** The syrip_for_train copy of validate100 has GT `images[].posture` labels: supine 43, prone 41, sitting 11, standing 5.
- `analyze.py` now takes `"posture": <json>` in the manifest. With it, the taxonomy and error-vs-orientation figures use the GT posture instead of the geometric proxy.
- It also writes `view_proxy_vs_posture_<split>.csv`, which validates the proxy on Test100 so it can be trusted on Val400 (no labels there).

**Waiting on:** `$R/data/SPLITS_READY`, the fidip env (job 11659313), the released checkpoints, and R001 predictions.
