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
