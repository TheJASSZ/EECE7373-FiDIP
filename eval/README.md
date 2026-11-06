# eval/ — the project's single evaluator and analysis tools

Every number in the report comes from `eval_coco.py` applied to a prediction JSON (decision D2). Use it for both upstream FiDIP and MMPose predictions. Never use a framework's own printed AP. See "Why not MMPose's printed AP" below.

## Quick start
```bash
# cluster: conda activate $R/envs/fidip   (needs numpy + pycocotools; nothing else)
python eval/eval_coco.py --gt $R/data/splits/test100.json \
    --pred $R/runs/R001_fidip-w48_released/preds_test100.json \
    --out  $R/runs/R001_fidip-w48_released/metrics_test100.json \
    --per-image $R/runs/R001_fidip-w48_released/per_image_test100.json
python -m pytest eval/tests -q        # unit tests (mock data; no SyRIP needed)
```
Options:
- `--no-rescore`: the primary metrics use the file's own `score`. Both variants are always written.
- `--area {gt,bbox}`: the OKS scale. Default `gt`.
- `--protocol {paper,fixed}`: the primary matching protocol. Default `paper`. Both are always written.
- `--nms {oks,none}`, `--nms-thr 0.9`, `--kpt-thr 0.2`.
- `--run-id`: provenance tag. Inferred from a `runs/<RUN_ID>/` path when omitted.
- `--verbose`: print pycocotools' table.

## What it computes
1. **GT checks.**
   - Exactly one person category with 17 keypoints and unique annotation ids.
   - `num_keypoints` is filled if missing.
   - **Degenerate-area guard**: refuses a GT whose `area` is identical for every annotation unless `--area bbox` is given. SyRIP's raw validate500 and train files ship a constant placeholder (2443298.48 and 14232.77), which would make OKS meaningless (D8). The split files in `$R/data/splits/` use `area = bbox_w*bbox_h`, the convention of the official validate100 file.
   - The area rule actually used is recorded in `gt.area_check`.
2. **Prediction loading.** Robust to:
   - `image_id` as int, `"27"`, `27.0`, `"test27"`, `"test27.jpg"`, or an `img_path`/`file_name` field;
   - keypoints as flat 51, nested 17×3, 34 + `keypoint_scores`, or MMPose `pred_instances` (1×17×2 + scores);
   - extra fields (`center`, `scale`, `bbox`, `id`, `bbox_score`…);
   - a missing `category_id`;
   - a `{"annotations": [...]}` wrapper.

   Predictions for images not in the GT are dropped with a warning. That lets you evaluate validate500 predictions on the Val400 subset. GT images without a prediction count as misses.
3. **Rescoring** (default): `score = mean(conf[conf > 0.2]) × box_score`, where `box_score = 1` for GT boxes and the score is 0 if no confidence exceeds 0.2.
   - This is upstream FiDIP's rule (`lib/dataset/syrip.py::evaluate`) and MMPose's `score_mode='bbox_keypoint'`.
   - With one GT box per image, AP depends only on how scores *rank* the images, so one shared rule keeps frameworks comparable.
   - `pred.max_abs_diff_file_score_vs_rule` shows whether a file's own scores already follow the rule.
4. **OKS-NMS 0.9** per image. This is a verbatim port of upstream `lib/nms/nms.py::oks_nms`; the NMS area is `prod(scale*200)` when the file carries upstream's `scale`. It is a no-op with one GT box per image (SyRIP).
5. **pycocotools COCOeval** (`iouType='keypoints'`, default COCO-17 sigmas, GT `area`, maxDets 20).
   - Only `image_id`, `category_id`, `keypoints`, and `score` are passed to `loadRes`, so every framework takes the same code path.

## The two matching protocols (D10)
SyRIP Test100 (validate100) uses `annotation id == image id`, so **test0.jpg has annotation id 0**. pycocotools (2.0.8 and 2.0.11) stores the matched GT id in `dtMatches` and then treats 0 as "unmatched":
- `cocoeval.py:295` `dtm[tind,dind] = gt[m]['id']`
- `cocoeval.py:375-376` `tps = logical_and(dtm, …)` / `fps = logical_and(logical_not(dtm), …)`

The consequence is that test0's GT can never be a true positive and its detection always counts as a false positive.

| Protocol | Output key | GT ids | Use |
|---|---|---|---|
| `paper` (primary) | `metrics` | as shipped | Comparable with the published FiDIP table and with upstream's printed AP. Headline tables and bootstraps. |
| `fixed` | `metrics_fixed` | shifted so none is 0 | Sensitivity check; the "true" COCO AP. Identical to `paper` when no id is 0 (e.g. Val400). |

Ceiling under `paper` on Test100 for a *perfect* model, where r is the score rank of test0's detection (unit test `test_paper_protocol_ceiling_for_perfect_model`):
- AR = 0.99.
- Every per-threshold AP equals (r + 0.99·(100−r))/101. That lies in [0.980, 0.990].

The published AP50 = AP75 = 98.5 sits inside this band, so on the 99 matchable images those models are already perfect at OKS 0.5 and 0.75.

**Why not MMPose's printed AP.** MMPose 1.3.2's `CocoMetric` uses xtcocotools 1.14.3. That fork initialises `dtm` to −1 and tests `dtm >= 0` (cocoapi PR #332), so MMPose's own Test100 numbers correspond to the `fixed` protocol. They are about 1 AP higher than upstream FiDIP's printed numbers for identical predictions.

## Output JSON (`--out`)
| Key | Content |
|---|---|
| `run_id`, `split`, `evaluator` | Provenance. `gt.sha256` and `pred.sha256` pin the exact files. |
| `metrics` | AP, AP50, AP75, APm, APl, AR, AR50, AR75, ARm, ARl, `AP_80_95`. Values are fractions; `null` means COCOeval's −1 (e.g. APm, since no SyRIP infant is "medium"). |
| `metrics_fixed` | Same, other protocol. |
| `metrics_original_scores[_fixed]` | Un-rescored (the file's `score`), if every prediction has one. |
| `ap_per_oks_thr` / `ar_per_oks_thr` | AP/AR at OKS 0.50…0.95, per protocol, from COCOeval's precision/recall arrays. `AP_80_95` is their mean over 0.80–0.95, where a 98.5-AP50 model has all its headroom. |
| `oks` | Per-image OKS summary: mean, median, min, p10, fraction ≥ 0.5/0.75/0.9/0.95. |
| `per_keypoint` | Per joint: n, n with v=1, mean/median error / sqrt(GT bbox w·h), PCK@0.05/0.1/0.2. |
| `per_keypoint_summary` | Pooled over joints. |
| `pred.*` | Counts: n_in, dropped, after NMS, GT images without a prediction, score ties. |
| `id0_artifact` | D13. Where test0's detection ranks by score (`score_rank`, 1 = most confident; `rank_percentile`), its OKS, and `paper_minus_fixed` for AP/AP50/AP75/AP_80_95/AR. `null` if the GT has no annotation id 0. |
| `warnings` | e.g. the id-0 note, dropped predictions. |

**Per-keypoint normaliser.** sqrt(GT bbox w·h). For Test100 and the fixed Val400 this is exactly sqrt(`area`), the OKS scale, so PCK and OKS use the same ruler. It also stays meaningful for any GT whose `area` field is unreliable. Labelled keypoints (v=1 or 2) count. A GT without any prediction counts as an infinite error, so it fails every PCK.

**Per-image records (`--per-image x.json|x.csv`).** One record per GT instance:
- the best-OKS detection's OKS (exactly COCOeval's `computeOks`, unit-tested against `E.ious`);
- the score used for ranking, the rescored score, and the file score;
- GT and predicted keypoints;
- per-keypoint normalised errors and per-keypoint OKS terms.

`analyze.py`, `visualize.py`, and `bootstrap_ci.py` consume these.

## Tests (`eval/tests/`, pytest)
- GT-as-prediction gives AP = AR = 1.
- The id-0 artifact (paper vs fixed) and the exact paper-protocol ceiling formula.
- Monotone degradation under increasing keypoint noise (common random numbers).
- Format robustness: 12 encodings give identical metrics.
- Unknown images are dropped; missing predictions count as misses.
- Rescoring invariances:
  - a file scored with the same rule gives identical results;
  - random file scores are ignored after rescoring;
  - strictly monotone score transforms don't change un-rescored AP;
  - confidence scaling doesn't change the result.
- The rescore rule equals upstream's loop.
- The constant-area guard.
- Per-image OKS equals COCOeval's `ious`.
- Exact per-keypoint error/PCK on constructed shifts; v=0 joints are excluded.
- Per-threshold AP averages back to AP/AP50/AP75.
- OKS-NMS keep lists equal upstream `nms.py` on random instance sets.
- CLI end to end (JSON + CSV).

## Statistics: `bootstrap_ci.py`
Modes: `paired` (one file per arm), `multiseed` (lists of seed runs), `contrasts` (a declared JSON spec with Holm adjustment), and `single` (CI of one model's AP).
- **Resampling.** The statistic is the COCO AP of a resampled multiset of images, with B = 10000 resamples by default and fixed seed 0. Both arms use the *same* resamples (paired).
- **Multi-seed.** The output gives:
  - mean ± std across seeds (seed variance);
  - a bootstrap over images of the *seed-averaged* AP (sampling variance);
  - `seed_pairs_a_gt_b`, the fraction of (a, b) seed pairs with AP_a > AP_b;
  - optionally `--hierarchical`, which resamples seeds and images jointly.
- **Inference.** A 95% percentile CI of Δ. The one-sided p for H1: A > B is (1 + #{Δ* ≤ 0})/(B + 1). The Holm step-down adjustment runs across the declared contrasts.
- **Both protocols (D13).** Every comparison also runs under the other protocol on the *same* resamples (`sensitivity`), and Holm is applied per protocol.
  - `a_gt_b_robust_both_protocols` is true when the 95% CI excludes 0 under both.
  - `robust_claim` (contrasts mode) also requires Holm rejection under both.
  - The headline is `paper`. Only a robust improvement counts.
- **Consistency.** Per-image OKS win/tie/loss counts, with an exact sign test and a Wilcoxon signed-rank test.
- **Exact fast path.** Per-image matches are read verbatim from `COCOeval.evalImgs`, and only `accumulate()` is re-implemented. Every run asserts that the full-set fast AP equals COCOeval to ≤ 1e-12 (`self_check` in the output). Tests also compare against COCOeval on explicitly duplicated datasets.

The spec for `contrasts` mode:
```json
{"B": 10000, "seed": 0, "protocol": "paper", "metric": "AP", "alpha": 0.05,
 "contrasts": [{"name": "ViTPose-H vs FiDIP", "a": ["$R/runs/R030_..._s1/preds_test100.json", "..."],
                "b": ["$R/runs/R001_fidip-w48_released/preds_test100.json"], "label_a": "ViTPose-H",
                "label_b": "FiDIP released", "hypothesis": "..."}]}
```

## Ensembling: `ensemble_preds.py`
Averages coordinates across N prediction files, either a plain mean or `--weighting conf`; the confidence is the mean confidence.
- Instances are matched per image. Multi-person images are matched by GT box or by keypoint centroid, never by GT keypoints.
- An ensemble of identical files is bit-identical to its input (unit-tested).
- A `.meta.json` sidecar records the input sha256 hashes.

## Analysis and figures
- `taxonomy.py`: the failure taxonomy (pure, unit-tested).
  - **L/R swap per pair.** Swapping the two predicted joints must more than halve their error and raise OKS by > 0.02, and the GT joints must be > 0.05·sqrt(area) apart. A whole-body mirror gain is also reported.
  - **In-plane torso angle**, from mid-hip to mid-shoulder: upright < 45°, sideways 45–135°, inverted > 135°.
  - **View proxy.** The sign of cross(torso, l_sho − r_sho) is rotation invariant: front ≈ supine, back ≈ prone, profile ≈ side-lying.
  - Occlusion (number of v=1 joints) and scale.
- `analyze.py models --manifest M.json`:
  - figures: AP vs OKS threshold (paper/fixed protocols, published points, ceiling band), OKS histogram and survival curve, per-keypoint error/PCK with CIs, failure-taxonomy bars, error vs orientation, score vs OKS (ranking quality), and pairwise per-image OKS scatter;
  - tables: per-threshold AP, per-keypoint, taxonomy, L/R swaps, occlusion, ranking quality (oracle AP), per-image taxonomy, and ΔOKS by category.
- `analyze.py id0rank --gt test100.json --model "label=preds.json:slot"` (D13): the paper and fixed AP of one fixed prediction set as test0's detection is moved through every score rank, with the model's actual rank marked.
- `analyze.py splits` compares Test100 and Val400 difficulty. `analyze.py curves` plots per-epoch train/val loss and Val400 AP/AR. `analyze.py gtbox` builds the v100-vs-v500 GT-box 2×2.
- `visualize.py gallery|failures|compare`: skeleton overlays on square crops. GT is white with hollow markers, the prediction is in the model colour, and marker shape encodes body side.
- `plot_style.py`: one palette, validated for colour-vision deficiency. Vector PDF and 300-dpi PNG.

## Cluster pipeline
- `sbatch scripts/slurm/analysis.sbatch eval-all` evaluates every `$R/runs/*/preds_<split>.json` into `$R/analysis/eval/<RUN_ID>/` and writes `index.csv`.
  - Files whose sha256 hashes are unchanged are skipped.
  - **Test100 files are evaluated only for released/retrain reproduction targets and pre-registered RUN_IDs.** Everything else is withheld and never computed (`analysis_pipeline.test_allowed`, unit-tested).
  - **Derived sensitivity split `test100_clean`.** Test100 minus the 5 images that are exact (md5: test 20, 42, 46, 50) or near-exact (test 7) copies of annotated official training images. The ids are read from repro's `VERIFY.json`. It is written to `$R/analysis/splits_derived/`, evaluated automatically for every run allowed on Test100, and gated identically. test0 stays, so both protocols apply.
- `bash scripts/pull_analysis.sh` pulls metrics, figures, and tables to the Mac.
- `python scripts/collect_results.py` writes `results/tables/*.csv|json` and `report/generated/results_tables.tex`, the booktabs tables and `\newcommand` macros. Every number carries its RUN_IDs. `report/generated/MACROS.md` lists every macro.
