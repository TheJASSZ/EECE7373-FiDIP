# AGENT.md — running lab notebook

Claude Code (model: Claude Opus 5.5) keeps this notebook live, working alongside the student (Vishnu Purohitham).
- Entries are chronological, with the newest at the bottom.
- This file records what the human asked, what the agent tried, what failed, and why the direction changed.
- Per-stream detail (commands, job IDs, stack traces) lives in `notebook/<stream>.md`.
- Inventory and provenance live in `MANIFEST.md`. Cluster how-to lives in `SKILL.md`.

## Decision index
| # | When | Decision | Why |
|---|---|---|---|
| D1 | 11-05 19:45 | **Use Val400 (validate500 \ validate100, 400 real images) for all model and epoch selection.** Test100 (validate100) is used only for pre-registered final evals. | Test100 is the paper's reporting set. The upstream training loop appears to pick `model_best` on the evaluation set, which amounts to test-set selection. The bonus rules forbid test-set tuning. Val400 is disjoint from Test100 and leaves the training data identical to the paper's (200 real + 1000 synthetic). |
| D2 | 11-05 19:45 | **Every number comes from one evaluator** (`eval/eval_coco.py`: pycocotools, standard 17-keypoint COCO sigmas, GT boxes, one rescoring rule) applied to prediction JSONs. | The rubric asks for "comparable evaluation conditions". Two frameworks (upstream FiDIP, MMPose) must not differ in their metric code. |
| D3 | 11-05 19:40 | **Data, environments and checkpoints stay on Explorer `/scratch`.** Only code, docs, metrics and figures sync to the Mac. | The Mac has 7.6 GB free (97% full). The cluster has H200/A100 GPUs. |
| D4 | 11-05 19:48 | **Run four parallel agent streams** (repro, improve, analysis, report), with this orchestrator owning AGENT/MANIFEST/SKILL. | The human asked to "deploy as many agents quickly". The streams are independent once data is downloaded. |
| D5 | 11-05 20:00 | **Reproduce FiDIP with `TRAIN.CHECKPOINT` = COCO DarkPose weights and `TRAIN.LAMBDA 0.0005`.** Also run the same recipe with λ = 0 (the paper's "DarkPose + Finetune" row) as the domain-adaptation ablation. 3 seeds each; per-epoch validation on Val400. | Code audit: the README command trains from **random init** with **λ = 0**. Only the `_custom.yaml` turns adaptation on (λ = 5e-4, PRE_EPOCH 10). Running the README literally would not be FiDIP. |
| D6 | 11-05 20:00 | **Build the FiDIP env pip-only** (py3.12, torch 2.3.1 cu121, numpy 1.26.4…) instead of `conda env create -f fidip_env.yml`. | The yml's pip stage pins `grpcio==1.48.2` (no cp312 wheel) and `mkl-*` packages that have no wheels. The packages the code actually imports are pinned to the yml's versions. |
| D7 | 11-05 20:00 | **Domain-classifier pre-stage:** if `train_pre_infant` (1904 COCO + 2000 SURREAL adults) isn't in `syrip_for_train.zip`, pre-train the classifier on SyRIP train (infant real vs synthetic) instead, and document the deviation. | The adult set is "available on request" and nothing in the repo builds it. The SURREAL license blocks rebuilding it quickly. |
| D8 | 11-05 20:06 | **Recompute `area = w×h` for val400/validate500** (and the neutral train JSON). Keep Test100 = validate100 as shipped. | validate500 ships a constant placeholder area (2443298.48), which makes OKS meaningless. validate100 uses w×h. |
| D9 | 11-05 20:08 | **Add a single-seed λ sweep {1e-4, 2e-3, 1e-2}** (Val400 only), plus the guide's demo CLI/Streamlit/gpu_test and per-epoch loss/AP CSVs. | STUDENT_GUIDE lists DA tweaks, demos and over/underfitting monitoring, and asks for the accuracy of every attempt. |
| D10 | 11-05 20:11 | **Keep annotation ids as shipped.** Headline = `paper` protocol; also report a `fixed` protocol (ids shifted off 0) as a sensitivity check. | Test100 has annotation id 0. pycocotools' accumulate treats matched GT id 0 as unmatched, so AR ≤ 0.99 and AP drops ~1 pt. The published numbers carry this (AR50 = 99.0 across the table). |
| D11 | 11-05 20:12 | **Improvement strategy:** ViTPose-L/H tested at 384×288 as well as 256×192 (resolution arm). HRNet-W48+DARK 384×288 in MMPose as the control. 1:1 real:synthetic mixing as in FiDIP. Recipe knobs (rotation ±180°, cosine/EMA) chosen on Val400. 5 seeds for the finalists if runs are short. A seed/cross-architecture coordinate ensemble as a secondary, pre-registered claim, always shown next to single-model numbers. | Literature: no published result beats 93.6 under this protocol. AP50 = AP75 = 98.5, so the headroom is at strict OKS (0.80–0.95), i.e. precise localization. Resolution and backbone are the levers. ViT-B lost to HRNet on another infant set (NeoPose), so B is skipped. RTMPose/Sapiens are excluded because of their extra pretraining data. |
| D12 | 11-05 20:12 | **Add one paper-recipe attempt** (Adam 1e-3, 100 epochs, batch 64, λ = 5e-4) next to the yaml recipe. | The paper (arXiv v5) and the released yaml disagree: lr 1e-3/100 ep/batch 64/λ 5e-4 vs 1e-4/20 ep/batch 20/λ 0. |
| D13 | 11-05 20:52 | **An improvement claim counts only if it holds under both evaluation protocols.** The headline uses `paper`, comparable with 93.6. The paired-bootstrap CI must exclude 0 under both `paper` and `fixed`. Report test0's score rank for every model. | Verified on the real Test100 GT: at realistic accuracy, `paper` vs `fixed` differ by 0.8–2.1 AP depending only on where test0's confidence ranks. A one-image artifact is about as large as the gains we expect. |
| D14 | 11-05 20:53 | **GPU slot policy:**<br>• repro and improve each get ≤ 2 running / ≤ 4 submitted on `gpu`, and ≤ 1 / ≤ 2 on `sharing`<br>• analysis uses CPU partitions only<br>• job-name prefixes `rp_` / `im_` / `an_`<br>• pack several runs per GPU job<br>• idle slots may be borrowed<br>• repro's core-points runs have priority | Shared per-user QOS caps (gpu 4 running / 8 submitted / 4 GPUs / 8 h; sharing 2 / 4 / 2 / 1 h). Without a policy the bonus screening could starve the reproduction runs. |
| D15 | 11-05 21:07 | **Val400 = 397 images.** Ids 207/217/222 dropped as duplicates (exact md5) or near-duplicates (dHash ≤ 10 and pose-OKS ≥ 0.85, rule fixed in advance) of Test100 images. | Without this, the selection set would contain Test100 images. That leaks test information into model selection. |
| D16 | 11-05 21:07 | **Official Test100 stays as shipped (primary).** Also report **Test100-clean** (95 images, minus {7, 20, 42, 46, 50}) as a sensitivity check. | The official training set contains 4 byte-identical copies and 1 near-copy of Test100 images. Keeping the official set preserves comparability with 93.6; the clean subset shows whether memorization affects conclusions. |
| D17 | 11-05 21:07 | **Train on SyRIP.zip `200R_1000S` labels, not syrip_for_train.zip's train JSON.** Score test images from SyRIP.zip. D7 revised: `train_pre_infant` **is** shipped, so PRE_EPOCH 10 uses the real adult set. | syrip_for_train (2024) has identical image bytes but MINI-RGBD labels on all 1000 synthetic entries (median 191 px off), and its test11.jpg is a copy of test24.jpg. Optional attempt R020 quantifies what training on the README-literal labels costs. |

---

## Log

### 2026-11-05 ~19:25–19:52 EST — Session 1: kickoff and planning (plan mode, read-only)
**Human asked:**
- First message: help completing the FiDIP hands-on exercise with "maximum points + bonus". The grading breakdown was pasted (50 pts: implementation and reproducibility 20, results and evaluation 15, report and documentation 15; up to +5 bonus for verified improvement). The human also asked to "create a folder in documents and start working".
- Then: "Wait". The session switched to plan mode.
- Shared the course slides ("EECE 7373 — Utilizing the Explorer Cluster", incl. required SKILL.md / AGENT.md / MANIFEST.md) with "use NEU HPC if you need".
- Follow-ups: "Please continue on the work", "Deploy as many agents quickly to finish off the work in efficient way possible".

**Agent did (read-only exploration):**
- **Local machine:** Apple M4, 16 GB RAM, **7.6 GB free disk**. conda, pandoc, gh and git are present. No LaTeX.
- **Course material search:** Spotlight found no STUDENT_GUIDE.md, SyRIP or FiDIP files locally. `~/.ssh/config` already has `Host explorer`.
- **First SSH probe:** hung more than 60 s. The cause was a slow `du ~` / `module avail` on the login node (stdout was buffered behind `head`). I stopped it.
- **Verbose SSH:** showed public-key auth succeeding in 3.2 s, so **passwordless SSH works**.
- **Lighter probe (with `timeout` wrappers):**
  - Slurm accounts: `cs5100.202610`, `cs5130.202630` (default).
  - Partitions: `gpu` (8 h; H200/A100/V100), `sharing` (1 h; A100/H100/L40…), `short` (2 d).
  - Modules: `miniconda3/25.9.1`, `cuda/12.1.1…12.8.0`.
  - `/scratch/purohitham.v` exists.
  - **`/projects/aclab` → permission denied**, so the lab's copy of the data can't be reused.
  - No TeX on the cluster.
- **Official repo README** (github.com/ostadabbas/Infant-Pose-Estimation):
  - Published Test100 numbers: DarkPose+FiDIP HRNet-W48 384×288 **AP 93.6** (AR 94.6). SimpleBaseline+FiDIP R-50 91.1. MobileNet+FiDIP 79.2. DarkPose fine-tune without DA 92.7.
  - Environment is now Python 3.12 / CUDA 12.1 (`fidip_env.yml`).
  - Checkpoints are on Google Drive.
- **SyRIP dataset page:** `SyRIP.zip` 313 MB, `syrip_for_train.zip` 608 MB. The README says train = 200 real + 1000 synthetic, validate500 = 500 real, and **validate100 (= Test100) ⊂ validate500**.
- **Background Explore agents launched:**
  1. A code audit of the upstream repo: env, configs, training flow, what `model_best` is selected on, needed compat patches.
  2. A literature/engineering scan of what has beaten FiDIP on SyRIP and how (ViTPose / MMPose specifics).

**What failed / surprises:**
- The hung SSH probe described above.
- No lab-share access.
- The Mac disk is nearly full.
- `train_pre_infant` (1904 COCO val2017 + 2000 SURREAL adults, used by FiDIP's domain-classifier stage) is "available on request" per the README. That is a possible reproduction blocker; it is being checked.

**Decisions:** D1–D4 above.

**Plan:** written in plan mode and **approved by the human at ~19:50 EST**. Summary:
- Reproduce: released checkpoints, then an official retrain × 3 seeds.
- Improve: controlled backbone swap (HRNet-W48 vs ViTPose) under one MMPose recipe, plus data/augmentation/TTA ablations, all selected on Val400.
- Then pre-register, run the final Test100 eval with paired-bootstrap CIs, write the LaTeX report, and have a fresh-context reviewer grade against the rubric.

### 2026-11-05 19:52 EST — Step 0: scaffolding
- Created the local git repo `~/Documents/EECE7373-FiDIP` with the planned layout.
- Created the cluster tree `/scratch/purohitham.v/fidip/{repo,data,models,envs,runs,slurm_logs,logs}` and `~/fidip_artifacts`.
- Started the SyRIP download + unzip as a background job on the login node (pid 687507, explorer-02, 19:54 EST). Log: `$R/data/downloads/download.log`. Markers: `DOWNLOAD_DONE`, `UNZIP_DONE`. Checksums go to `SHA256SUMS`.
- Wrote the first versions of SKILL.md (connection facts, partitions, ops, guardrails), MANIFEST.md (inventory skeleton) and CLAUDE.md (project rules every agent loads).

### 2026-11-05 19:55–20:04 EST — Code audit lands; first infrastructure failures; workers launched
**Code-audit findings** (Explore agent, upstream @ a76b4f4). These change the plan; see D5–D7.
- **Not reproducible as written:**
  - `tools/train_adaptive_model_hrnet.py` with only `--cfg` starts from **random weights**: `MODEL.PRETRAINED` parses as `''`, and `TRAIN.CHECKPOINT` must be passed.
  - The infant yaml has **`TRAIN.LAMBDA 0`** and `PRE_EPOCH 0`, so it is plain fine-tuning, not FiDIP.
- **Test-set selection:** the per-epoch validation set is `DATASET.TEST_SET = validate_infant` (= Test100), and `model_best.pth` is chosen on it. This confirms D1.
- **Missing adult set:** the loader always builds `train_pre_infant` (needs its JSON even when PRE_EPOCH = 0). That adult real/synthetic set is not public.
- **Other details:**
  - The adaptation is alternating updates with a negative-weighted BCE (`loss_p = MSE − λ·BCE`), not a gradient-reversal layer.
  - The first 109 parameter tensors are frozen.
  - The balanced real/synthetic sampler doesn't reshuffle.
  - `test_adaptive_model.py` loads weights with `strict=False`, so a `module.`-prefix mismatch would fail silently.
  - The compiled `nms` libs are never actually used, so `make` can be skipped.
  - `fidip_env.yml`'s pip stage has unbuildable pins.
- The full patch list (P1–P10) went into the repro worker's brief.

**Failures:**
- The SyRIP **unzip was `Killed` on the login node** (login nodes enforce limits even for IO jobs). The downloads themselves finished (SHA256 below). Lesson: unzip inside a Slurm CPU job. Delegated to the repro worker.
- **`rsync` from the Mac failed** ("unexpected end of file"). macOS ships openrsync (protocol 29), which doesn't interoperate with the cluster's rsync 3.2.5. Fix: `scripts/sync_to_cluster.sh` (tar-over-ssh push, never deletes remote files) and `scripts/sync_results.sh` (pulls only small run artifacts).
- The auto-mode guard **refused a `rm -rf` of a nested `.git`** while vendoring upstream. Fix: imported the pinned commit from the GitHub tarball API instead (no deletion needed), recorded in `third_party/UPSTREAM_COMMIT`.

**Download checksums** (sha256):
- `SyRIP.zip`: 1d13e816…1692
- `syrip_for_train.zip`: 1f267878…ee1b
- The full values are in MANIFEST §3.

**20:03, launched 4 background workers with self-contained briefs:**
- **repro:** data → splits marker → env/patches/ckpts → released-ckpt eval → λ = 5e-4 / λ = 0 retrains × 3 seeds → demo.
- **improve:** MMPose env → HRNet-W48 vs ViTPose-L/H screening on Val400 → ablations → 3-seed confirmatory set; no Test100 before pre-registration.
- **analysis:** unified evaluator + tests → bootstrap CIs → figures and failure taxonomy → results/LaTeX table generator.
- **report:** LaTeX report, verified refs, TikZ diagrams, README, build + visual check.

The literature-scan Explore agent is still running. Its findings will be forwarded to the improve worker.

### 2026-11-05 20:05–20:08 EST — Data bug in official GT; STUDENT_GUIDE.md arrives
**Analysis worker found a GT defect (D8):**
- In `validate500`, every annotation has the same placeholder `area` (2443298.48, i.e. √area ≈ 1563 px). OKS normalizes by area, so Val400 AP would be ~100 for any model and useless for selection.
- `validate100` (Test100) uses `area = bbox_w × bbox_h` exactly.
- The train JSON also has a constant placeholder area (14232.77).
- validate100 ⊂ validate500 (same ids and filenames). Keypoints are identical on 98/100 images. The validate100 boxes are a median 12% larger than validate500's boxes for the same images.
- **Decision D8:**
  - Build `val400.json` / `validate500.json` with `area = w × h` from validate500's own boxes.
  - Use `test100.json` = validate100 exactly as shipped.
  - Add an assertion in `make_splits.py`, and a degenerate-area guard in `eval_coco.py`.
  - Relayed to the repro worker before it wrote the splits.
  - Asked the analysis worker to quantify the v100-vs-v500 box effect (AP of the released model on the 100 shared images under both GTs).

**Human provided `STUDENT_GUIDE.md`.** It is mostly consistent with the plan. New requirements are tracked in `docs/REQUIREMENTS_CHECKLIST.md`:
- `tools/demo.py` with the guide's CLI, plus `tools/streamlit_demo.py`.
- `gpu_test.py`.
- Over/underfitting monitoring (train/val loss and val AP per epoch).
- A MANIFEST that reproduces the results "from this file alone".
- A report with the accuracy of every attempt, the final best config vs baseline, and a reflection on the agentic workflow.
- The guide names "domain-adaptation tweaks" as an expected extension. **Decision D9:** add a cheap λ sweep (single seed, Val400 only) to the repro stream as documented DA attempts.

**Integrity note:**
- The guide's `gpu_test.py`, `demo.py`, `streamlit_demo.py` and `fidip_simple.yml` aren't in upstream.
- A GitHub code search for them only surfaced classmates' public forks, whose notes say the files exist nowhere public.
- We **did not open or use any other student's repository**. We implement these files ourselves from the guide's documented interface.

### 2026-11-05 20:09 EST — Global secrets rule
- **Human asked:** "Yes please add it globally", i.e. the STUDENT_GUIDE Part 0.2 recommendation.
- Created `~/.claude/CLAUDE.md` with the course's Secrets rule. The file didn't exist before. The rule says:
  - read credentials from `.env`;
  - never print or commit it;
  - add `.env` to `.gitignore` before creating it;
  - pass secrets via stdin/env vars, not CLI args.
- This applies to every Claude Code project on this machine. The project-level `CLAUDE.md` already has the same rule.

### 2026-11-05 20:11 EST — pycocotools id-0 artifact on Test100 (D10)
- **Analysis worker:** validate100 annotation ids equal image ids, so `test0.jpg` has annotation id 0. COCOeval's `evaluateImg` stores matched GT ids in `dtMatches`, and `accumulate` treats 0 as "unmatched". So the image-0 infant is never a TP, and its detection counts as an FP.
- **Toy check** (pycocotools 2.0.8, 10 perfect predictions): AP 0.811 / AR 0.900 with ids starting at 0, vs 1.000 / 1.000 with ids starting at 1.
- **Fingerprint in the paper:** AR50 = 99.0 for essentially every published model.
- **Decision D10:** don't renumber ids. Headline metrics use the `paper` protocol, comparable with 93.6. `fixed` (ids shifted) is reported as a sensitivity check. The two are identical on Val400, which has no id 0.
- Relayed to the repro worker (no id renumbering, plus an assertion) and to the report worker (evaluator section + comparable-conditions box).

### 2026-11-05 20:12 EST — Literature scan lands (D11, D12)
- **No 2021–2026 paper beats 93.6** on SyRIP Test100 with GT boxes (61 citing papers screened, 4 unreadable).
  - Gama et al. 2025 benchmark zero-shot models on the 500-image set with detector boxes (ViTPose-H AP 41.9). Not comparable.
  - AggPose and InfantNet use their own datasets.
- **Headroom analysis:** published AP50 = AP75 = 98.5, so the mean AP at OKS 0.80–0.95 is ≈ 86.3 for FiDIP vs ≈ 84.0 for plain fine-tuning. FiDIP's gain was precise localization, and ours must be too. This motivated the resolution arm and the strict-threshold breakdown in the evaluator.
- **Paper vs released config:** the paper says Adam lr 1e-3, 100 epochs, batch 64, λ = 5e-4. The yaml has lr 1e-4, 20 epochs, batch 20, λ = 0. Added an attempt to test this (D12).
- **MMPose facts:**
  - Exact checkpoint URLs (ViTPose-L 53609f55, ViTPose-H e32adcd4, HRNet-W48 DARK 384×288 39c3c381).
  - Install stack: py3.10, torch 2.1.2 cu121, mmcv 2.1.0, mmengine 0.10, mmpose 1.3.2, mmpretrain, numpy < 2; `chumpy` needs `--no-build-isolation`.
  - ViTPose at 384×288 via codec/backbone sizes, with automatic pos-embed resize.
- **Forwarded:**
  - improve: URLs, install, recipe, resolution, 5 seeds, ensemble;
  - repro: paper-recipe attempt R019;
  - analysis: per-OKS-threshold AP, `ensemble_preds.py`, Holm correction, never report Test500 as clean;
  - report: related work with sources, the recipe discrepancy, Expanded SyRIP ≈ our Val400, so out of scope.
- **Disagreement with the scan:** it suggested 5-fold CV on the 200 real training images for selection. We keep Val400 (validate500 \ validate100; disjoint from Test100, training data unchanged). It gives a 400-image real selection set without shrinking training data. The cost is that Test500 can't be reported as clean, which we note.

### 2026-11-05 20:15 EST — Human: "Don't mention Co-Authored by Claude"
- **Human asked:** no Claude co-author mentions.
- **Action:**
  - Told all 4 workers to stop adding the `Co-Authored-By` trailer. The report worker was also told the report/README author is the student only.
  - Rewrote the 10 local, unpushed commits to drop the trailer, using `git commit-tree` + a compare-and-swap `update-ref`. That approach doesn't touch the working tree or index, so workers' uncommitted files were safe. Trees, authors and dates are unchanged; only the SHAs changed.
  - Added the rule to project `CLAUDE.md`.
- **Old → new SHAs:**
  - 91a40f8 → cf33818 (upstream import)
  - 2739f4d → da804a0
  - 4f20279 → fdaaa82
  - 18ab6c5 → c4cf349
  - 8842520 → 27aba42
  - 8871e1f → 9d18530
  - ddfedab → 724b106
  - f14baa0 → d15feac
  - 10d33a7 → 414a911
  - 87f5689 → e8373c4
- Workers now also record the tree SHA (`HEAD^{tree}`) in `run.json`, because tree SHAs survive message rewrites.
- **Self-correction:** the preserved commit times showed that four of my AGENT.md headers and decision rows (and the SKILL.md snapshot) had **estimated** times (20:25/20:30/20:40) where the real times were 20:08–20:13. I corrected them to match the git commit times and added a rule to `CLAUDE.md`: always take timestamps from `date`, never estimate.

### 2026-11-05 20:19 EST — Unified evaluator v1.0 ready; cross-framework metric trap
- The analysis worker committed `eval/eval_coco.py` v1.0 (`c6eb701`, 24 pytest tests pass).
- **Outputs:**
  - `metrics`: paper protocol, rescored. This is the headline.
  - `metrics_fixed`: id-0 artifact removed.
  - `metrics_original_scores`.
  - Per-OKS-threshold AP, and `AP_80_95`.
  - Per-keypoint error / √(w·h) and PCK@.05/.1/.2.
  - Per-image OKS.
- **Finding:** MMPose 1.3.2's `CocoMetric` uses **xtcocotools 1.14.3**. It initialises `dtm = -1` and tests `dtm >= 0` (cocoapi PR #332), so it **does not** have the id-0 artifact. pycocotools 2.0.8/2.0.11 still does.
  - So MMPose-printed Test100 AP equals our `fixed` protocol, about 1 AP above paper-comparable numbers for identical predictions.
  - Comparing MMPose's own printout with 93.6 would have silently inflated our improvement by about 1 AP.
  - This confirms D2: only `eval_coco.py` numbers are compared across frameworks. Relayed to the improve worker.
  - Val400 is unaffected because it contains no id 0.
- **Test100 ceiling under the paper protocol** (unit-tested):
  - AR ≤ 0.99.
  - A perfect model gets per-threshold AP of 0.980–0.990, depending on how test0's score ranks.
  - The published AP50 = AP75 = 98.5 therefore means FiDIP is already perfect at OKS 0.5/0.75 on the 99 matchable images. All headroom is at OKS ≥ 0.8.

### 2026-11-05 20:47 EST — Analysis pipeline gated and armed; data-integrity check requested
- **Analysis worker committed `ac8e359`:**
  - `scripts/slurm/analysis.sbatch eval-all` re-evaluates every `$R/runs/*/preds_<split>.json`.
  - It has a **unit-tested gate**: Test100 and validate500 (which contains Test100) are evaluated only for released, retrain and pre-registered RUN_IDs. All others are withheld, never computed. This enforces D1 in code rather than by convention.
  - `scripts/analysis/run_released.sh` builds the full figure set and CIs as soon as R001's predictions exist.
- **Request relayed to repro:** add `preds_validate500.json` for R001 (released checkpoint, validate500's own boxes) to separate the crop and metric effects of the v100/v500 box difference.
- **Possible data tampering:** the repro log notes that `raw/SyRIP/SyRIP` was "flattened" at 20:08 by an unknown actor. The analysis worker says it only reads there.
  - Asked repro to verify integrity against `unzip -l` and member md5s before SPLITS_READY.
  - Asked improve whether it touched the directory, and made `$R/data` read-only for it. The repro worker owns the data; other layouts go in symlink trees.

### 2026-11-05 20:52 EST — id-0 artifact verified on real Test100; claims must hold under both protocols (D13)
- **Analysis job 11659551:** over 12 noisy-prediction cases, our `paper` protocol is bit-identical to raw pycocotools 2.0.11, and `fixed` is bit-identical to raw xtcocotools 1.14.3 (MMPose).
- **Perfect predictions:** `paper` AP is 98.02 (test0 ranked first) to 99.01 (ranked last). `fixed` is 100.
- **Realistic predictions** (AP ≈ 93.6): the two protocols differ by **0.8–2.1 AP**, depending on test0's rank.
- **Decision D13:** improvement claims must survive both protocols. We report test0's score rank per model, and the report gets a figure explaining the sensitivity.

### 2026-11-05 20:53 EST — Shared QOS caps → slot policy (D14); improve-stream environment notes
- **The improve worker hit `QOSMaxJobsPerUserLimit`:** gpu allows 4 running / 8 submitted, sharing 2 / 4, shared by all streams under one user. The H200/A100 queue on `gpu` is long, while 44–48 GB L40S/L40/A6000 cards on `sharing` are free.
- **Decision D14 (slot policy):** written into SKILL.md and sent to all workers. The R010–R012 seeds should run concurrently on one GPU where memory allows.
- **Environment facts from the improve worker:**
  - mmcv 2.1.0's CUDA ops have no sm_90 kernels, so MMPose runs go on A100/L40S/A6000.
  - mmpose 1.3.2 imports mmdet unconditionally (`rtmo_head.py`), so mmdet must be installed even with GT boxes.
  - mmpretrain's pos-embed auto-resize fails for ViTPose's non-square 16×12 grid, so the resize is done offline (committed script + SHA256 requested).
- **Data-integrity follow-up:** the improve worker never touched `$R/data/raw`. Its only write under `$R/data` was a temporary `improve_splits` dir (20:46), since moved to `$R/mmpose_data/splits`. The 20:08 'flatten' of `raw/SyRIP/SyRIP` is still unexplained. It predates both other streams' data writes and most likely came from the repro worker's own unzip job; verification was requested.

### 2026-11-05 21:07 EST — Splits final; six dataset-hygiene findings (D15–D17)
- **Repro worker finished data prep at 21:02:55** (`$R/data/SPLITS_READY`, `$R/data/splits/VERIFY.txt`). Assertions passed: area = w×h, ids as shipped, Test100 ⊂ validate500, Val400 ∩ Test100 = ∅ by id/name/md5. Full SHA256s are in MANIFEST §3.
- **Findings:**
  1. Three Val400 images duplicate Test100 images and were removed (D15).
  2. **The official split leaks**: 4 Test100 images are byte-identical to training images, and 1 is a near-copy (D16).
  3. syrip_for_train.zip's train JSON puts MINI-RGBD labels on SyRIP's synthetic images (D17).
  4. Its `validate_infant/test11.jpg` is a copy of `test24.jpg`.
  5. 300 unannotated real images sit in SyRIP.zip's train folder, 199 of them identical to validate500 images. Only JSON-listed images are used.
  6. `train_pre_infant` is shipped after all (2029 + 2028), so D7's substitute isn't needed.
  - Also: `test489.jpg` is PNG bytes (decodes fine).
- **Raw data integrity:** all 7666 zip members are verified (name/size/CRC32 plus 120 md5 samples).
  - The 20:08 'flatten' of `raw/SyRIP/SyRIP` is **not explained**. The repro worker's first job started at 20:10:26. The improve and analysis workers deny touching it, and the orchestrator's own unzip had been killed earlier. The full verification shows no content change, so the impact is nil.
- **Checkpoints:** downloaded from Drive without quota problems; all strict-load clean. Env `$R/envs/fidip` is ready.
- **Environment trap:** `~/.local/lib/python3.12` leaked into the env, so `PYTHONNOUSERSITE=1` is now required (SKILL.md updated; improve and analysis told).
- **Relays:**
  - improve: rebuild its pre-21:02 derived split copies, and use only `train_infant.json`;
  - analysis: Test100-clean subset, the cost of the test11 defect, whether leaked images score higher;
  - report: all six hygiene findings for §3/§7.
