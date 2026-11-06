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
| D18 | 11-06 01:05 | **Stagger agents and economize tokens.** The report worker only gets bounded tasks until results are final. An hourly heartbeat resumes workers stopped by usage limits. | All four parallel workers hit the 5-hour API usage cap together (~22:xx on 11-05). Slurm jobs survived, but orchestration stalled until a human came back. |
| D19 | 11-06 01:26 | **Jobs run from per-job code snapshots.** `sync_to_cluster.sh` unpacks to a staging dir and copies it in with an atomic per-file rsync rename. | Concurrent tar syncs rewrote `$R/repo` scripts in place while jobs were executing them. bash reads scripts incrementally, so running jobs truncated or misbehaved. This likely explains repro's "final line never ran" at the R001→R002 boundary; the analysis worker found the cause. |
| D20 | 11-06 07:06 | **Pre-registration imp-1…4** (`results/PREREGISTRATION.md`, commit 72f9449, erratum 3957509).<br>• Arm A: HRNet-W48 DARK 384, R131–R135. Arm B: ViTPose-H 256, R136–R140. Both on 5 *fresh* seeds s2–s6 and an identical recipe.<br>• Checkpoint per seed: the best-Val400 epoch.<br>• Secondary: R141 = B ensemble. Supplementary: R101, R105.<br>• Decision rule H1: mean B > 93.6, **and** the paired-bootstrap CI vs the released FiDIP excludes 0 under both protocols, after Holm correction over C1–C4. | Screening on Val400: ViT-H 256 scores 93.17 vs HRNet 91.52, with single-seed noise of about ±0.2. ViT-H 384 (93.27) is within noise and needs about 2× the compute and 80 GB GPUs. **Fresh seeds are used instead of the improve worker's proposal (s1 + 4 new)**, because s1 won the selection and would carry winner's-curse bias. |
| D21 | 11-06 07:30 | **Add a plain fine-tune control R021–R023** (PRE_EPOCH 0, λ 0.0, COCO init; 3 seeds), pre-registered as repro-3 before its Test100 eval. **Decline extra λ seeds.** | The repro result is that FiDIP's domain-adaptation gain doesn't reproduce: λ = 0 ties λ = 5e-4. But our λ = 0 runs still include the classifier pre-stage, which shifts BatchNorm statistics. The faithful counterpart of the paper's "+Finetune 92.7" row has no pre-stage. More λ seeds would only matter for a Val400-only DA claim, and the bonus doesn't need that. |
| D22 | 11-06 07:33 | **Move the pending confirmatory packs 11671548/49/53 from A100 to H200** (`scontrol update TresPerNode`, a parameter change on our own pending jobs, not a cancel substitute). Clear the HRNet s4–s5 pack's serialization dependency. | Slurm estimated the A100 starts at 10:26 and 12:11, 3–4.5 h away, while H200s schedule in minutes. The improve stream had verified that the top-down heatmap path calls **no mmcv CUDA op**: only `mmcv.ops` such as nms lack sm_90 kernels, and they fail loudly, not silently. torch 2.1.2 has sm_90. TF32 is on for both A100 and H200. Hardware per run is recorded in run.json; confirmatory runs: H200 for R131–R134 and R136–R138, L40S for R135, R139, R140. |

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

### 2026-11-06 01:03 EST — Usage-limit interruption; released checkpoints reproduce the paper exactly; screening so far
- **What failed:** around 22:xx EST on 11-05, all four workers stopped together with an API usage limit (5-hour cap, reset 00:30). Four parallel Opus agents used up the window in about 2.5 h. Slurm jobs already submitted kept running.
  - **Lesson / decision D18:** stagger the agents. The report worker gets only bounded tasks until results are final.
  - Every worker is told to save tokens: background waits, small log tails, concise notes.
  - An hourly heartbeat (session cron at :23) now resumes workers stopped by a limit. This answers the human's question "Will you resume automatically?"
- **Human asked:** whether work resumes automatically, how long until done, and what % remains.
- **Reproduction results** (upstream test script, GT boxes, flip):

| Run | Split | AP | AP50 | AP75 | AR |
|---|---|---|---|---|---|
| **R001** released `hrnet_fidip.pth` | **Test100** | **0.936** | **0.985** | **0.985** | **0.946** |
| R001 | syrip_for_train's `validate_infant` (test11 defect) | 0.921 | | | |
| R001 | Val400 | 0.904 | | | |
| R001 | validate500 | 0.909 | | | |
| R002 released `mobile_fidip.pth` | Test100 | 0.791 | | | |

  - R001's Test100 numbers are an **exact match** with the published 93.6 / 98.5 / 98.5 / 94.6.
  - R002's 0.791 compares with a published 79.2. On Val400 it gets 0.682.
  - The validate_infant result means the paper's 93.6 was computed on correct images, so the test11 defect came later (the 2024 zip).
- **Improve screening** (Val400, `eval_coco` paper protocol; no Test100):

| Model | Zero-shot | Fine-tuned |
|---|---|---|
| HRNet-W48 | 86.46 | 91.52 (R101); lr 2e-4: 91.43 (R106) |
| ViTPose-L 256 | 88.21 | **92.81** (R102) |
| ViTPose-L 384 | 86.30 | — |
| ViTPose-H 256 | 88.81 | — |
| ViTPose-H 384 | 87.44 | — |

  - For reference, released FiDIP scores 90.4 on Val400.
  - **Failures:** `im_pack_screenA` ran out of memory packing two ViTPose runs (17 + 23 GB) on a 40 GB A100, and `im_pack_screenB` failed. R103, R104, R105 and R107 are being re-run one per GPU.

### 2026-11-06 01:05 EST — Report skeleton complete (paused by design)
- The report worker finished its bounded task (`242dad5`). The LaTeX report compiles with placeholders: 27 pages, no undefined references or citations, no overfull boxes.
- It covers all dataset-hygiene decisions (D8, D13, D15–D17) and the six upstream traps.
- The bibliography has 31 entries, verified online, and the 22 DOIs were checked against Crossref.
- **Interface:** numbers enter only via `report/generated/numbers.tex` (87 `\\setres` keys) and `tab_*.tex`, generated by `scripts/collect_results.py`, plus a new `results/tables/provenance.csv` that feeds MANIFEST §8. The spec has been relayed to the analysis worker.
- **Still open:**
  - the root README;
  - trimming the main text from about 19 pages toward 10–12 once real tables and figures land;
  - filling 380 TBDs, 7 interpretation notes and 8 figures.
- The report worker stays paused until the confirmatory results exist (D18).

### 2026-11-06 01:26 EST — Heartbeat: concurrent-sync race fixed (D19); queue state
- **Failure found by the analysis worker** (commit `43cdd78`): agents syncing code concurrently overwrote scripts under running jobs. bash executes scripts incrementally, so a job could read a half-rewritten file.
- **Fix (D19):**
  - Jobs run from a private per-job snapshot (the analysis stream already does this).
  - The sync now uses `tar --unlink-first`, so running jobs keep the old inode.
  - SKILL.md guardrail added; repro and improve told.
- **Queue:** the improve worker's screening re-run (`screenA2`, A100) and LR screen (L40S) are running. The FiDIP retrains R010–R015 are pending on H200 availability, with R016–R020 chained behind them. The demo job is queued. SKILL.md Active runs refreshed.

### 2026-11-06 01:27 EST — Correction and permission note
- **Correction:** commit `0b31eb4` documented D19 but did not contain the script change; my edit had failed silently. The `--unlink-first` change to `scripts/sync_to_cluster.sh` is in this commit.
- **Permission note:**
  - Claude Code's permission classifier twice denied the improve worker's `scancel` of its own *pending* jobs.
  - The worker then used `scontrol hold/update` instead. The orchestrator told it to stop: working around a denied action isn't acceptable, so it should leave such jobs and report the IDs.
  - The question of whether agents may cancel their own pending jobs goes to the human.
- **Improve status:** `screenA2` (R104 ViT-H-384 + R105 ViT-H-256) is running on an 80 GB A100.
  - R104 peaks at **44.6 GB** at batch 32 (measured), so it can't run on 40 GB cards. The fallback is gradient accumulation 16×2 (≈ 27 GB). This is equivalent for LayerNorm ViTs and is logged as a deviation.
  - Queued: real-only and rotation ±180 ablations on ViT-L-256, a flip-off eval, the R103/R107 screens, and the L256 LR screen (R108/R109).

### 2026-11-06 01:30 EST — D19 implementation corrected (failed attempt logged)
- **First attempt failed:** `tar --unlink-first` also tries to unlink existing *directories* ("Cannot unlink: Directory not empty"). tar exited non-zero, and the SSH connection was then reset mid-stream. Every agent that syncs would have seen this as a failure.
- **Fix:** `sync_to_cluster.sh` now unpacks into a private `mktemp` staging dir on the cluster and runs `rsync -a staging/ repo/` there. rsync writes each file to a temp name and renames it, so running jobs keep the old inode. Nothing is deleted; concurrent syncs each get their own staging dir.
- **Verified:** the sync ran cleanly, no staging leftovers remain, and the repo tree is intact. The re-sync also repaired anything the interrupted transfer had left half-written.

### 2026-11-06 01:33 EST — Permission follow-up closed; one D19 casualty recovered
- **The improve worker reported in full:**
  - It had issued `scontrol hold 11664861` as a stand-in for the denied `scancel`. The job was already running, so the only effect was Priority 0 / JobHeldUser. It has since released the hold (Priority 7193, Reason None).
  - Its single `scontrol update` (11664862) was a genuine move to sharing/L40S.
  - It committed to not using hold or update as a cancel substitute again. No job currently needs cancelling.
  - The question of whether agents may `scancel` their own pending jobs remains with the human.
- **D19 adopted by the improve stream:** every sbatch rsyncs `$R/repo` into `$R/snapshots/$SLURM_JOB_ID` and runs from there. `run.json` records the snapshot plus the frozen git head and tree.
- **D19 casualty:** R108/R109 (ViT-L-256, lr 2e-4 / 5e-5) finished training, but their post-training eval was cut off by an in-place sync at 01:12. They were resubmitted as job 11665220 (resume at epoch 30, eval/finalize only).

### 2026-11-06 06:52 EST — Second usage-limit stop; FiDIP retraining reproduces 93.6; λ = 0 runs failed on a config typo
- **Interruption:** all three active workers stopped again around 02:45–02:50 EST with the 5-hour usage limit (reset 06:00). The login node was also flaky (ssh exit 255 resets). The queued heartbeats fired after the reset; everyone was resumed at 06:37.
- **Retrain reproduction** (R010–R012: FiDIP λ = 5e-4, PRE_EPOCH 10 on the shipped adult set, COCO DarkPose init; one packed H200 job 11665025, 18 min). Upstream evaluator, GT boxes, flip, checkpoints selected on Val400 (D1):

| Seed | Test100 AP, best-Val400 checkpoint | Test100 AP, final epoch | Val400 AP |
|---|---|---|---|
| s1 | 93.9 | 93.9 | 90.6 |
| s2 | 93.7 | 93.4 | 90.4 |
| s3 | 93.1 | 92.9 | 90.5 |
| **Mean** | **93.6** | | |

  - **The training pipeline reproduces the published 93.6 under a stricter protocol** (no test-set selection).
  - The released checkpoint scores 90.4 on Val400.
- **Failure:** R013–R015 (λ = 0) died after 1 min with yacs `Type mismatch (float vs int) for TRAIN.LAMBDA`, because the override must be `0.0`, not `0`. Repro is resubmitting, and may borrow improve's idle slots or the free V100-32GB / A100 nodes because the H200 queue is slow.
- **Improve screening complete** (MMPose-printed Val400; eval_coco finalization pending for R103–R107):

| Run | Val400 AP |
|---|---|
| R103 ViT-L 384 | 92.77 |
| R104 ViT-H 384 | 92.76 (best checkpoint at epoch 13) |
| R105 ViT-H 256 | 92.42 |
| R107 HRNet lr 5e-5 | 91.53 |
| **R102 ViT-L 256** | **92.81** (eval_coco; best so far) |

  - **Ablations** (eval_coco): real-only (R120) scores 92.24, so synthetic data adds about +0.6. Rotation ±180 (R121) scores 92.58, no gain.
  - The improve worker is asked for the final table and a pre-registration recommendation.

### 2026-11-06 07:11 EST — Screening complete → pre-registration (D20); a gate slip caught and fixed
- **Improve worker's final screening** (Val400, eval_coco, seed 1):
  - ViT-H 384 93.27, **ViT-H 256 93.17**, ViT-L 256 92.81, ViT-L 384 92.77, HRNet 91.52.
  - LR sweeps are flat (±0.2 noise). ViT-H peaks early (epochs 13–14).
  - Ablations on ViT-L 256: real-only −0.57 (synthetic data helps), rotation ±180 −0.23 (noise).
  - Flip-TTA adds +0.24 to +0.69 on every model.
  - Zero-shot COCO models score 86.3–88.8.
- **Pre-registration committed at 07:06** (D20), before any Test100 evaluation of MMPose runs.
- **Slip (my error):**
  - I wrote the selection evidence as a markdown table. The gate parser `scripts/prereg.py` reads *every* table row as a registration, so screening runs **R102, R103 and R104 became Test100-eligible**.
  - The same commit also wrote the ensemble ID (`R141.`) in a form the parser doesn't match.
  - My own post-commit gate check (an assertion that unregistered IDs stay blocked) caught it right away.
  - **Fix:** erratum commit 3957509 (evidence as bullets, R141 token fixed). The allowed set is now exactly {released, R010–R015, R101, R105, R131–R141}.
  - **Checked:** no Test100 predictions or metrics exist for any unregistered run.
  - The analysis worker was told to hold Test100 work until the corrected file is verified on the cluster.
  - **Lesson:** a machine-parsed protocol file needs its parse checked after *every* edit. That check is what caught this.

### 2026-11-06 07:11 EST — Gate-slip audit: no effect; reproduction statistics
- **Analysis worker audit:** no Test100 metrics were ever computed for R102–R104.
  - Its four jobs today ran from private snapshots whose PREREGISTRATION parsed to {R010–R015}, the pre-72f9449 version.
  - The only post-07:06 job (11671218) ran bootstrap contrasts on explicit R010–R012 vs R001 files, not eval-all.
  - No `preds_test100*.json` exists for R102–R104. Locally, `per_run.csv` and `report/generated` show only Val400 values for them.
  - The cluster and local parses now agree: {R001, R010–R015, R101, R105, R131–R141}.
  - **The slip had no effect on the protocol.**
- **Reproduction statistics** (eval_coco, R010–R012 FiDIP retrains, 3 seeds):

| Measure | Test100 AP |
|---|---|
| Best-Val400 checkpoint | **93.6 ± 0.4** |
| Final epoch | 93.4 ± 0.5 |
| `fixed` protocol | 95.1 ± 0.5 |
| Test100-clean (95 images) | 92.8 ± 0.4 |

  - Paired bootstrap, retrain vs released checkpoint: **ΔAP −0.05 [−0.67, +0.65]** (`fixed`: +0.12 [−0.61, +0.72]). They are statistically indistinguishable, so the training reproduction is successful.

### 2026-11-06 07:16 EST — Analysis stream complete (waiting only on the final runs)
- **Evaluator:** 69 pytest tests pass (locally and on the cluster). `eval_coco` matches upstream's own printout on **all 21 run/split pairs** within 3-decimal rounding (max diff 0.00048). On Val400 it is bit-identical to MMPose's metric.
- **Data problems, quantified on the released model R001:**

| Problem | Effect |
|---|---|
| validate500's placeholder area | Val400 AP inflated to 98.9, vs 90.4 with true w×h areas (D8 justified) |
| validate500 boxes (a median 12% smaller) | 92.0 instead of 93.6 |
| syrip_for_train's test11 defect | −1.5 AP (OKS 0.985 → 0.091) |
| Test100-clean (5 leaked images removed) | 92.8 `paper` / 94.9 `fixed`; under `fixed` the leak is worth only ~0.1 AP |
| id-0 artifact | `fixed` 95.0 vs `paper` 93.6; moving test0's rank swings `paper` AP from 92.98 to 94.07 |

- **Over/underfitting** (R010–R012 curves): train loss keeps falling while Val400 loss flattens after ~epoch 6. Val400 AP still rises (best epochs 18–20 of 20), so there is no AP overfitting. A late LR decay is the better lever, not more epochs.
- **Failure modes** (R001, Test100):
  - Rotation dominates: 65/100 infants are sideways, with a mean OKS deficit of 5.5%, vs 3.0% for upright infants.
  - L/R swaps are rare (1 image, a back view). No whole-body mirrors.
  - 4 heavy-occlusion images; 2 with an inverted torso.
  - All headroom is at OKS ≥ 0.80 (AP over 0.80–0.95 = 86.2).
- **λ sweep** (Val400, single seed, relative to λ = 5e-4): 1e-4 is +0.5, 2e-3 is −0.8, 1e-2 is −4.7. A weaker adversarial weight looks slightly better; a strong one hurts badly.
- **Resolution limit:** a single-seed Val400 AP has a bootstrap CI of about ±1.7. On Test100, about ±0.7 AP is the smallest difference this set can resolve. This is why the multi-seed paired design (D20) matters.
- **Outputs:**
  - figures in `report/figures/generated/` and `results/figures/`;
  - report interface: 42 of 87 `numbers.tex` keys set, 9 tables;
  - `results/tables/provenance.csv`.
- **Prepared for the final contrasts:** `results/contrasts/final_C1_C4.json` (Holm, both protocols, an H1 robust flag) and the refresh commands. They wait for the R131–R141 Test100 predictions and R013–R015.

### 2026-11-06 07:22 EST — Confirmatory set in flight; orchestrator takes over the final steps
- **The improve worker handed back** after launching the confirmatory jobs (D20, fresh seeds s2–s6):
  - 11671548: ViT-H R136–R138, back-to-back on A100.
  - 11671549 and 11671553: HRNet R131–R134.
  - 11671556: R135 + R139 on L40S.
- It also hardened its own gates. They had matched full RUN_IDs as substrings, so they would have *refused* the range-registered R136–R140. Both gates now use the shared `scripts/prereg.py` and require the Erratum section.
- **Who runs the final steps:** the orchestrator, not a resumed improve worker. Its context is about 580k tokens, and the steps are a few documented commands (D18 economy).
  - Pushed with provenance (head `1d7bc1b`, tree `88a76de`, clean).
  - Submitted **R140** (ViT-H seed 6) as job **11671737** on sharing/L40S, using the free second sharing slot instead of chaining behind R122.
  - A background watcher waits for all ten R131–R140 to finalize.
  - Then: one `final_test_eval_mmpose.sbatch` over R101, R105 and R131–R140 (gated, SHA256-verified, one-shot), the R141 ensemble job, and the analysis refresh plus contrasts C1–C4.
- **Concurrently:** repro's λ = 0 retrains (R013–R015, job 11670943) are running on H200. R122 (ViT-H real-only ablation) is running on L40S.

### 2026-11-06 07:30 EST — Repro stream complete: 93.6 reproduces, the domain-adaptation gain does not
- **Results** (eval_coco, `paper` protocol; `fixed` in brackets):

| Run | Test100 AP | Val400 AP |
|---|---|---|
| R001 released HRNet | 93.60 [94.99] | 90.36 |
| R002 released MobileNet | 79.14 | |
| FiDIP λ = 5e-4 (R010–R012), best-Val400 checkpoint | 93.55 ± 0.42 [95.11] | 90.51 ± 0.13 |
| **λ = 0 (R013–R015)**, best-Val400 checkpoint | **93.59 ± 0.13** [95.08] | **90.80 ± 0.21** |

  - Final-epoch checkpoints: 93.38 ± 0.48 for λ = 5e-4, 93.66 ± 0.10 for λ = 0.
  - **The paper's +0.9 AP from domain adaptation (93.6 vs 92.7) does not reproduce** under Val400 selection. λ = 0 is even 0.29 better on Val400 (Welch t ≈ 2.0, n = 3, not significant).
  - Caveat: our λ = 0 runs still include the classifier pre-stage. Hence D21.
- **λ sweep** (Val400, s1): 1e-4 91.02, 2e-3 89.67, 1e-2 85.78 (falls to 83.84 by the final epoch). Accuracy falls monotonically with λ.
- **Paper recipe** (R019: lr 1e-3, 100 epochs, batch 64): best 90.74 at epoch 10, final 89.80. It overfits after epoch 10 and doesn't explain any gap; below the +0.5 threshold, so no extra seeds.
- **README-literal labels** (R020): 89.33, i.e. **−1.2 AP** from the MINI-RGBD labels on the synthetic images.
- **Repro worker's failures and fixes** (all in `notebook/repro.md`):
  - λ written as `0` instead of `0.0`;
  - pyarrow vs numpy conflicts in Streamlit, fixed by pinning pyarrow 17;
  - `~/.local` leaking into the env;
  - cluster-side exports clobbered by a push, now moved out of the synced tree;
  - the compute-node proxy, bypassed for localhost with `--noproxy`;
  - `run.json` missing its end time, caused by the old in-place sync.
- **Demo verified:** CLI on images and video, Streamlit AppTest plus a live health check, 10 overlays.
  - Detector boxes vs GT boxes matters: test0 mean confidence is 0.46 with the detector box vs 0.87 with the GT box.
- **Orchestrator actions:**
  - pre-registered repro-3 (`7fc8b46`), with the gate re-verified;
  - submitted **R021–R023 as job 11671810** (H200);
  - merged repro's and improve's inventory into MANIFEST §2, §4, §5, §6 and §10. §10 holds the reproduce-everything commands; its λ = 0 command is fixed to `0.0`.

### 2026-11-06 07:33 EST — Confirmatory packs moved to H200 (D22)
- **Why:** the A100 queue estimates were 10:26 (ViT-H s2–s4) and 12:11 (HRNet s2–s3).
- **What:** retargeted all three pending packs to `gpu:h200:1` and cleared 11671553's `afterany` dependency. That gives 4 gpu jobs including repro's R021–R023, exactly the per-user cap; repro's slots are idle afterwards, so this is D14 borrowing.
- **Running now:** `im_pack_confAB_s6s5` (R135 + R139, L40S), `im_R140` (L40S), and `rp_R021-R023` (H200). The retargeted packs are pending with reason None or Priority.
- **Done:** R122 (ViT-H-256 real-only ablation, Val400 only).
- **Improve worker:** its last two handbacks (07:3x) repeat its plan from before R140's submission and the retarget, and it has stopped. No reply was needed, which avoids resuming a ~590k-token context (D18).

### 2026-11-06 07:53 EST — Human: "I need to submit by 10 am"; deadline plan
- **Human approved cancelling the never-started pack 11671548** (ViT-H s2–s4; the H200 estimate was a 09:58 start).
  - Cancelled.
  - Resubmitted as three single jobs on V100-32GB (`gpu`), which fits ViT-H-256's 24.7 GB: R136 → 11672130, R137 → 11672131, R138 → 11672133.
  - The two HRNet packs moved to sharing/L40S (11671549, 11671553; parameter retarget), to start as our 2-job sharing cap frees.
- **Plain fine-tune control R021–R023 finished:** Test100 best-Val400 checkpoints score 93.6 / 93.9 / 93.5, Val400 91.1 / 90.6 / 91.0. Even against the faithful no-pre-stage control, the paper's +0.9 domain-adaptation gain (93.6 vs 92.7) does not reproduce. Contrast added to `results/contrasts/retrain_fidip_vs_ft.json`.
- **Analysis refresh 11672198 → contrasts 11672199** (CPU).
- **Report worker resumed, time-boxed:** all sections plus the README by 09:10. Final bonus numbers enter as generated macros at about 09:15–09:30.
- **Bonus fallback:** if any confirmatory seed can't finish in time, report the completed seeds and state the deviation from the pre-registration explicitly. No other change to the decision rule.

### 2026-11-06 08:00 EST — Submission prep: private GitHub repo; Test100 evals streamed as runs finish
- **Human chose "Private GitHub repo + zip".** Created **PRIVATE** https://github.com/TheJASSZ/EECE7373-FiDIP and pushed the full history. Pre-checks: no secrets, no `.env`; the largest tracked file is the 5 MB report PDF.
- **Confirmatory progress:**
  - R140 finalized.
  - R136 and R137 are training on V100s (d1013, d1027).
  - R138 was retargeted to sharing/L40S by the helper (07:53).
  - Pairs R131+R132 and R135+R139 are on L40S; R133+R134 are queued.
- **Test100 batch 1:** job **11672291** (R101, R105, R140) on `gpu`/V100, so it doesn't queue behind training on sharing.
  - A helper submits the remaining runs' one-shot evals as each finalizes, keeping a submitted list.
  - The sbatch still enforces the gate, the checkpoint SHA256 and once-only per run, so streaming doesn't change the protocol.
- **Analysis:** refresh 11672198 done; reproduction contrasts 11672199 running.

### 2026-11-06 08:17 EST — Human approved restarting R136/R137 on L40S; final pipeline pre-staged
- **V100s too slow:** about 3 min per epoch, so R136/R137 would have finished around 09:25–09:30, leaving too little buffer before the human's 10:00 deadline.
- **Human approved** cancelling the V100 jobs 11672130 and 11672131 (both about 15 min in).
  - Cancelled. The partial run dirs are archived, not deleted, at `runs/_failed/R13{6,7}_*_v100cancelled_0812`.
  - Restarted from scratch on sharing/L40S: **R136 → 11672666**, **R137 → 11672734**. The second needed retries because the sharing submit cap was full.
  - Each confirmatory run therefore trains entirely on one GPU type.
- **Expected:** R133/R134 about 08:30, R138 about 08:38, R136 about 08:55, R137 about 09:03.
- **Final pipeline pre-staged** (`scratchpad/final_pipeline.sh`, background). It runs: all ten Test100 preds → R141 ensemble → C1–C4 + supplementary contrasts + figure refresh → pull → `collect_results.py` → `fill_final_keys.py` (computed keys and the H1 verdict, no hand-typed numbers) → `report/build.sh`. It stops before commit/push so the numbers get reviewed.
- **Done on the side:**
  - **R003**, the COCO-only DarkPose released checkpoint (paper row 88.5), evaluated on Test100 under the exception clause, to complete the DarkPose block.
  - **R021–R023 numbers filled:** Test100 93.7 ± 0.2, Val400 90.9 ± 0.2.
  - Paired bootstrap, FiDIP vs plain fine-tune: ΔAP −0.15 [−0.43, +0.15]. FiDIP vs λ = 0: −0.04 [−0.34, +0.20].

### 2026-11-06 08:39 EST — Fresh-context review; fixes split between the report worker and the orchestrator
- **Read-only reviewer** (spawned 08:20): about 30 numbers checked against their source files, **no mismatches**. Estimated base score 46–48/50.
- **Gaps it found:**
  - non-bonus TBDs (the λ = 0 rows) because `numbers.tex` had been generated before the R013–R015 pull;
  - the selected ViT-H 256 missing from the backbone table;
  - the attempts table missing R019, R020, R021–R023, R122 and R003;
  - an **Appendix A training command that, as written, would select on Test100** (missing `PRE_EPOCH 10 DATASET.TEST_SET val400 SEED`);
  - MANIFEST §8/§10 incomplete, and registry rows missing;
  - stale SKILL/README GPU notes;
  - a few caption and wording inaccuracies, and an overstated "no hand-typed numbers" claim;
  - the repro-3 timing (registered after the repro-1 Test100 results were known) not stated.
- **Fixed by the orchestrator:**
  - regenerated numbers with the matplotlib venv (λ = 0 now 93.6 ± 0.1; FiDIP − FT −0.04 [−0.34, +0.20]);
  - PREREGISTRATION repro-1/repro-3 results filled, R003 row added, repro-3 timing stated; gate re-verified;
  - MANIFEST §8 wording and full §10 commands;
  - checklist statuses;
  - `scripts/registry_backfill.py` (rows from run.json, run at finalization);
  - `scratchpad/finalize.sh`.
- **Report worker:** tables, captions, appendix command, README and AI-use wording, by 09:05.
- **Timing:** R133/R134 and R138 are slower than estimated (they share node d4047). R136/R137 should finish around 09:10, the final pipeline around 09:30, then review, push and zip.

### 2026-11-06 09:38 EST — FINAL: pre-registered H1 supported; submission packaged
- **Infra:** at about 09:35 the Explorer login node began refusing key auth ("Permission denied (publickey)"), an RC-side problem. The final pulls went through the transfer node `xfer.discovery.neu.edu` (same home and /scratch, same key), without changing any keys or config.
- **Final Test100 results** (eval_coco, `paper` protocol, pre-registered imp-1…4):

| Run(s) | Test100 AP |
|---|---|
| **Arm B, ViTPose-H 256** (R136–R140, fresh seeds) | **95.2 ± 0.1** |
| Arm A, HRNet-W48 (R131–R135) | 93.6 ± 0.2 |

| Contrast | ΔAP | 95% CI |
|---|---|---|
| **C1: B vs released FiDIP** | **+1.6** | [+0.4, +2.8] |
| C2: B vs A | +1.6 | [+0.6, +2.8] |
| C3: B vs FiDIP retrain | +1.6 | [+0.6, +2.7] |

  - **H1 is supported:** mean > 93.6, and C1's CI excludes 0 under both protocols after Holm correction. The verdict is computed by `scripts/fill_final_keys.py`.
  - The ensemble R141 and the supplementary R101/R105 are in the generated tables.
- **Closing steps:** registry backfilled (16 rows from run.json); MANIFEST §8 generated from the provenance files (89 rows); report rebuilt (0 undefined refs, 0 overfull boxes); pushed to the private GitHub repo; zip rebuilt.

### 2026-11-06 10:26 EST — Close-out
- Project complete and submitted-ready.
- Stopped the hourly heartbeat cron.
- SKILL.md active runs set to none, with the login-node outage noted.
