# notebook/report.md — live log for the report stream

Newest entries at the bottom. Timestamps in EST.

## 2026-11-05 20:04 EST — report stream start
- Read CLAUDE.md, SKILL.md, AGENT.md, MANIFEST.md, results/PREREGISTRATION.md, results/registry.csv, upstream README.
- Scope (from orchestrator): `report/` (report.tex, refs.bib, figures/, generated/ placeholders, PDF build) + repo-root `README.md`.
  Not touching AGENT.md / MANIFEST.md / SKILL.md (requests go to orchestrator in the final message).
- Local toolchain check: no tectonic / pdflatex / latexmk; `pdftoppm` present (/opt/homebrew/bin); brew present; 7.6 GB free.
- Plan: install tectonic via brew (small) -> write generated/*.tex placeholders -> report.tex + TikZ -> refs.bib (verify every entry online) -> README.md -> build + render pages to PNG + visual check.

## 2026-11-05 20:36 EST — sources verified, coordinator messages folded in, interface designed
- Tectonic 0.17.0 installed via brew (~130 MB incl. deps; bundle cache 45 MB). Test build with tikz/tcolorbox/natbib OK.
- Read upstream code myself for the method section (facts used in the report):
  - `DomainClassifier` (lib/models/adaptive_pose_hrnet.py): 2x [3x3 conv-BN-ReLU-maxpool2] (48->32->8 ch) + FC 8*22*16->256->64->1 on stage-4 high-res branch (48x96x72). Paper text says "three fully connected layers" only.
  - `train_adaptive` (lib/core/function.py): step I BCE on `feature_outputs.detach()`, step II `MSE - LAMBDA*BCE`; no GRL module.
  - Freeze: first 109 parameter tensors (`idx <= 108`, comment "stage 2 + bottleneck").
  - `BalancedBatchSampler` (lib/utils/sampler.py): alternates real/syn, oversamples real to 1000 at construction, no reshuffle per epoch.
  - `SYNJointsMSELoss` takes a domain label but ignores it; HRNet trainer uses plain `JointsMSELoss` -> paper's S(I_i) scaling not in code.
  - model_best chosen by `validate_adaptive` on `TEST_SET=validate_infant`.
- Downloaded arXiv 2010.06100v5 PDF to scratchpad, extracted text, deleted PDF. Paper recipe confirmed: Adam lr 1e-3, init session batch 128 x 1 epoch, formal 100 epochs batch 64, "GRL parameter lambda 0.0005", freeze Res1-3 (ablation), Test500/Test100 definitions, mAP over 10 OKS thresholds.
- SyRIP README fetched; sha256 2b7ebd83... matches MANIFEST.
- Bibliography sub-agent (background) returned 29 verified entries (publisher pages + Crossref; dblp blocked). I will re-verify DOIs myself via Crossref before using them.
- Coordinator messages received and to be reflected: STUDENT_GUIDE spine (docs/REQUIREMENTS_CHECKLIST.md), D8 (placeholder areas), D10 (ann-id-0 artifact), literature scan, no co-author trailer / student-only authorship.
- Interface decision: scalar results via `\setres{key}{value}` in `generated/numbers.tex` (keys may contain digits/dots); tables as one `tabular` per file `generated/tab_*.tex`; figures searched in `figures/generated/`, `../results/figures/`, `../results/figures/demo/`. Missing value -> red TBD + LaTeX warning; missing file -> framed placeholder.

## 2026-11-06 01:04 EST — draft skeleton complete, compiles with placeholders; committing
- Report source: `report/report.tex` + `sections/*.tex` (11 files), TikZ `figures/{fidip_arch,workflow,protocol}.tex`, `refs.bib` (31 entries, all verified online; 22 DOIs re-checked by me against Crossref with `report/check_bib.py`; non-DOI items checked on PMLR/JMLR/arXiv pages; MMPose model-zoo COCO APs 78.2/78.8/77.2 checked in the v1.3.2 metafiles), `build.sh` (tectonic; prints undefined keys / TBD / pending files / overfull boxes), `keymap.tex`.
- Interface: `generated/numbers.tex` (`\setres{key}{value}` canonical keys), `generated/tab_*.tex` (one tabular each: reproduction, final, boot, attempts, backbone, ablations, sensitivity, perkeypoint, failures, hparams, compute, patches), figures searched in `figures/generated/`, `../results/figures/`, `../results/figures/demo/`.
- Compatibility with the analysis stream: `report.tex` inputs `generated/results_tables.tex` (collect_results.py) in macros-only mode (its floats are discarded via `environ`, tested), so its `\newcommand` macros can feed `\setres` keys (directly or via `keymap.tex`) without label clashes.
- Folded in coordinator findings: D8, D10/D13 (claims must hold under both protocols; test0 rank), D15 (Val400 = 397), D16 (5 train/test duplicates; Test100-clean 95), D17 (SyRIP.zip labels; test11 copy; R020), unannotated duplicates, train_pre_infant shipped (2029+2028; PRE_EPOCH 10), STUDENT_GUIDE spine, student-only authorship, reflection marked `% STUDENT: please review/personalize`.
- Build: 27 pages with placeholders (main text ~19 pp incl. placeholder floats); 0 undefined keys/refs/cites, 0 overfull boxes; 380 TBD values, 7 interpretation notes, 8 pending figures (expected until results land).
- Not yet done (next session, per coordinator): repo-root README.md; fill results via generated files; replace `\interp` notes; trim main text to ~12 pp; final visual check.
- Pitfalls hit: `\code` (url-based) breaks in captions -> short caption titles; underscores in placeholder descriptions; `step` is a TikZ key; url drops spaces -> `obeyspaces`; `comment` pkg cannot skip floats inside `\IfFileExists` -> `environ`.

## 2026-11-06 08:04 EST — final-results pass (deadline mode)
- Sections 4-6 rewritten with final results via keys: generated numbers.tex (analysis), keymap.tex -> results_tables.tex macros (\setresmac, fail-soft, math-mode minus), and generated/manual_numbers.tex for sourced numbers without a key (31 + 1 rows, each with a source comment; mirrored in results/tables/provenance.csv).
- Pending keys default to TBD via \setresdefault (never override generated values): rt.ftnp.* (R021-R023), armA.test.ap.*, c2.*, c3.*; best.*/ens.*/rt.ft.* stay generated. \finaloutcome is a \newcommand line in manual_numbers.tex for the orchestrator to set.
- Data-hygiene findings condensed into tab:hygiene (Sec 3.3); implementation detail, attempts, screening, sensitivity, failure counts, success cases and the demo figure moved to appendices I-K.
- Main text now 16 pp (was 22); 28 pp total; 0 undefined keys/refs/cites, 0 overfull boxes, 0 interp notes; 142 TBDs = pending final numbers.
- README.md written: overview, sourced results summary, findings table, guide-ordered quickstart pointing to MANIFEST section 10, guide/upstream differences, repo map, docs/integrity.
- Visual check: title page, hygiene table, keybox page, analysis figures OK; fixed a protocol-figure label overlap and figure widths.
