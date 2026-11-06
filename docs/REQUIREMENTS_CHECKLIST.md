# Requirements checklist (grading rubric + STUDENT_GUIDE.md)

The orchestrator and the final reviewer use this list to trace every requirement to where it is satisfied.

Status values:
- ☐ = todo
- ◐ = in progress
- ☑ = done, with a pointer to where

## A. Grading rubric (50 pts + 5 bonus)
| # | Requirement | Pts | Where satisfied | Status |
|---|---|---|---|---|
| A1 | Environment setup and data preparation | 5 | `scripts/get_data.sh`, `scripts/make_splits.py`, `envs/`, report §3 | ◐ |
| A2 | Working inference and evaluation pipeline | 10 | released-ckpt eval R001/R002, `eval/eval_coco.py`, `tools/demo.py` | ◐ |
| A3 | Clear instructions, code, configs to reproduce | 5 | README quickstart, MANIFEST "reproduce" section, `scripts/slurm/` | ◐ |
| A4 | Correct quantitative evaluation | 7 | unified evaluator + unit tests + upstream cross-check | ◐ |
| A5 | Comparison with published baseline | 4 | report §4–5 tables (paper vs released vs ours) | ☐ |
| A6 | Visualizations + analysis of successes and failures | 4 | `results/figures/`, report §6 | ☐ |
| A7 | Clear organization | 5 | report structure, README | ◐ |
| A8 | Methods and implementation choices explained | 5 | report §2–3, §5 | ◐ |
| A9 | Limitations, lessons learned, references, AI-tool use | 5 | report §7–9, verified `refs.bib` | ◐ |
| B1 | Improved primary metric (Test100 AP), same split, comparable eval | ≤3 | report §5 box, PREREGISTRATION | ☐ |
| B2 | Modification explained + controlled comparison / ablation | ≤1 | HRNet-W48 vs ViTPose under one recipe; λ ablation | ☐ |
| B3 | Reproducible evidence: configs, logs, ckpt details, repeats | ≤1 | 3 seeds, MANIFEST, registry, bootstrap CIs | ☐ |
| B4 | Baseline, our result and evaluation conditions clearly identified; no test-set tuning or leakage | — | Val400 protocol (D1), PREREGISTRATION | ◐ |

## B. STUDENT_GUIDE.md requirements
| # | Requirement (guide section) | Where satisfied | Status |
|---|---|---|---|
| G0.1 | Passwordless SSH to Explorer (Part 0.1) | verified 2026-11-05; SKILL.md | ☑ |
| G0.2 | `.env` in `.gitignore` before creation; agent told never to print/commit secrets (Part 0.2) | `.gitignore`, project `CLAUDE.md`, global `~/.claude/CLAUDE.md` (Secrets rule, added 2026-11-05 at the human's request) | ☑ |
| G0.3a | `SKILL.md`: connection facts, directory layout, exact sbatch/squeue/sacct commands, **active runs**, guardrails | `SKILL.md` (keep Active runs current) | ◐ |
| G0.3b | `AGENT.md`: what was asked, what the agent did, experiments, hypotheses, dead ends, direction changes | `AGENT.md` + `notebook/*.md` | ◐ |
| G0.3c | `MANIFEST.md`: datasets, checkpoints + paths on /scratch, configs, run behind the final numbers, **reproducible from this file alone** | `MANIFEST.md` (needs exact reproduce commands) | ◐ |
| G0.4 | Short chained jobs (`--dependency`), periodic checkpoints, agent polls squeue/sacct/logs | sbatch scripts; SKILL.md | ◐ |
| G1 | Env setup; GPU check (`gpu_test.py`); build libs | `envs/`, `gpu_test.py`, patch notes | ◐ |
| G2 | SyRIP download + expected layout + pretrained models | `scripts/get_data.sh`, MANIFEST §3–4 | ◐ |
| G3 | Understand architecture (HRNet, domain adaptation, data handling) | report §2–3 (from the code audit) | ◐ |
| G4 | Training; **monitor over/underfitting** with train/val loss curves and val AP per epoch | per-epoch CSVs + curve figures + discussion | ☐ |
| G5 | Test released + own model; explain metrics (AP, AP50, AP75, AR) | R001/R002, retrains, report §4 | ◐ |
| G6.1 | CLI demo `tools/demo.py --cfg --image --model --output [--conf-threshold 0.3] [--output-size 512,512]` | added to `third_party/.../tools/demo.py` | ☐ |
| G6.2 | Streamlit demo `tools/streamlit_demo.py` (upload, confidence slider, show results/confidences) | added; verified headless | ☐ |
| G7 | Extensions: backbones, hyperparameters, augmentation, synthetic data, DA tweaks; **log every attempt** | improve stream, λ sweep, AGENT.md | ◐ |

## C. Final Submission Requirements (STUDENT_GUIDE.md, verbatim summary)
- Submit a **report** plus the **project repository**. The repo root contains `SKILL.md`, `AGENT.md` and `MANIFEST.md`.
- The report covers:
  1. the reproduction of the original FiDIP baseline numbers;
  2. what was tried to improve it (backbone, augmentation, hyperparameters, data, domain-adaptation tweaks…), **with the resulting accuracy for each attempt**;
  3. the final best configuration and its accuracy vs the published baseline;
  4. a short reflection on how the agentic workflow changed the research approach: what the agent was good at, and where the human had to step in.
- "A well-documented set of attempts that mostly didn't beat the baseline … is worth more than an undocumented lucky result."

## D. Notes
- The guide references `fidip_simple.yml`, `fidip_cuda12.2.yml`, `gpu_test.py`, `tools/demo.py` and `tools/streamlit_demo.py`. These are **not in the upstream repo** at a76b4f4, and a code search shows no public copy, so we implement them to the guide's documented interface.
- Classmates' public forks exist. We deliberately **do not consult other students' repositories**.
- The guide says the SyRIP download needs a password. The main SyRIP zips were public (2026-11-05); only Expanded SyRIP is password-protected.
- Expanded SyRIP's +400 real training images are probably validate500 \ validate100, which is our Val400. Training on them would consume the selection set, so we don't pursue it under the same-split rule.
