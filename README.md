# ECG classification when leads go missing

Supplementary material for the MD6329 Deep Learning for Healthcare final project
**"Robustness of 12-lead ECG classifiers to missing leads: masked training versus test-time repair"**
by Wang Youyang (G2606821F), Nanyang Technological University, October 2026.

- Report: `report/ECG_Lead_Robustness_WangYouyang.pdf` (cover page first)
- Presentation: `presentation/01_ecg_lead_masking.py` (Lectrace); spoken script in `presentation/PRESENTATION_SCRIPT.pdf`
- Online repository: https://github.com/youyang0313/ecg-lead-masking
- Presentation demo: https://youyang0313.github.io/ecg-lead-masking/

All code was written for this project. No course assignment code, tutorial or external repository
was used as a basis. The code builds on PyTorch, NumPy, WFDB for Python and scikit-learn, and the
presentation uses Lectrace; all are cited in the report.

## Contents

| Folder | What it holds |
| --- | --- |
| `report/` | final PDF; LaTeX sources (`paper/`), figures and the scripts that draw them; the prespecified protocol (`protocol_v1.json`, `EXPERIMENT_PROTOCOL.md`) |
| `data_review/` | PTB-XL download, decoding, preprocessing and audit scripts with their audit records (no ECG data) |
| `data/ptbxl_1_0_3/` | manifests and README of the prepared arrays (the arrays themselves are not included) |
| `training/` | model, training loop, checkpoint selection, test evaluation, statistics and the verification scripts, with logs and check results |
| `posthoc/` | the exploratory analyses E1–E13 (see the numbering table below) and their logs; `POSTHOC_RESULTS_ZH.md` is the working record of results (Chinese) |
| `experiments/` | outputs: training histories, all 52 trained checkpoints, test predictions and bootstrap draws for the primary analysis, and result summaries and figures for every exploratory analysis |
| `presentation/` | the Lectrace presentation, its helpers and assets, the built static site (`_site/`) and the spoken script |

Files whose names end in `_ZH.md` are working notes and review records in Chinese. They are kept
as the audit trail; the report is the authoritative description.

## Analysis numbering

The report numbers the exploratory analyses E1–E13 (Table 3 of the report). Script and folder
names keep the numbering used while the work was in progress, because the archived outputs record
those names. An explicit-availability-indicator analysis (old E3) was planned but not run, so the
later analyses moved up by one.

| Report | Question | Scripts (`posthoc/`) | Outputs (`experiments/posthoc/`) |
| --- | --- | --- | --- |
| E1 | S12-only checkpoint selection | `e1_replay.py`, `infer.py`, `analyze.py` | `e1_replay/`, `inference/`, `analysis/` |
| E2 | test-time repairs (rescaling, BatchNorm re-estimation) | `infer.py`, `analyze.py` | `inference/`, `analysis/` |
| E3 | per-configuration specialists (old E4) | `e4_specialist_train.py`, `e4_specialist_eval.py` | `e4_specialist_training/`, `e4_specialist_eval/` |
| E4 | masking ratio (old E5) | `e5_ratio_train.py`, `e5_ratio_eval.py` | `e5_ratio_training/`, `e5_ratio_eval/` |
| E5 | lead importance and arbitrary subsets (old E6) | `infer.py`, `analyze.py` | `inference/`, `analysis/` |
| E6 | calibration and fixed thresholds (old E7) | `analyze.py` | `analysis/` |
| E6b | calibration of repaired A (old E7b) | `e7b_e14_analysis.py` | `analysis/e7b_e14_summary.json` |
| E7 | GroupNorm instead of BatchNorm (old E8) | `e8_train.py`, `e8_infer.py`, `e8_analyze.py` | `e8_e9_training/GN_*`, `e8_e10_inference/` |
| E8 | held-out limb-lead family (old E9) | `e8_train.py`, `e8_infer.py`, `e8_analyze.py` | `e8_e9_training/HO_B_*`, `e8_e10_inference/` |
| E9 | representation of missing leads (old E10) | `e8_infer.py`, `e8_analyze.py` | `e8_e10_inference/` |
| E10 | deriving missing limb leads (old E11) | `e11_derived.py` | `e11_derived/` |
| E11 | strongest training-free repair, and its extension to 207 subsets (old E12) | `e12_combined.py`, `e12_subsets_v2.py`, `plot_e12_subsets.py` | `e12_combined/`, `e12_subsets_v2/` |
| E12 | time-cutout control (old E13) | `e13_cutout_train.py`, `e13_cutout_eval.py` | `e13_cutout_training/`, `e13_cutout_eval/` |
| E13 | physiological basis of B's MI output (old E14) | `e7b_e14_analysis.py` | `analysis/e7b_e14_summary.json` |
| — | seed-level t intervals | `seed_t_intervals.py` | `analysis/seed_t_intervals.json` |

`e12_subsets.py` and `experiments/posthoc/e12_subsets/` are the first version of the subset
extension. It derived missing limb leads only when leads I and II were both present; the report
uses the corrected `e12_subsets_v2` and describes the error in Section 5.8.

The primary (prespecified) analysis is in `training/` and `experiments/train/`,
`experiments/test/`: `experiment.py` trains, `evaluate_test.py` runs the fixed checkpoints on the
test fold, `analyze_test.py` and `test_statistics.py` compute the paired patient bootstrap, and
`verify_test.py` checks every metric with scikit-learn.

## Reproducing

1. **Data.** PTB-XL 1.0.3 is public (CC BY 4.0) at https://physionet.org/content/ptb-xl/1.0.3/.
   `data_review/prepare_development.py` and `preprocess_development.py` download and standardize
   the training and validation folds; `prepare_test.py plan|fetch|process|verify` releases the
   test fold separately. Each step checks file hashes against the records in this folder.
2. **Training and test.** `training/run_gpu.ps1 train` trains the six models (seeds 17/29/43 × arms
   A/B, 40 epochs); `evaluate_test.py`, `analyze_test.py` and `verify_test.py` produce and check
   the primary result. Training used Python 3.9, PyTorch 2.6.0, CUDA 12.6 on an RTX 4060 Laptop
   GPU, about 15 minutes of GPU time for all six models.
3. **Exploratory analyses.** Run the scripts in `posthoc/` in the order of the table above. Scripts
   that only evaluate fixed checkpoints run on CPU from the checkpoints included here.
4. **Presentation.** Python 3.11+ with `lectrace`, `numpy` and `torch`:
   `cd presentation`, `python prepare_assets.py`, then `lectrace serve 01_ecg_lead_masking.py`
   and open http://localhost:7000. `presentation/README.md` describes the four execution states.

Not included, to keep the archive small: the PTB-XL waveforms and prepared arrays (1.5 GB, public),
the per-analysis test prediction files of the exploratory analyses (490 MB; each evaluation script
regenerates them from the included checkpoints), and the checkpoints of two short engineering
trial runs. The test predictions and bootstrap draws of the primary analysis are included.

## Use of AI tools

OpenAI Codex helped formulate the problem and wrote and debugged the main pipeline. Claude Code
organized the audit records, proposed several of the exploratory analyses, drafted figures and
text, and built the presentation; further analyses came from simulated peer reviews of the report. The author made the
method decisions and chose which suggestions to run. Section 5.8 of the report describes the
division of work and the errors that the checks found.
