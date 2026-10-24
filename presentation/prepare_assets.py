"""Collect every number and figure the presentation shows into `assets/`.

The lecture must not contain a typed-in number: it reads `assets/results.json`, which this
script writes from the result files the experiments produced. Re-run it after any experiment
is re-run, then rebuild the lecture. Analysis IDs follow the report (E1--E13); the source
folders keep the original proposal numbers (old -> new: E4->E3, E5->E4, E8->E7, E9->E8,
E10->E9, E11->E10, E12->E11, E13->E12, E14->E13).

Usage: python prepare_assets.py
"""
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EXP = PROJECT/"experiments"
POST = EXP/"posthoc"
OUT = Path(__file__).resolve().parent/"assets"
OUT.mkdir(exist_ok=True)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contrast(summary, name, scope=None):
    for c in summary["contrasts"]:
        if c["contrast"] == name and (scope is None or c.get("scope") == scope):
            return {"estimate": c["estimate"], "ci": c["ci95"]}
    raise KeyError(name)


test = load(EXP/"test/20261022T202402963049Z/analysis/test_summary.json")
posthoc = load(POST/"analysis/posthoc_summary.json")
e7_e9 = load(POST/"analysis/e8_e10_summary.json")          # E7 GroupNorm, E8 held-out family, E9 fills
e11 = load(POST/"e12_combined/summary.json")                # E10/E11 repairs
e11s = load(POST/"e12_subsets_v2/summary.json")            # E11 subset extension
e12 = load(POST/"e13_cutout_eval/summary.json")            # E12 time-cutout control
e3 = load(POST/"e4_specialist_eval/summary.json")          # E3 specialists
e4 = load(POST/"e5_ratio_eval/summary.json")               # E4 masking ratio
seed_t = load(POST/"analysis/seed_t_intervals.json")
calib = load(POST/"analysis/e7b_e14_summary.json")         # E6b calibration, E13 MI subclasses

configs = ["S12", "S6", "S4", "S3", "S2"]
partial = configs[1:]
seed_t_primary = next(r for r in seed_t["rows"] if r["contrast"] == "B(protocol) zero - A(protocol) zero")

results = {
    "cohort": {"records": test["records"], "patients": test["patients"], "positives": test["positive_counts"]},
    "primary": {
        "estimate": test["primary"]["estimate"],
        "ci": test["primary"]["patient_bootstrap_95_ci"],
        "seed_effects": test["primary"]["paired_seed_effects"],
        "seed_t_ci": seed_t_primary["seed_t_ci95"],
        "A_partial": test["partial_macro_auroc_means"]["A"],
        "B_partial": test["partial_macro_auroc_means"]["B"],
    },
    "by_config": {c: {"A": test["macro_auroc_by_configuration"][c]["A_mean"],
                      "B": test["macro_auroc_by_configuration"][c]["B_mean"],
                      "diff": test["macro_auroc_by_configuration"][c]["paired_B_minus_A"],
                      "ci": test["macro_auroc_by_configuration"][c]["paired_difference_patient_ci_descriptive"]}
                  for c in configs},
    "complete_s12_selection": contrast(posthoc, "E1 complete: B(S12-sel) - A(S12-sel)"),
    "bn_shift_mean_over_seeds": {g: {c: posthoc["e2_bn_input_shift_by_layer"][g][c] for c in ["S12", "S2"]}
                                 for g in ["A_s12", "B_s12"]},
    "ladder": {
        "A_zero": e11["macro_auroc"]["A_s12_zero"],
        "A_derived": e11["macro_auroc"]["A_s12_derived"],
        "A_bn": e11["macro_auroc"]["A_s12_bn_recal"],
        "A_derived_bn": e11["macro_auroc"]["A_s12_derived_bn_recal"],
        "B_zero": e11["macro_auroc"]["B_s12_zero"],
        "A_groupnorm": {"partial_mean": e7_e9["macro_auroc"]["GN_A"]["partial_mean"]["mean"]},
        "time_cutout": {"partial_mean": e12["macro_auroc"]["TC"]["partial_mean"]},
        "specialists": {"partial_mean": e3["macro_auroc"]["SP"]["partial_mean"]},
        "B_minus_repaired_s12": contrast(e11, "B(s12) zero - A(s12) derived+bn_recal"),
        "B_minus_repaired_protocol": contrast(e11, "B(protocol) zero - A(protocol) derived+bn_recal"),
        "specialists_minus_B": contrast(e3, "SP - B_s12_zero", "partial"),
        "time_cutout_minus_A": contrast(e12, "TC - A(S12-sel)", "partial"),
    },
    "subsets": {
        "n": e11s["n_subsets"],
        "scope_sizes": e11s["scope_sizes"],
        "all": contrast(e11s, "B_s12_zero - A_s12_repaired", "all"),
        "two_or_more_limb": contrast(e11s, "B_s12_zero - A_s12_repaired", "two_or_more_limb"),
        "fewer_than_two_limb": contrast(e11s, "B_s12_zero - A_s12_repaired", "fewer_than_two_limb"),
        "single_lead": contrast(e11s, "B_s12_zero - A_s12_repaired", "single_lead"),
        "binomial_weighted_gap": e11s["gap_weighted_by_binomial_count"],
    },
    "masking_ratio": {k: e4["macro_auroc"][k]["partial_mean"] for k in ["p025", "B_s12_p050", "p075", "p100"]},
    "masking_ratio_complete": {k: e4["macro_auroc"][k]["per_config"]["S12"] for k in ["p025", "B_s12_p050", "p075", "p100"]},
    "ratio_1_minus_half": contrast(e4, "p100 - B_s12_p050", "partial"),
    "fills": {
        "noise_1_cost_B": contrast(e7_e9, "E10 partial B_protocol: noise_1.0 fill - zero-fill"),
        "other_record_cost_B": contrast(e7_e9, "E10 partial B_protocol: other_record fill - zero-fill"),
    },
    "calibration_S2": {
        "A_zero": calib["e7b"]["A_s12"]["zero"]["S2"]["ece"]["mean"],
        "A_repaired": calib["e7b"]["A_s12"]["derived_bn_recal"]["S2"]["ece"]["mean"],
        "B": calib["e7b"]["B_s12"]["zero"]["S2"]["ece"]["mean"],
    },
}
(OUT/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
shutil.copy(POST/"e12_subsets_v2/per_subset_s12.json", OUT/"per_subset_s12.json")
for name in ["test_discrimination.png", "e12_subset_gap.png"]:
    shutil.copy(PROJECT/"report/paper/figures"/name, OUT/name)
print("wrote", OUT)
