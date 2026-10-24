"""Statistics and figures for E8 (normalization), E9 (held-out limb family), E10 (fill type).

Exploratory post-hoc analyses. All new models use S12-only validation selection, so the
BN reference arms are the E1 S12-selected models. Intervals: 2,000-draw paired patient
bootstrap, conditional on the trained seeds.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"training"))
from metrics import multilabel_metrics
from test_statistics import WeightedAUROC, patient_weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
POSTHOC = PROJECT/"experiments/posthoc"
OUT = POSTHOC/"analysis"
SEEDS = [17, 29, 43]
PARTIAL = ["S6", "S4", "S3", "S2"]
LIMB = {"I", "II", "III", "aVR", "aVL", "aVF"}
DRAWS = 2000
GROUPS = ["A_s12", "B_s12", "B_protocol", "GN_A", "GN_B", "HO_B"]
COLORS = {"A_s12": "#333333", "B_s12": "#56B4E9", "B_protocol": "#0072B2",
          "GN_A": "#E69F00", "GN_B": "#D55E00", "HO_B": "#009E73"}
LABELS = {"A_s12": "BN A", "B_s12": "BN B", "B_protocol": "BN B (protocol sel.)",
          "GN_A": "GroupNorm A", "GN_B": "GroupNorm B", "HO_B": "B, limb-only held out"}


def macro(y, p):
    return multilabel_metrics(y, p, ["NORM", "MI", "STTC", "CD", "HYP"])["macro_auroc"]


def main():
    z = np.load(POSTHOC/"e8_e10_inference/predictions.npz", allow_pickle=False)
    manifest = json.loads((POSTHOC/"inference/manifest.json").read_text(encoding="utf-8"))
    available = manifest["configurations"]
    models, configs, fills = list(z["models"]), list(z["configurations"]), list(z["fill_names"])
    y, test, fill_p = z["test_targets"], z["test"], z["fills"]
    member = {g: [models.index("%s_%d" % (g, s)) for s in SEEDS] for g in GROUPS}
    standard = ["S12"]+PARTIAL
    sc = [configs.index(c) for c in standard]

    auroc = np.array([[macro(y, test[m, c]) for c in range(len(configs))] for m in range(len(models))])
    fill_auroc = np.array([[[macro(y, fill_p[m, f, c]) for c in range(len(PARTIAL))]
                            for f in range(len(fills))] for m in range(len(models))])

    # Conditions for the bootstrap: standard sets, then every fill x partial set.
    M = len(models)
    conditions = np.concatenate([test[:, sc].reshape(-1, *y.shape), fill_p.reshape(-1, *y.shape)])
    def cond_std(m, config):
        return m*len(standard)+standard.index(config)
    def cond_fill(m, fill, config):
        return M*len(standard)+(m*len(fills)+fills.index(fill))*len(PARTIAL)+PARTIAL.index(config)
    statistic = WeightedAUROC(y, conditions)
    patients, index = np.unique(z["patient_ids"], return_inverse=True)
    rng = np.random.default_rng(20261023)
    draws = []
    for _ in range(DRAWS):
        value = statistic(patient_weights(index, rng.integers(0, len(patients), len(patients))))
        if value is not None:
            draws.append(value.mean(axis=-1))
    draws = np.array(draws)
    point = np.array([macro(y, conditions[k]) for k in range(len(conditions))])

    def contrast(name, left, right):
        """left/right: equal-length lists of condition indices, paired by position."""
        l, r = np.array(left), np.array(right)
        estimate = float(point[l].mean()-point[r].mean())
        lo, hi = np.percentile(draws[:, l].mean(axis=1)-draws[:, r].mean(axis=1), [2.5, 97.5])
        return dict(contrast=name, estimate=estimate, ci95=[float(lo), float(hi)])

    def std(group, cfgs):
        return [cond_std(m, c) for m in member[group] for c in cfgs]

    def fil(group, fill, cfgs):
        return [cond_fill(m, fill, c) for m in member[group] for c in cfgs]

    contrasts = [
        contrast("E8 complete: GN_A - BN A", std("GN_A", ["S12"]), std("A_s12", ["S12"])),
        contrast("E8 complete: GN_B - BN B", std("GN_B", ["S12"]), std("B_s12", ["S12"])),
        contrast("E8 partial: GN_A - BN A", std("GN_A", PARTIAL), std("A_s12", PARTIAL)),
        contrast("E8 partial: GN_B - GN_A", std("GN_B", PARTIAL), std("GN_A", PARTIAL)),
        contrast("E8 partial: BN B - GN_A", std("B_s12", PARTIAL), std("GN_A", PARTIAL)),
        contrast("E8 partial: GN_B - BN B", std("GN_B", PARTIAL), std("B_s12", PARTIAL)),
        contrast("E9 held-out family S6/S2: HO_B - BN B", std("HO_B", ["S6", "S2"]), std("B_s12", ["S6", "S2"])),
        contrast("E9 seen family S4/S3: HO_B - BN B", std("HO_B", ["S4", "S3"]), std("B_s12", ["S4", "S3"])),
        contrast("E9 held-out family S6/S2: HO_B - BN A", std("HO_B", ["S6", "S2"]), std("A_s12", ["S6", "S2"])),
        contrast("E9 complete: HO_B - BN B", std("HO_B", ["S12"]), std("B_s12", ["S12"])),
    ]
    for fill in fills:
        for g in ["A_s12", "B_protocol", "GN_B"]:
            contrast_name = "E10 partial %s: %s fill - zero-fill" % (g, fill)
            contrasts.append(contrast(contrast_name, fil(g, fill, PARTIAL), std(g, PARTIAL)))
        contrasts.append(contrast("E10 partial %s fill: BN B - BN A" % fill,
                                  fil("B_protocol", fill, PARTIAL), fil("A_s12", fill, PARTIAL)))

    def summary(values, members):
        return dict(mean=float(values[members].mean()), sd=float(values[members].std(ddof=1)))

    table = {g: {c: summary(auroc[:, configs.index(c)], member[g]) for c in standard} for g in GROUPS}
    for g in GROUPS:
        table[g]["partial_mean"] = summary(auroc[:, [configs.index(c) for c in PARTIAL]].mean(axis=1), member[g])
    leads = [c[5:] for c in configs if c.startswith("keep_")]
    keep = {g: [float(auroc[member[g], configs.index("keep_"+l)].mean()) for l in leads] for g in GROUPS}
    by_count = {g: {k: float(auroc[member[g]][:, [i for i, c in enumerate(configs) if c.startswith("rand_k%02d" % k)]].mean())
                    for k in range(1, 12)} for g in GROUPS}
    # E9: every evaluated subset split by whether it is limb-only (held out for HO_B).
    subsets = [c for c in configs if c != "S12" and not c.startswith("drop_")]
    limb_only = [c for c in subsets if set(available[c]) <= LIMB]
    other = [c for c in subsets if not set(available[c]) <= LIMB]
    family = {g: dict(limb_only=float(auroc[member[g]][:, [configs.index(c) for c in limb_only]].mean()),
                      with_precordial=float(auroc[member[g]][:, [configs.index(c) for c in other]].mean()))
              for g in GROUPS}
    fill_table = {g: {f: summary(fill_auroc[:, i].mean(axis=1), member[g]) for i, f in enumerate(fills)} for g in GROUPS}
    for g in GROUPS:
        fill_table[g]["zero"] = table[g]["partial_mean"]
    curves = {}
    for g, prefix in [("GN_A", "GN_A"), ("GN_B", "GN_B"), ("HO_B", "HO_B")]:
        curves[g] = [json.loads((POSTHOC/"e8_e9_training"/("%s_%d" % (prefix, s))/"curve.json").read_text(encoding="utf-8"))
                     for s in SEEDS]
    result = dict(status="exploratory post-hoc analysis; primary protocol result unchanged",
                  valid_bootstrap_draws=len(draws), contrasts=contrasts, macro_auroc=table,
                  keep_one=dict(leads=leads, values=keep), random_subset_by_count=by_count,
                  e9_family=dict(limb_only_subsets=len(limb_only), with_precordial_subsets=len(other), values=family),
                  e10_partial_mean_by_fill=fill_table)
    (OUT/"e8_e10_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    figures(result, curves, fills)
    for c in contrasts:
        print("%-55s %+.4f [%+.4f, %+.4f]" % (c["contrast"], c["estimate"], *c["ci95"]))


def figures(r, curves, fills):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    standard = ["S12"]+PARTIAL

    # E8: BN vs GroupNorm per configuration + keep-one curve by retained count.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    groups = ["A_s12", "GN_A", "B_s12", "GN_B"]
    x = np.arange(len(standard))
    for i, g in enumerate(groups):
        axes[0].bar(x+(i-1.5)*0.2, [r["macro_auroc"][g][c]["mean"] for c in standard], 0.2,
                    yerr=[r["macro_auroc"][g][c]["sd"] for c in standard], color=COLORS[g], capsize=2, label=LABELS[g])
        ks = np.arange(1, 12)
        axes[1].plot(ks, [r["random_subset_by_count"][g][str(k)] if str(k) in r["random_subset_by_count"][g]
                          else r["random_subset_by_count"][g][k] for k in ks], "o-", ms=3, color=COLORS[g])
    axes[0].set(xticks=x, xticklabels=standard, ylim=(0.45, 0.96), ylabel="Test macro-AUROC (mean ± seed SD)",
                title="Protocol configurations")
    axes[1].set(xlabel="Retained leads k (random subsets)", title="Random subsets (mean)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "e8_normalization_ablation", tight=False)

    # E8 training dynamics: GroupNorm A keeps partial-input performance as training continues.
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for g, color in [("GN_A", COLORS["GN_A"]), ("GN_B", COLORS["GN_B"])]:
        for i, curve in enumerate(curves[g]):
            ax.plot([e["epoch"] for e in curve], [np.mean([e["validation_macro_auroc"][c] for c in PARTIAL]) for e in curve],
                    color=color, lw=1.2, label=LABELS[g] if i == 0 else None)
    for i, seed in enumerate(SEEDS):
        curve = json.loads((POSTHOC/"e1_replay"/("seed_%d_A" % seed)/"curve.json").read_text(encoding="utf-8"))
        ax.plot([e["epoch"] for e in curve], [np.mean([e["validation_macro_auroc"][c] for c in PARTIAL]) for e in curve],
                color=COLORS["A_s12"], lw=1.2, label=LABELS["A_s12"] if i == 0 else None)
    ax.set(xlabel="Epoch", ylabel="Validation partial-input mean AUROC")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "e8_training_dynamics")

    # E9: held-out limb family.
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    cats = ["S6\n(held out)", "S2\n(held out)", "S4\n(seen family)", "S3\n(seen family)", "All limb-only\nsubsets", "Subsets with\nprecordial"]
    groups = ["A_s12", "HO_B", "B_s12"]
    x = np.arange(len(cats))
    for i, g in enumerate(groups):
        values = [r["macro_auroc"][g][c]["mean"] for c in ["S6", "S2", "S4", "S3"]] + \
                 [r["e9_family"]["values"][g]["limb_only"], r["e9_family"]["values"][g]["with_precordial"]]
        ax.bar(x+(i-1)*0.27, values, 0.27, color=COLORS[g], label=LABELS[g])
    ax.set(xticks=x, xticklabels=cats, ylim=(0.45, 0.95), ylabel="Test macro-AUROC")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "e9_heldout_limb_family")

    # E10: representation of unavailable leads.
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    order = ["zero"]+fills
    groups = ["A_s12", "B_protocol", "GN_A", "GN_B"]
    x = np.arange(len(order))
    for i, g in enumerate(groups):
        ax.bar(x+(i-1.5)*0.2, [r["e10_partial_mean_by_fill"][g][f]["mean"] for f in order], 0.2,
               yerr=[r["e10_partial_mean_by_fill"][g][f]["sd"] for f in order], color=COLORS[g], capsize=2, label=LABELS[g])
    ax.set(xticks=x, xticklabels=["zero (train mean)", "noise σ=0.1", "noise σ=1", "baseline wander", "other record"],
           ylim=(0.45, 0.95), ylabel="Partial-input mean AUROC")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "e10_missing_lead_representation")


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT/(name+"."+ext), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
