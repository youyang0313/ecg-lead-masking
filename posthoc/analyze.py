"""Post-hoc statistics and figures for E1/E2/E6/E7 from saved predictions (CPU only).

All results are secondary and exploratory: they were defined after the primary test
evaluation. Thresholds are chosen on validation only. Intervals are 2,000-draw patient
bootstrap percentiles conditional on the trained seeds, as in the primary analysis.
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
DRAWS = 2000
COLORS = {"A_protocol": "#888888", "A_s12": "#333333", "B_protocol": "#0072B2", "B_s12": "#56B4E9"}
LABELS = {"A_protocol": "A (protocol selection)", "A_s12": "A (S12-only selection)",
          "B_protocol": "B (protocol selection)", "B_s12": "B (S12-only selection)"}


def ece(y, p, bins=15):
    """Equal-width expected calibration error for one binary label."""
    edges = np.linspace(0, 1, bins+1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins-1)
    return float(sum(abs(y[which == b].mean()-p[which == b].mean())*(which == b).mean()
                     for b in range(bins) if (which == b).any()))


def youden_thresholds(y, p):
    thresholds = []
    for j in range(y.shape[1]):
        candidates = np.unique(p[:, j])
        pos, neg = y[:, j] == 1, y[:, j] == 0
        sens = (p[pos, j][None, :] >= candidates[:, None]).mean(axis=1)
        spec = (p[neg, j][None, :] < candidates[:, None]).mean(axis=1)
        thresholds.append(float(candidates[np.argmax(sens+spec-1)]))
    return np.array(thresholds)


def macro(y, p):
    return multilabel_metrics(y, p, ["NORM", "MI", "STTC", "CD", "HYP"])["macro_auroc"]


def main():
    z = np.load(POSTHOC/"inference/predictions.npz", allow_pickle=False)
    manifest = json.loads((POSTHOC/"inference/manifest.json").read_text(encoding="utf-8"))
    models, configs, standard = list(z["models"]), list(z["configurations"]), list(z["standard"])
    y, yv, classes = z["test_targets"], z["validation_targets"], list(z["classes"])
    test, val, variants = z["test"], z["validation"], z["variants"]
    OUT.mkdir(parents=True, exist_ok=True)
    group = lambda arm, rule: [models.index("%s_%s_%d" % (arm, rule, s)) for s in SEEDS]
    groups = {g: group(*g.split("_")) for g in COLORS}
    sc = [configs.index(c) for c in standard]

    # Point estimates: macro-AUROC per model and test configuration.
    auroc = np.array([[macro(y, test[m, c]) for c in range(len(configs))] for m in range(len(models))])
    variant_auroc = np.array([[[macro(y, variants[m, v, c]) for c in range(len(standard))]
                               for v in range(variants.shape[1])] for m in range(len(models))])

    # Patient bootstrap for named contrasts on standard configurations.
    conditions = np.concatenate([test[:, sc], variants[:, 0], variants[:, 1]])  # [3*models, 5, N, 5]
    statistic = WeightedAUROC(y, conditions)
    patients, index = np.unique(z["patient_ids"], return_inverse=True)
    rng = np.random.default_rng(20261023)
    draws = []
    for _ in range(DRAWS):
        value = statistic(patient_weights(index, rng.integers(0, len(patients), len(patients))))
        if value is not None:
            draws.append(value.mean(axis=-1))  # macro over labels -> [conditions, configs]
    draws = np.array(draws)
    M = len(models)
    variant_offset = {"zero": 0, "scale": M, "bnrecal": 2*M}
    point_table = np.concatenate([auroc[:, sc], variant_auroc[:, 0], variant_auroc[:, 1]])

    def contrast(name, left, right, scope):
        cols = [standard.index(c) for c in (PARTIAL if scope == "partial" else ["S12"])]
        def value(table):
            return np.mean([table[..., variant_offset[left[1]]+l, cols].mean(axis=-1)
                            - table[..., variant_offset[right[1]]+r, cols].mean(axis=-1)
                            for l, r in zip(groups[left[0]], groups[right[0]])], axis=0)
        estimate = float(value(point_table))
        lo, hi = np.percentile(value(draws), [2.5, 97.5])
        return dict(contrast=name, scope=scope, estimate=estimate, ci95=[float(lo), float(hi)])

    contrasts = [
        contrast("E1 complete: A(S12-sel) - A(protocol)", ("A_s12", "zero"), ("A_protocol", "zero"), "complete"),
        contrast("E1 complete: B(S12-sel) - A(S12-sel)", ("B_s12", "zero"), ("A_s12", "zero"), "complete"),
        contrast("E1 complete: B(protocol) - A(S12-sel)", ("B_protocol", "zero"), ("A_s12", "zero"), "complete"),
        contrast("E1 partial: B(protocol) - A(S12-sel)", ("B_protocol", "zero"), ("A_s12", "zero"), "partial"),
        contrast("E1 partial: B(S12-sel) - A(S12-sel)", ("B_s12", "zero"), ("A_s12", "zero"), "partial"),
        contrast("E2 partial: A(protocol) BN-recal - zero-fill", ("A_protocol", "bnrecal"), ("A_protocol", "zero"), "partial"),
        contrast("E2 partial: A(S12-sel) BN-recal - zero-fill", ("A_s12", "bnrecal"), ("A_s12", "zero"), "partial"),
        contrast("E2 partial: A(S12-sel) 12/k scale - zero-fill", ("A_s12", "scale"), ("A_s12", "zero"), "partial"),
        contrast("E2 partial: B(protocol) - A(S12-sel) BN-recal", ("B_protocol", "zero"), ("A_s12", "bnrecal"), "partial"),
        contrast("E2 partial: B(protocol) BN-recal - zero-fill", ("B_protocol", "bnrecal"), ("B_protocol", "zero"), "partial"),
    ]

    def summary(table, members):
        return dict(mean=float(table[members].mean()), sd=float(table[members].std(ddof=1)))

    e1 = {g: {c: summary(auroc[:, configs.index(c)], m) for c in standard} for g, m in groups.items()}
    for g, m in groups.items():
        e1[g]["partial_mean"] = summary(auroc[:, [configs.index(c) for c in PARTIAL]].mean(axis=1), m)
    e1_epochs = json.loads((POSTHOC/"e1_replay/summary.json").read_text(encoding="utf-8"))["results"]
    e2 = {g: {v: {c: summary(table[:, i], m) for i, c in enumerate(standard)}
              for v, table in [("zero", auroc[:, sc]), ("scale_12_over_k", variant_auroc[:, 0]),
                               ("bn_recalibration", variant_auroc[:, 1])]} for g, m in groups.items()}
    shift = {}
    for entry in manifest["bn_input_shift"]:
        arm_rule = "_".join(entry["model"].split("_")[:2])
        shift.setdefault(arm_rule, {}).setdefault(entry["configuration"], []).append(
            [layer["mean_standardized_shift"] for layer in entry["layers"]])
    shift = {g: {c: np.mean(v, axis=0).tolist() for c, v in d.items()} for g, d in shift.items()}

    # E6 lead importance and retained-count curve.
    leads = [c[5:] for c in configs if c.startswith("drop_")]
    e6 = {g: dict(drop=[float(auroc[m, configs.index("drop_"+l)].mean()) for l in leads],
                  keep=[float(auroc[m, configs.index("keep_"+l)].mean()) for l in leads],
                  keep_per_label=[[float(np.mean([multilabel_metrics(y, test[i, configs.index("keep_"+l)], classes)
                                                  ["per_label"][c]["auroc"] for i in m])) for c in classes] for l in leads],
                  by_count={k: [float(v) for v in auroc[m][:, [i for i, c in enumerate(configs) if c.startswith("rand_k%02d" % k)]].mean(axis=0)]
                            for k in range(1, 12)}) for g, m in groups.items()}

    # E7 calibration and fixed validation-chosen thresholds (S12 validation, per model and label).
    e7 = {}
    for g, members in groups.items():
        rows = {c: dict(ece=[], brier=[], sensitivity=[], specificity=[], positive_rate=[]) for c in standard}
        for m in members:
            threshold = youden_thresholds(yv, val[m, standard.index("S12")])
            for c in standard:
                p = test[m, configs.index(c)]
                predicted = p >= threshold
                rows[c]["ece"].append(np.mean([ece(y[:, j], p[:, j]) for j in range(5)]))
                rows[c]["brier"].append(float(((p-y)**2).mean()))
                rows[c]["sensitivity"].append(float(np.mean([predicted[y[:, j] == 1, j].mean() for j in range(5)])))
                rows[c]["specificity"].append(float(np.mean([(~predicted[y[:, j] == 0, j]).mean() for j in range(5)])))
                rows[c]["positive_rate"].append(float(predicted.mean()))
        e7[g] = {c: {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in r.items()}
                 for c, r in rows.items()}

    result = dict(status="exploratory post-hoc analysis; primary protocol result unchanged",
                  valid_bootstrap_draws=len(draws), attempted_draws=DRAWS, contrasts=contrasts,
                  e1_macro_auroc=e1, e1_selected_epochs=e1_epochs, e2_macro_auroc=e2,
                  e2_bn_input_shift_by_layer=shift, e6=dict(leads=leads, classes=classes, **{"results": e6}),
                  e7=e7, threshold_rule="per-label Youden J on S12 validation predictions, fixed across test configurations")
    (OUT/"posthoc_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    figures(result, standard)
    print(json.dumps(dict(contrasts=contrasts), indent=2))


def figures(r, standard):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    groups = ["A_protocol", "A_s12", "B_protocol"]

    # E1: validation trajectories from the replay curves.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for arm, color in [("A", COLORS["A_s12"]), ("B", COLORS["B_protocol"])]:
        for seed in SEEDS:
            curve = json.loads((POSTHOC/"e1_replay"/("seed_%d_%s" % (seed, arm))/"curve.json").read_text(encoding="utf-8"))
            epochs = [e["epoch"] for e in curve]
            complete = [e["validation_macro_auroc"]["S12"] for e in curve]
            partial = [np.mean([e["validation_macro_auroc"][c] for c in PARTIAL]) for e in curve]
            kw = dict(color=color, alpha=0.8, lw=1.2, label=arm if seed == 17 else None)
            axes[0].plot(epochs, complete, **kw)
            axes[1].plot(epochs, partial, **kw)
    for item in r["e1_selected_epochs"]:
        color = COLORS["A_s12"] if item["arm"] == "A" else COLORS["B_protocol"]
        axes[0].plot(item["s12"]["epoch"], item["s12"]["score"], "v", color=color, ms=6)
        axes[1].axvline(item["protocol"]["epoch"], color=color, ls=":", lw=0.8)
    axes[0].set(title="Validation S12 macro-AUROC (▼ S12-only selection)", xlabel="Epoch", ylabel="Macro-AUROC")
    axes[1].set(title="Validation partial-input mean (: protocol selection)", xlabel="Epoch")
    axes[0].legend(frameon=False)
    save(fig, "e1_validation_trajectories")

    # E2: test-time repairs for A versus masked training.
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = [("A_s12", "zero", "A zero-fill"), ("A_s12", "scale_12_over_k", "A ×12/k"),
            ("A_s12", "bn_recalibration", "A BN re-estimated"), ("B_protocol", "zero", "B zero-fill"),
            ("B_protocol", "bn_recalibration", "B BN re-estimated")]
    palette = ["#333333", "#999999", "#E69F00", "#0072B2", "#56B4E9"]
    x = np.arange(len(standard))
    for i, ((g, v, label), color) in enumerate(zip(bars, palette)):
        means = [r["e2_macro_auroc"][g][v][c]["mean"] for c in standard]
        sds = [r["e2_macro_auroc"][g][v][c]["sd"] for c in standard]
        ax.bar(x+(i-2)*0.16, means, 0.16, yerr=sds, color=color, label=label, capsize=2)
    ax.set(xticks=x, xticklabels=standard, ylim=(0.4, 0.97), ylabel="Test macro-AUROC (mean ± seed SD)")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "e2_test_time_repairs")

    # E6: lead importance and retained-count curve.
    e6, leads = r["e6"]["results"], r["e6"]["leads"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(len(leads))
    for i, g in enumerate(groups):
        axes[0].bar(x+(i-1)*0.27, e6[g]["drop"], 0.27, color=COLORS[g], label=LABELS[g])
        axes[1].bar(x+(i-1)*0.27, e6[g]["keep"], 0.27, color=COLORS[g])
        ks = np.arange(1, 12)
        values = [e6[g]["by_count"][str(k)] if str(k) in e6[g]["by_count"] else e6[g]["by_count"][k] for k in ks]
        axes[2].plot(ks, [np.mean(v) for v in values], "o-", color=COLORS[g], ms=3)
        axes[2].fill_between(ks, [np.min(v) for v in values], [np.max(v) for v in values], color=COLORS[g], alpha=0.15)
    axes[0].set(xticks=x, xticklabels=leads, ylim=(0.85, 0.94), title="Drop one lead", ylabel="Test macro-AUROC")
    axes[1].set(xticks=x, xticklabels=leads, ylim=(0.4, 0.9), title="Keep only one lead")
    axes[2].set(xlabel="Retained leads k (random subsets)", title="Random subsets: mean and range")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "e6_lead_importance", tight=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for ax, g, title in zip(axes, ["A_s12", "B_protocol"], ["A (S12-only selection)", "B (protocol selection)"]):
        image = ax.imshow(np.array(e6[g]["keep_per_label"]).T, vmin=0.4, vmax=0.95, cmap="viridis", aspect="auto")
        ax.set(xticks=range(12), xticklabels=leads, yticks=range(5), yticklabels=r["e6"]["classes"], title=title)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Single-lead test AUROC")
    save(fig, "e6_single_lead_per_label", tight=False)

    # E7: calibration and fixed-threshold operating points.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    x = np.arange(len(standard))
    for i, g in enumerate(groups):
        for ax, key in zip(axes, ["ece", "sensitivity", "specificity"]):
            ax.bar(x+(i-1)*0.27, [r["e7"][g][c][key]["mean"] for c in standard], 0.27,
                   yerr=[r["e7"][g][c][key]["sd"] for c in standard], color=COLORS[g], capsize=2,
                   label=LABELS[g] if key == "ece" else None)
    axes[0].set(xticks=x, xticklabels=standard, title="Macro ECE (lower is better)")
    axes[1].set(xticks=x, xticklabels=standard, ylim=(0, 1), title="Macro sensitivity @ validation threshold")
    axes[2].set(xticks=x, xticklabels=standard, ylim=(0, 1), title="Macro specificity @ validation threshold")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "e7_calibration_thresholds", tight=False)


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT/(name+"."+ext), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
