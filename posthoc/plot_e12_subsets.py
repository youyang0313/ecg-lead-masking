"""Figure: B (zero-fill) minus E12-repaired A across the 207 E6 subsets (S12-only selection).

Reads experiments/posthoc/e12_subsets_v2/per_subset_s12.json; each point is one subset
(mean over three seeds); lines are means per retained count; panels split subsets by
whether at least two limb leads remain (limb-lead derivation possible).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT/"experiments/posthoc/e12_subsets_v2/per_subset_s12.json"
OUTS = [PROJECT/"experiments/posthoc/analysis", PROJECT/"report/paper/figures"]


def main():
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    groups = [("At least two limb leads (n = %d)", lambda r: r["n_limb"] >= 2, "#0072B2"),
              ("Fewer than two limb leads (n = %d)", lambda r: r["n_limb"] < 2, "#D55E00")]
    rng = np.random.default_rng(0)
    for ax, (title, keep, color) in zip(axes, groups):
        sel = [r for r in rows if keep(r)]
        k = np.array([r["k"] for r in sel])
        gap = np.array([r["B_s12_zero"]-r["A_s12_repaired"] for r in sel])
        ax.scatter(k+rng.uniform(-0.15, 0.15, len(k)), gap, s=8, color=color, alpha=0.5, linewidths=0)
        ks = sorted(set(k))
        ax.plot(ks, [gap[k == v].mean() for v in ks], color=color, lw=1.5, marker="o", ms=3)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_title(title % len(sel), fontsize=8)
        ax.set_xlabel("Retained leads k")
        ax.set_xticks(range(1, 12))
    axes[0].set_ylabel("B minus repaired A\n(macro-AUROC)")
    fig.tight_layout()
    for out in OUTS:
        for ext in ["pdf", "png"]:
            fig.savefig(out/("e12_subset_gap."+ext), dpi=200)
    print("saved")


if __name__ == "__main__":
    main()
