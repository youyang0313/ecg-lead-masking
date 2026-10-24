"""Appendix figure B1: three leads of one training record per diagnostic superclass.

Selection is deterministic and metadata-only: training records carrying exactly one
superclass, every diagnostic statement at likelihood 100, validated by a human, adult
(age >= 18) and without any PTB-XL signal-quality flag; the lowest ECG id per class is
taken. Leads II, V2 and V5 are shown in physical mV (training scaler inverted), so both
limb-lead (inferior) and precordial (anterior/lateral) findings can appear. No model is run.
"""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
PREPARED = PROJECT/"data/ptbxl_1_0_3/prepared"
OUT = PROJECT/"report/paper/figures"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FLAGS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "extra_beats", "pacemaker"]
LEADS = ["II", "V2", "V5"]


def main():
    rows = json.loads((PREPARED/"train.records.json").read_text(encoding="utf-8"))
    y = np.load(PREPARED/"train.labels.npy")
    x = np.load(PREPARED/"train.signals.npy", mmap_mode="r")
    scaler = json.loads((PREPARED/"scaler.json").read_text(encoding="utf-8"))
    leads = [scaler["canonical_leads"].index(l) for l in LEADS]
    db = pd.read_csv(PROJECT/"data_review/metadata/ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(PROJECT/"data_review/metadata/scp_statements.csv", index_col=0)
    diagnostic = set(scp.index[scp.diagnostic == 1.0])
    clean = db[FLAGS].isna().all(axis=1)

    def eligible(ecg_id):
        r = db.loc[ecg_id]
        codes = {c: v for c, v in ast.literal_eval(r.scp_codes).items() if c in diagnostic}
        return (clean.loc[ecg_id] and bool(r.validated_by_human) and r.age >= 18 and codes
                and all(v == 100.0 for v in codes.values())), codes

    chosen = []
    for j, name in enumerate(CLASSES):
        for i in sorted((i for i, r in enumerate(rows) if y[i, j] == 1 and y[i].sum() == 1), key=lambda k: rows[k]["ecg_id"]):
            ok, codes = eligible(rows[i]["ecg_id"])
            if ok:
                chosen.append((name, i, rows[i]["ecg_id"], codes))
                break
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(CLASSES), len(LEADS), figsize=(7.5, 7.0), sharex=True)
    t = np.arange(1000)/100.0
    for row, (name, i, ecg_id, codes) in enumerate(chosen):
        for col, (lead, k) in enumerate(zip(LEADS, leads)):
            ax = axes[row, col]
            ax.plot(t, np.asarray(x[i, k], dtype=np.float64)*scaler["applied_scale_mV"][k]+scaler["mean_mV"][k], color="#0072B2", lw=0.6)
            if col == 0:
                ax.set_ylabel("mV")
                ax.set_title("%s: %s (ECG %d)" % (name, ", ".join("%s %d" % (c, v) for c, v in codes.items()), ecg_id), loc="left", fontsize=7)
            if row == 0:
                ax.text(0.99, 0.95, lead, transform=ax.transAxes, ha="right", va="top", fontsize=7, fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT/("appendix_class_examples."+ext), dpi=200)
    (OUT/"appendix_class_examples.json").write_text(json.dumps(dict(
        rule="training records with exactly one superclass, all diagnostic statements at likelihood 100, validated_by_human, age >= 18, "
             "no PTB-XL quality flag; lowest ECG id per class; leads II, V2, V5 in mV",
        records={name: dict(ecg_id=int(ecg_id), codes=codes) for name, _, ecg_id, codes in chosen}), indent=2), encoding="utf-8")
    print({name: (ecg_id, codes) for name, _, ecg_id, codes in chosen})


if __name__ == "__main__":
    main()
