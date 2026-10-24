"""E7b and E14 (post-hoc, CPU only, from saved predictions).

E7b: calibration and fixed-threshold operating points for the repaired complete-input arm.
     Uses E12 test predictions (zero / derived / BN re-estimated / both) and the S12
     validation predictions of the same archived checkpoints. Repairs never change S12
     input, so per-label Youden thresholds are taken from S12 validation as in E7.
E14: MI subclass check. Using the MI output, AUROC for inferior MI (IMI) and anterior MI
     (AMI) records versus records without any MI, per configuration. Records carrying both
     IMI and AMI statements are excluded from both positive sets.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT/"training"))
from metrics import binary_metrics  # noqa: E402

POSTHOC = PROJECT/"experiments/posthoc"
OUT = POSTHOC/"analysis"
SEEDS = [17, 29, 43]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins+1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins-1)
    return float(sum(abs(y[which == b].mean()-p[which == b].mean())*(which == b).mean()
                     for b in range(bins) if (which == b).any()))


def youden(y, p):
    out = []
    for j in range(y.shape[1]):
        cand = np.unique(p[:, j])
        pos, neg = y[:, j] == 1, y[:, j] == 0
        sens = (p[pos, j][None, :] >= cand[:, None]).mean(axis=1)
        spec = (p[neg, j][None, :] < cand[:, None]).mean(axis=1)
        out.append(float(cand[np.argmax(sens+spec-1)]))
    return np.array(out)


def operating(y, p, thr):
    pred = p >= thr
    return dict(ece=float(np.mean([ece(y[:, j], p[:, j]) for j in range(5)])),
                sensitivity=float(np.mean([pred[y[:, j] == 1, j].mean() for j in range(5)])),
                specificity=float(np.mean([(~pred[y[:, j] == 0, j]).mean() for j in range(5)])),
                positive_rate=float(pred.mean()))


def e7b():
    e12 = np.load(POSTHOC/"e12_combined/predictions.npz", allow_pickle=False)
    inf = np.load(POSTHOC/"inference/predictions.npz", allow_pickle=False)
    y, yv = e12["targets"], inf["validation_targets"]
    keys12, conds, cfgs = list(e12["models"]), list(e12["conditions"]), list(e12["configurations"])
    keys_inf, std = list(inf["models"]), list(inf["standard"])
    result = {}
    for arm in ["A", "B"]:
        for rule in ["protocol", "s12"]:
            rows = {}
            for cond in conds:
                per_cfg = {c: [] for c in cfgs}
                for s in SEEDS:
                    key = "%s_%s_%d" % (arm, rule, s)
                    thr = youden(yv, inf["validation"][keys_inf.index(key), std.index("S12")])
                    for ci, c in enumerate(cfgs):
                        per_cfg[c].append(operating(y, e12["probabilities"][keys12.index(key), conds.index(cond), ci], thr))
                rows[cond] = {c: {k: dict(mean=float(np.mean([r[k] for r in v])), sd=float(np.std([r[k] for r in v], ddof=1)))
                                  for k in v[0]} for c, v in per_cfg.items()}
            result["%s_%s" % (arm, rule)] = rows
    return result


def e14():
    inf = np.load(POSTHOC/"inference/predictions.npz", allow_pickle=False)
    rows = json.loads((PROJECT/"data/ptbxl_1_0_3/test_prepared/test.records.json").read_text(encoding="utf-8"))
    db = pd.read_csv(PROJECT/"data_review/metadata/ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(PROJECT/"data_review/metadata/scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1.0]
    sub = {code: diag.loc[code, "diagnostic_subclass"] for code in diag.index}
    subs = [set(sub[c] for c in ast.literal_eval(db.loc[r["ecg_id"], "scp_codes"]) if c in sub) for r in rows]
    y_mi = inf["test_targets"][:, CLASSES.index("MI")]
    imi = np.array(["IMI" in s and "AMI" not in s for s in subs])
    ami = np.array(["AMI" in s and "IMI" not in s for s in subs])
    neg = y_mi == 0
    assert not (imi & neg).any() and not (ami & neg).any()
    keys, cfgs = list(inf["models"]), list(inf["configurations"])
    out = dict(counts=dict(imi=int(imi.sum()), ami=int(ami.sum()), both=int(sum("IMI" in s and "AMI" in s for s in subs)),
                           no_mi=int(neg.sum())), auroc={})
    for group in ["A_s12", "A_protocol", "B_s12", "B_protocol"]:
        out["auroc"][group] = {}
        for c in ["S12", "S6", "S4", "S3", "S2"]:
            vals = {}
            for name, pos in [("IMI", imi), ("AMI", ami)]:
                mask = pos | neg
                scores = [binary_metrics(pos[mask].astype(int), inf["test"][keys.index("%s_%d" % (group, s)), cfgs.index(c)][mask, CLASSES.index("MI")])["auroc"]
                          for s in SEEDS]
                vals[name] = dict(mean=float(np.mean(scores)), sd=float(np.std(scores, ddof=1)))
            out["auroc"][group][c] = vals
    return out


def main():
    res = dict(status="exploratory post-hoc analysis (CPU only, saved predictions)", e7b=e7b(), e14=e14())
    (OUT/"e7b_e14_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for g in ["A_s12", "B_s12", "A_protocol", "B_protocol"]:
        for cond in ["zero", "derived_bn_recal"]:
            r = res["e7b"][g][cond]
            print("%-11s %-17s" % (g, cond), " ".join("%s ECE %.3f sens %.2f spec %.2f pos %.2f" % (c, r[c]["ece"]["mean"], r[c]["sensitivity"]["mean"], r[c]["specificity"]["mean"], r[c]["positive_rate"]["mean"]) for c in ["S6", "S2"]))
    print("E14 counts", res["e14"]["counts"])
    for g, v in res["e14"]["auroc"].items():
        print("%-11s" % g, " | ".join("%s IMI %.3f AMI %.3f" % (c, x["IMI"]["mean"], x["AMI"]["mean"]) for c, x in v.items()))


if __name__ == "__main__":
    main()
