"""Seed-level t intervals (df = 2) for the decision-relevant exploratory contrasts.

Uses the per-seed paired effects already stored in the E11, E12, E13 and E4 summaries.
These intervals describe variation between the three trained seeds; the patient-bootstrap
intervals in the summaries are conditional on those seeds.
"""
from pathlib import Path
import json

import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
POSTHOC = PROJECT/"experiments/posthoc"
WANTED = {
    "e12_combined": ["B(protocol) zero - A(protocol) zero", "B(s12) zero - A(s12) zero",
                     "B(protocol) zero - A(protocol) derived", "B(s12) zero - A(s12) derived",
                     "B(protocol) zero - A(protocol) bn_recal", "B(s12) zero - A(s12) bn_recal",
                     "B(protocol) zero - A(protocol) derived+bn_recal", "B(s12) zero - A(s12) derived+bn_recal",
                     "B(protocol) derived+bn_recal - A(protocol) derived+bn_recal",
                     "B(s12) derived+bn_recal - A(s12) derived+bn_recal"],
    "e13_cutout_eval": ["TC - A(S12-sel)", "B(S12-sel) - TC"],
    "e4_specialist_eval": ["SP - B_s12_zero", "SP - B_protocol_zero", "SP - A_s12_repaired", "SP - A_protocol_repaired"],
}


def main():
    t = stats.t.ppf(0.975, df=2)
    rows = []
    for folder, names in WANTED.items():
        for c in json.loads((POSTHOC/folder/"summary.json").read_text(encoding="utf-8"))["contrasts"]:
            if c["contrast"] in names and c.get("scope", "partial") == "partial" and "per_seed" in c:
                seeds = np.array(c["per_seed"])
                half = t*seeds.std(ddof=1)/np.sqrt(len(seeds))
                rows.append(dict(source=folder, contrast=c["contrast"], estimate=c["estimate"], patient_ci95=c["ci95"],
                                 per_seed=seeds.tolist(), seed_t_ci95=[float(seeds.mean()-half), float(seeds.mean()+half)]))
    out = POSTHOC/"analysis/seed_t_intervals.json"
    out.write_text(json.dumps(dict(df=2, t_quantile=float(t), rows=rows), indent=2), encoding="utf-8")
    for r in rows:
        print("%-58s %+.4f patient [%+.4f, %+.4f] seed-t [%+.4f, %+.4f]" % (r["contrast"], r["estimate"], *r["patient_ci95"], *r["seed_t_ci95"]))


if __name__ == "__main__":
    main()
