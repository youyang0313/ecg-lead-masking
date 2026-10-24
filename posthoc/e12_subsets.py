"""E12 on E6 subsets (post-hoc, inference only): strongest training-free repair beyond the
four protocol configurations.

Every distinct partial-input lead set evaluated in E6 (protocol sets, single leads, drop-one
leads and the 204 random subsets) is repaired as in E12: missing limb leads are derived from
I and II when both are present, and BatchNorm statistics are re-estimated on the same 4,096
training records presented with that representation. Applied to A and B under both selection
rules; zero-fill predictions are read from the E1/E2/E6 inference archive. Paired patient
bootstrap (2,000 draws, seed 20261023) for subset-averaged contrasts, conditional on the seeds.
"""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"training"))
sys.path.insert(0, str(HERE))
from runtime import CLASSES, LEADS, PREPARED, PROJECT, PROTOCOL, configure, load_development, now, state_hash, write_json
from metrics import multilabel_metrics
from test_statistics import WeightedAUROC, patient_weights
from infer import RECALIBRATION_RECORDS, configurations, load_test
from e11_derived import listing
from e12_combined import predict, recalibrate
from model import ECGClassifier
import numpy as np
import torch

OUT = PROJECT/"experiments/posthoc/e12_subsets"
SEEDS = [17, 29, 43]
DRAWS = 2000
LIMB = {"I", "II", "III", "aVR", "aVL", "aVF"}
GROUPS = ["A_s12", "A_protocol", "B_s12", "B_protocol"]


def distinct_partial_sets():
    _, everything = configurations()
    seen, out = set(), []
    for name, available in everything:
        key = frozenset(available)
        if len(key) < 12 and key not in seen:
            seen.add(key)
            out.append((name, sorted(available, key=LEADS.index)))
    return out


def main():
    device = configure()
    development = load_development()
    test = load_test()
    y = test["y"]
    scaler = json.loads((PREPARED/"scaler.json").read_text(encoding="utf-8"))
    mu = torch.tensor(scaler["mean_mV"], device=device, dtype=torch.float32).view(1, 12, 1)
    sd = torch.tensor(scaler["applied_scale_mV"], device=device, dtype=torch.float32).view(1, 12, 1)
    records = np.sort(np.random.default_rng(20261025).choice(len(development["train"]["y"]), RECALIBRATION_RECORDS, replace=False))
    subsets = distinct_partial_sets()
    archive = np.load(PROJECT/"experiments/posthoc/inference/predictions.npz", allow_pickle=False)
    names, cfgs = list(archive["models"]), list(archive["configurations"])
    assert np.array_equal(archive["test_targets"], y)
    items = {i["key"]: i for i in listing()}
    keys = ["%s_%d" % (g, s) for g in GROUPS for s in SEEDS]
    zero = np.stack([archive["test"][names.index(k)][[cfgs.index(n) for n, _ in subsets]] for k in keys])
    repaired = np.zeros_like(zero)
    for m, key in enumerate(keys):
        model = ECGClassifier(PROTOCOL).to(device)
        model.load_state_dict(torch.load(items[key]["path"], map_location="cpu", weights_only=True)["model_state"])
        assert state_hash(model) == items[key]["state_sha256"], key
        for c, (_, available) in enumerate(subsets):
            fixed = recalibrate(model, development["train"]["x"], records, available, True, mu, sd, device)
            repaired[m, c] = predict(fixed, test["x"], available, True, mu, sd, device)
        assert state_hash(model) == items[key]["state_sha256"]
        print(json.dumps(dict(stage="model_done", model=key)), flush=True)
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(OUT/"predictions.npz", repaired=repaired, targets=y, models=np.array(keys),
                        subsets=np.array([n for n, _ in subsets]))

    counts = np.array([len(a) for _, a in subsets])
    limb = np.array([set(a) <= LIMB for _, a in subsets])
    scopes = {"all": np.ones(len(subsets), bool), "limb_only": limb, "with_precordial": ~limb}
    arms = {}
    for g, group in enumerate(GROUPS):
        arms[group+"_zero"] = zero[g*3:(g+1)*3]
        arms[group+"_repaired"] = repaired[g*3:(g+1)*3]
    arm_names = list(arms)
    point = {a: np.array([[multilabel_metrics(y, p[s, c], CLASSES)["macro_auroc"] for c in range(len(subsets))]
                          for s in range(len(SEEDS))]) for a, p in arms.items()}
    patients, index = np.unique(np.array([r["patient_id"] for r in test["rows"]]), return_inverse=True)
    rng = np.random.default_rng(20261023)
    weights = [patient_weights(index, rng.integers(0, len(patients), len(patients))) for _ in range(DRAWS)]
    # Draws are shared across arms; each arm is evaluated separately to bound memory.
    draws = {}
    for a in arm_names:
        statistic = WeightedAUROC(y, arms[a].reshape(-1, *y.shape))
        values = []
        for w in weights:
            v = statistic(w)
            values.append(None if v is None else v.mean(axis=-1).reshape(len(SEEDS), len(subsets)))
        draws[a] = values
        print(json.dumps(dict(stage="bootstrap_done", arm=a)), flush=True)
    valid = [i for i in range(DRAWS) if all(draws[a][i] is not None for a in arm_names)]

    def contrast(left, right, scope):
        cols = scopes[scope]
        est = float(point[left][:, cols].mean()-point[right][:, cols].mean())
        diff = np.array([draws[left][i][:, cols].mean()-draws[right][i][:, cols].mean() for i in valid])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        per_seed = (point[left][:, cols].mean(axis=1)-point[right][:, cols].mean(axis=1)).tolist()
        return dict(contrast="%s - %s" % (left, right), scope=scope, n_subsets=int(cols.sum()), estimate=est,
                    ci95=[float(lo), float(hi)], per_seed=per_seed)

    pairs = [("B_s12_zero", "A_s12_repaired"), ("B_protocol_zero", "A_protocol_repaired"),
             ("B_s12_repaired", "A_s12_repaired"), ("B_s12_zero", "A_s12_zero"), ("A_s12_repaired", "A_s12_zero")]
    contrasts = [contrast(l, r, s) for l, r in pairs for s in scopes]
    by_count = {a: {int(k): float(point[a][:, counts == k].mean()) for k in sorted(set(counts))} for a in arm_names}
    table = {a: {s: float(point[a][:, cols].mean()) for s, cols in scopes.items()} for a in arm_names}
    write_json(OUT/"summary.json", dict(completed=now(), valid_draws=len(valid), n_subsets=len(subsets),
                                        n_limb_only=int(limb.sum()), macro_auroc=table, by_retained_count=by_count,
                                        contrasts=contrasts, weights_updated=False))
    for a, v in table.items():
        print("%-22s" % a, {s: round(x, 4) for s, x in v.items()})
    for c in contrasts:
        print("%-38s %-16s %+.4f [%+.4f, %+.4f] seeds %s" % (c["contrast"], c["scope"], c["estimate"], *c["ci95"],
                                                           [round(x, 3) for x in c["per_seed"]]))
    for a in ["A_s12_zero", "A_s12_repaired", "B_s12_zero"]:
        print(a, {k: round(x, 3) for k, x in by_count[a].items()})


if __name__ == "__main__":
    main()
