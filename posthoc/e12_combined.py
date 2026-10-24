"""E12 (post-hoc, inference only): strongest training-free repair of the complete-input arm.

For each partial configuration, four input conditions: zero-fill; E11 derivation of
missing limb leads from I and II; E2 BatchNorm re-estimation; and both combined, where
BatchNorm statistics are re-estimated on the same 4,096 training records presented with
the same derived representation. Applied to A and B under both selection rules. No
weight updates; paired patient bootstrap as in the other analyses.
"""
from pathlib import Path
import copy
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"training"))
sys.path.insert(0, str(HERE))
from runtime import CLASSES, CONFIG, LEADS, PREPARED, PROJECT, PROTOCOL, batch_tensor, configure, load_development, now, state_hash, write_json
from metrics import multilabel_metrics
from test_statistics import WeightedAUROC, patient_weights
from infer import BATCH, RECALIBRATION_RECORDS, load_test
from e11_derived import listing
from model import ECGClassifier
import numpy as np
import torch
from torch import nn

OUT = PROJECT/"experiments/posthoc/e12_combined"
SEEDS = [17, 29, 43]
PARTIAL = ["S6", "S4", "S3", "S2"]
CONDITIONS = ["zero", "derived", "bn_recal", "derived_bn_recal"]
DRAWS = 2000
INDEX = {lead: i for i, lead in enumerate(LEADS)}


def build_input(z, available, derive, mu, sd, device):
    keep = torch.tensor([lead in available for lead in LEADS], device=device).view(1, 12, 1).float()
    batch = z*keep
    if derive and "I" in available and "II" in available:
        phys = z*sd+mu
        I, II = phys[:, INDEX["I"]], phys[:, INDEX["II"]]
        for lead, value in {"III": II-I, "aVR": -(I+II)/2, "aVL": I-II/2, "aVF": II-I/2}.items():
            if lead not in available:
                k = INDEX[lead]
                batch[:, k] = (value-mu[0, k])/sd[0, k]
    return batch


@torch.no_grad()
def recalibrate(model, train_x, records, available, derive, mu, sd, device):
    clone = copy.deepcopy(model)
    for module in clone.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None
    clone.train()
    for start in range(0, len(records), BATCH):
        clone(build_input(batch_tensor(train_x, records[start:start+BATCH], device), available, derive, mu, sd, device))
    clone.eval()
    return clone


@torch.inference_mode()
def predict(model, x, available, derive, mu, sd, device):
    model.eval()
    out = []
    for start in range(0, len(x), BATCH):
        z = batch_tensor(x, np.arange(start, min(start+BATCH, len(x))), device)
        out.append(torch.sigmoid(model(build_input(z, available, derive, mu, sd, device))).float().cpu().numpy())
    return np.concatenate(out)


def main():
    device = configure()
    development = load_development()
    test = load_test()
    scaler = json.loads((PREPARED/"scaler.json").read_text(encoding="utf-8"))
    mu = torch.tensor(scaler["mean_mV"], device=device, dtype=torch.float32).view(1, 12, 1)
    sd = torch.tensor(scaler["applied_scale_mV"], device=device, dtype=torch.float32).view(1, 12, 1)
    configs = {c["id"]: c["available"] for c in CONFIG["configurations"]}
    # Same 4,096 recalibration records as E2 (seed 20261025).
    records = np.sort(np.random.default_rng(20261025).choice(len(development["train"]["y"]), RECALIBRATION_RECORDS, replace=False))
    items = listing()
    y = test["y"]
    OUT.mkdir(parents=True, exist_ok=False)
    probs = np.zeros((len(items), len(CONDITIONS), len(PARTIAL), len(y), 5), dtype=np.float32)
    for m, item in enumerate(items):
        model = ECGClassifier(PROTOCOL).to(device)
        model.load_state_dict(torch.load(item["path"], map_location="cpu", weights_only=True)["model_state"])
        assert state_hash(model) == item["state_sha256"], item["key"]
        for c, cid in enumerate(PARTIAL):
            available = configs[cid]
            probs[m, 0, c] = predict(model, test["x"], available, False, mu, sd, device)
            probs[m, 1, c] = predict(model, test["x"], available, True, mu, sd, device)
            probs[m, 2, c] = predict(recalibrate(model, development["train"]["x"], records, available, False, mu, sd, device),
                                     test["x"], available, False, mu, sd, device)
            probs[m, 3, c] = predict(recalibrate(model, development["train"]["x"], records, available, True, mu, sd, device),
                                     test["x"], available, True, mu, sd, device)
        assert state_hash(model) == item["state_sha256"]
        print(json.dumps(dict(stage="model_done", model=item["key"])), flush=True)
    keys = [i["key"] for i in items]
    np.savez_compressed(OUT/"predictions.npz", probabilities=probs, targets=y, models=np.array(keys),
                        conditions=np.array(CONDITIONS), configurations=np.array(PARTIAL))

    point = np.array([[[multilabel_metrics(y, probs[m, f, c], CLASSES)["macro_auroc"] for c in range(len(PARTIAL))]
                       for f in range(len(CONDITIONS))] for m in range(len(items))])
    statistic = WeightedAUROC(y, probs.reshape(-1, *y.shape))
    patients, index = np.unique(np.array([r["patient_id"] for r in test["rows"]]), return_inverse=True)
    rng = np.random.default_rng(20261023)
    draws = []
    for _ in range(DRAWS):
        value = statistic(patient_weights(index, rng.integers(0, len(patients), len(patients))))
        if value is not None:
            draws.append(value.mean(axis=-1).reshape(len(items), len(CONDITIONS), len(PARTIAL)))
    draws = np.array(draws)

    def group(arm, rule):
        return [keys.index("%s_%s_%d" % (arm, rule, s)) for s in SEEDS]

    def contrast(name, left, right):
        (la, lr, lf), (ra, rr, rf) = left, right
        l, r = group(la, lr), group(ra, rr)
        lf, rf = CONDITIONS.index(lf), CONDITIONS.index(rf)
        est = float(point[l, lf].mean()-point[r, rf].mean())
        diff = draws[:, l, lf].mean(axis=(1, 2))-draws[:, r, rf].mean(axis=(1, 2))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        seed_effects = (point[l, lf].mean(axis=1)-point[r, rf].mean(axis=1)).tolist()
        return dict(contrast=name, estimate=est, ci95=[float(lo), float(hi)], per_seed=seed_effects)

    contrasts = []
    for rule in ["protocol", "s12"]:
        contrasts += [
            contrast("B(%s) zero - A(%s) zero" % (rule, rule), ("B", rule, "zero"), ("A", rule, "zero")),
            contrast("B(%s) zero - A(%s) derived" % (rule, rule), ("B", rule, "zero"), ("A", rule, "derived")),
            contrast("B(%s) zero - A(%s) bn_recal" % (rule, rule), ("B", rule, "zero"), ("A", rule, "bn_recal")),
            contrast("B(%s) zero - A(%s) derived+bn_recal" % (rule, rule), ("B", rule, "zero"), ("A", rule, "derived_bn_recal")),
            contrast("B(%s) derived+bn_recal - A(%s) derived+bn_recal" % (rule, rule), ("B", rule, "derived_bn_recal"), ("A", rule, "derived_bn_recal")),
            contrast("A(%s) derived+bn_recal - A(%s) derived" % (rule, rule), ("A", rule, "derived_bn_recal"), ("A", rule, "derived")),
        ]
    table = {}
    for arm in ["A", "B"]:
        for rule in ["protocol", "s12"]:
            g = group(arm, rule)
            for f, cond in enumerate(CONDITIONS):
                table["%s_%s_%s" % (arm, rule, cond)] = dict(
                    per_config={cid: float(point[g, f, c].mean()) for c, cid in enumerate(PARTIAL)},
                    partial_mean=float(point[g, f].mean()), partial_mean_seed_sd=float(point[g, f].mean(axis=1).std(ddof=1)))
    write_json(OUT/"summary.json", dict(completed=now(), valid_draws=len(draws), macro_auroc=table, contrasts=contrasts,
                                        recalibration_records=RECALIBRATION_RECORDS, recalibration_seed=20261025, weights_updated=False))
    for c in contrasts:
        print("%-52s %+.4f [%+.4f, %+.4f] seeds %s" % (c["contrast"], c["estimate"], *c["ci95"], [round(s, 3) for s in c["per_seed"]]))
    for k, v in table.items():
        print("%-26s" % k, {c: round(x, 3) for c, x in v["per_config"].items()}, "partial %.4f" % v["partial_mean"])


if __name__ == "__main__":
    main()
