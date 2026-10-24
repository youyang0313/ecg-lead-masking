"""E11 (post-hoc, inference only): fill missing limb leads by Einthoven/Goldberger derivation.

Whenever I and II are both available, every missing limb lead is computed from them in
physical units (III = II - I, aVR = -(I+II)/2, aVL = I - II/2, aVF = II - I/2) and
re-standardized with the training scaler; measured leads are kept as measured and
missing precordial leads stay zero. S4 and S3 therefore both become all-limb + V2, and
S2 becomes all limb leads. Fixed checkpoints, no weight updates; S12-selected and
protocol-selected BatchNorm models of both arms; paired patient bootstrap as before.
"""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"training"))
sys.path.insert(0, str(HERE))
from runtime import CLASSES, CONFIG, LEADS, PREPARED, PROJECT, PROTOCOL, batch_tensor, configure, now, state_hash, write_json
from metrics import multilabel_metrics
from test_statistics import WeightedAUROC, patient_weights
from infer import BATCH, E1, SELECTED, keep_vector, load_test
from model import ECGClassifier
import numpy as np
import torch

OUT = PROJECT/"experiments/posthoc/e11_derived"
SEEDS = [17, 29, 43]
PARTIAL = ["S6", "S4", "S3", "S2"]
LIMB = ["I", "II", "III", "aVR", "aVL", "aVF"]
DRAWS = 2000


def listing():
    archived = {(c["seed"], c["arm"]): c for c in json.loads(SELECTED.read_text(encoding="utf-8"))["checkpoints"]}
    e1 = {(r["seed"], r["arm"]): r for r in json.loads((E1/"summary.json").read_text(encoding="utf-8"))["results"]}
    items = []
    for seed in SEEDS:
        for arm in ["A", "B"]:
            items.append(dict(key="%s_protocol_%d" % (arm, seed), path=archived[(seed, arm)]["checkpoint_path"],
                              state_sha256=archived[(seed, arm)]["selected_state_sha256"]))
            items.append(dict(key="%s_s12_%d" % (arm, seed), path=str(E1/("seed_%d_%s" % (seed, arm))/"s12.pt"),
                              state_sha256=e1[(seed, arm)]["s12"]["state_sha256"]))
    return items


def derived_available(available):
    if "I" in available and "II" in available:
        return sorted(set(available) | set(LIMB), key=LEADS.index)
    return list(available)


@torch.inference_mode()
def predict(model, x, available, derive, device, mu, sd):
    model.eval()
    keep = keep_vector(available, device).float()
    index = {lead: i for i, lead in enumerate(LEADS)}
    out = []
    for start in range(0, len(x), BATCH):
        z = batch_tensor(x, np.arange(start, min(start+BATCH, len(x))), device)
        batch = z*keep
        if derive and "I" in available and "II" in available:
            phys = z*sd+mu
            I, II = phys[:, index["I"]], phys[:, index["II"]]
            formulas = {"III": II-I, "aVR": -(I+II)/2, "aVL": I-II/2, "aVF": II-I/2}
            for lead, value in formulas.items():
                if lead not in available:
                    k = index[lead]
                    batch[:, k] = (value-mu[0, k])/sd[0, k]
        out.append(torch.sigmoid(model(batch)).float().cpu().numpy())
    return np.concatenate(out)


def macro(y, p):
    return multilabel_metrics(y, p, CLASSES)["macro_auroc"]


def main():
    device = configure()
    test = load_test()
    scaler = json.loads((PREPARED/"scaler.json").read_text(encoding="utf-8"))
    mu = torch.tensor(scaler["mean_mV"], device=device, dtype=torch.float32).view(1, 12, 1)
    sd = torch.tensor(scaler["applied_scale_mV"], device=device, dtype=torch.float32).view(1, 12, 1)
    configs = {c["id"]: c["available"] for c in CONFIG["configurations"]}
    items = listing()
    y = test["y"]
    OUT.mkdir(parents=True, exist_ok=False)
    probs = np.zeros((len(items), 2, len(PARTIAL), len(y), 5), dtype=np.float32)  # [model, zero/derived, config]
    for m, item in enumerate(items):
        model = ECGClassifier(PROTOCOL).to(device)
        model.load_state_dict(torch.load(item["path"], map_location="cpu", weights_only=True)["model_state"])
        assert state_hash(model) == item["state_sha256"], item["key"]
        for c, cid in enumerate(PARTIAL):
            probs[m, 0, c] = predict(model, test["x"], configs[cid], False, device, mu, sd)
            probs[m, 1, c] = predict(model, test["x"], configs[cid], True, device, mu, sd)
        assert state_hash(model) == item["state_sha256"]
        print(json.dumps(dict(stage="model_done", model=item["key"])), flush=True)
    keys = [i["key"] for i in items]
    np.savez_compressed(OUT/"predictions.npz", probabilities=probs, targets=y, models=np.array(keys),
                        configurations=np.array(PARTIAL), fills=np.array(["zero", "derived"]))

    point = np.array([[[macro(y, probs[m, f, c]) for c in range(len(PARTIAL))] for f in range(2)] for m in range(len(items))])
    statistic = WeightedAUROC(y, probs.reshape(-1, *y.shape))
    patients, index = np.unique(np.array([r["patient_id"] for r in test["rows"]]), return_inverse=True)
    rng = np.random.default_rng(20261023)
    draws = []
    for _ in range(DRAWS):
        value = statistic(patient_weights(index, rng.integers(0, len(patients), len(patients))))
        if value is not None:
            draws.append(value.mean(axis=-1).reshape(len(items), 2, len(PARTIAL)))
    draws = np.array(draws)

    def group(arm, rule):
        return [keys.index("%s_%s_%d" % (arm, rule, s)) for s in SEEDS]

    def contrast(name, left, right):
        (la, lr, lf), (ra, rr, rf) = left, right
        l, r = group(la, lr), group(ra, rr)
        est = float(point[l, lf].mean() - point[r, rf].mean())
        diff = draws[:, l, lf].mean(axis=(1, 2)) - draws[:, r, rf].mean(axis=(1, 2))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        return dict(contrast=name, estimate=est, ci95=[float(lo), float(hi)])

    Z, D = 0, 1
    contrasts = [
        contrast("B(protocol) zero - A(protocol) zero [primary, reproduced]", ("B", "protocol", Z), ("A", "protocol", Z)),
        contrast("B(protocol) zero - A(protocol) derived", ("B", "protocol", Z), ("A", "protocol", D)),
        contrast("B(protocol) derived - A(protocol) derived", ("B", "protocol", D), ("A", "protocol", D)),
        contrast("A(protocol) derived - A(protocol) zero", ("A", "protocol", D), ("A", "protocol", Z)),
        contrast("B(protocol) derived - B(protocol) zero", ("B", "protocol", D), ("B", "protocol", Z)),
        contrast("B(s12) zero - A(s12) zero", ("B", "s12", Z), ("A", "s12", Z)),
        contrast("B(s12) zero - A(s12) derived", ("B", "s12", Z), ("A", "s12", D)),
        contrast("B(s12) derived - A(s12) derived", ("B", "s12", D), ("A", "s12", D)),
    ]
    table = {}
    for arm in ["A", "B"]:
        for rule in ["protocol", "s12"]:
            g = group(arm, rule)
            for f, fill in enumerate(["zero", "derived"]):
                table["%s_%s_%s" % (arm, rule, fill)] = dict(
                    per_config={cid: float(point[g, f, c].mean()) for c, cid in enumerate(PARTIAL)},
                    partial_mean=float(point[g, f].mean()), partial_mean_seed_sd=float(point[g, f].mean(axis=1).std(ddof=1)))
    write_json(OUT/"summary.json", dict(completed=now(), valid_draws=len(draws), derived_inputs={c: derived_available(configs[c]) for c in PARTIAL},
                                        macro_auroc=table, contrasts=contrasts, weights_updated=False))
    for c in contrasts:
        print("%-60s %+.4f [%+.4f, %+.4f]" % (c["contrast"], c["estimate"], *c["ci95"]))
    for k, v in table.items():
        print(k, {c: round(x, 4) for c, x in v["per_config"].items()}, "partial %.4f" % v["partial_mean"])


if __name__ == "__main__":
    main()
