"""E12 on E6 subsets, v2: limb-lead derivation from ANY two available limb leads.

v1 (e12_subsets.py) derived missing limb leads only when I and II were both present, which
left 94 subsets that contain at least two other limb leads with BatchNorm re-estimation only.
Every limb lead is a linear combination of I and II (III = II - I, aVR = -(I+II)/2,
aVL = I - II/2, aVF = II - I/2) and no two limb-lead axes are parallel, so any two limb leads
determine all six. Here I and II are solved by least squares from all available limb leads
(exact for two), the missing limb leads are computed from them, and measured leads keep their
values. Subsets whose input is unchanged from v1 (both I and II present, or fewer than two
limb leads) reuse the v1 repaired predictions. Same 4,096 re-estimation records; paired
patient bootstrap (2,000 draws, seed 20261023), arms evaluated in parallel processes.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import copy
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"training"))
sys.path.insert(0, str(HERE))
from runtime import CLASSES, LEADS, PREPARED, PROJECT, PROTOCOL, batch_tensor, configure, load_development, now, state_hash, write_json
from metrics import multilabel_metrics
from test_statistics import WeightedAUROC, patient_weights
from infer import BATCH, RECALIBRATION_RECORDS, load_test
from e11_derived import listing
from e12_subsets import GROUPS, LIMB, SEEDS, distinct_partial_sets
from model import ECGClassifier
import numpy as np
import torch
from torch import nn

OUT = PROJECT/"experiments/posthoc/e12_subsets_v2"
V1 = PROJECT/"experiments/posthoc/e12_subsets"
DRAWS = 2000
INDEX = {lead: i for i, lead in enumerate(LEADS)}
COEF = {"I": (1.0, 0.0), "II": (0.0, 1.0), "III": (-1.0, 1.0), "aVR": (-0.5, -0.5), "aVL": (1.0, -0.5), "aVF": (-0.5, 1.0)}


def derivable_change(available):
    limb = [l for l in available if l in LIMB]
    return len(limb) >= 2 and not ("I" in available and "II" in available)


def build_input(z, available, mu, sd, device):
    keep = torch.tensor([lead in available for lead in LEADS], device=device).view(1, 12, 1).float()
    batch = z*keep
    limb = [l for l in available if l in LIMB]
    if len(limb) >= 2:
        phys = z*sd+mu
        c = torch.tensor([COEF[l] for l in limb], device=device)
        values = torch.stack([phys[:, INDEX[l]] for l in limb], dim=1)  # [N, L, T]
        basis = torch.einsum("kl,nlt->nkt", torch.linalg.pinv(c), values)  # [N, 2, T] = I, II
        for lead in LIMB - set(available):
            a, b = COEF[lead]
            k = INDEX[lead]
            batch[:, k] = (a*basis[:, 0]+b*basis[:, 1]-mu[0, k])/sd[0, k]
    return batch


@torch.no_grad()
def recalibrate(model, train_x, records, available, mu, sd, device):
    clone = copy.deepcopy(model)
    for module in clone.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None
    clone.train()
    for start in range(0, len(records), BATCH):
        clone(build_input(batch_tensor(train_x, records[start:start+BATCH], device), available, mu, sd, device))
    clone.eval()
    return clone


@torch.inference_mode()
def predict(model, x, available, mu, sd, device):
    model.eval()
    out = []
    for start in range(0, len(x), BATCH):
        z = batch_tensor(x, np.arange(start, min(start+BATCH, len(x))), device)
        out.append(torch.sigmoid(model(build_input(z, available, mu, sd, device))).float().cpu().numpy())
    return np.concatenate(out)


def bootstrap_arm(args):
    y, probs, index, n_patients, n_seeds, n_subsets = args
    rng = np.random.default_rng(20261023)
    statistic = WeightedAUROC(y, probs.reshape(-1, *y.shape))
    values = []
    for _ in range(DRAWS):
        v = statistic(patient_weights(index, rng.integers(0, n_patients, n_patients)))
        values.append(None if v is None else v.mean(axis=-1).reshape(n_seeds, n_subsets))
    return values


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
    v1 = np.load(V1/"predictions.npz", allow_pickle=False)
    keys = list(v1["models"])
    assert list(v1["subsets"]) == [n for n, _ in subsets] and np.array_equal(v1["targets"], y)
    repaired = v1["repaired"].copy()
    changed = [c for c, (_, a) in enumerate(subsets) if derivable_change(a)]
    items = {i["key"]: i for i in listing()}
    for m, key in enumerate(keys):
        model = ECGClassifier(PROTOCOL).to(device)
        model.load_state_dict(torch.load(items[key]["path"], map_location="cpu", weights_only=True)["model_state"])
        assert state_hash(model) == items[key]["state_sha256"], key
        for c in changed:
            available = subsets[c][1]
            repaired[m, c] = predict(recalibrate(model, development["train"]["x"], records, available, mu, sd, device),
                                     test["x"], available, mu, sd, device)
        assert state_hash(model) == items[key]["state_sha256"]
        print(json.dumps(dict(stage="model_done", model=key, recomputed_subsets=len(changed))), flush=True)
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(OUT/"predictions.npz", repaired=repaired, targets=y, models=np.array(keys),
                        subsets=np.array([n for n, _ in subsets]), recomputed=np.array(changed))

    archive = np.load(PROJECT/"experiments/posthoc/inference/predictions.npz", allow_pickle=False)
    names, cfgs = list(archive["models"]), list(archive["configurations"])
    zero = np.stack([archive["test"][names.index(k)][[cfgs.index(n) for n, _ in subsets]] for k in keys])
    del archive
    counts = np.array([len(a) for _, a in subsets])
    n_limb = np.array([len(set(a) & LIMB) for _, a in subsets])
    has_i_ii = np.array([("I" in a and "II" in a) for _, a in subsets])
    scopes = {"all": np.ones(len(subsets), bool), "limb_only": np.array([set(a) <= LIMB for _, a in subsets]),
              "with_precordial": np.array([not set(a) <= LIMB for _, a in subsets]),
              "two_or_more_limb": n_limb >= 2, "fewer_than_two_limb": n_limb < 2,
              "has_I_and_II": has_i_ii, "two_or_more_limb_without_I_II": (n_limb >= 2) & ~has_i_ii}
    arms = {}
    for g, group in enumerate(GROUPS):
        arms[group+"_zero"] = zero[g*3:(g+1)*3]
        arms[group+"_repaired"] = repaired[g*3:(g+1)*3]
    wanted = ["A_s12_zero", "A_s12_repaired", "A_protocol_zero", "A_protocol_repaired", "B_s12_zero", "B_s12_repaired", "B_protocol_zero"]
    point = {a: np.array([[multilabel_metrics(y, arms[a][s, c], CLASSES)["macro_auroc"] for c in range(len(subsets))]
                          for s in range(len(SEEDS))]) for a in wanted}
    patients, index = np.unique(np.array([r["patient_id"] for r in test["rows"]]), return_inverse=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(bootstrap_arm, [(y, arms[a], index, len(patients), len(SEEDS), len(subsets)) for a in wanted]))
    draws = dict(zip(wanted, results))
    valid = [i for i in range(DRAWS) if all(draws[a][i] is not None for a in wanted)]

    def contrast(left, right, scope):
        cols = scopes[scope]
        est = float(point[left][:, cols].mean()-point[right][:, cols].mean())
        diff = np.array([draws[left][i][:, cols].mean()-draws[right][i][:, cols].mean() for i in valid])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        per_seed = (point[left][:, cols].mean(axis=1)-point[right][:, cols].mean(axis=1)).tolist()
        return dict(contrast="%s - %s" % (left, right), scope=scope, n_subsets=int(cols.sum()), estimate=est,
                    ci95=[float(lo), float(hi)], per_seed=per_seed)

    pairs = [("B_s12_zero", "A_s12_repaired"), ("B_protocol_zero", "A_protocol_repaired"),
             ("B_s12_repaired", "A_s12_repaired"), ("A_s12_repaired", "A_s12_zero")]
    contrasts = [contrast(l, r, s) for l, r in pairs for s in scopes]
    scopes["single_lead"] = counts == 1
    contrasts.append(contrast("B_s12_zero", "A_s12_repaired", "single_lead"))
    by_count = {a: {int(k): float(point[a][:, counts == k].mean()) for k in sorted(set(counts))} for a in wanted}
    by_count_split = {}
    for name, mask in [("two_or_more_limb", n_limb >= 2), ("fewer_than_two_limb", n_limb < 2)]:
        by_count_split[name] = {int(k): dict(n=int(((counts == k) & mask).sum()),
                                             gap=float((point["B_s12_zero"]-point["A_s12_repaired"])[:, (counts == k) & mask].mean()))
                                for k in sorted(set(counts)) if ((counts == k) & mask).any()}
    from math import comb
    gap_k = {k: float((point["B_s12_zero"]-point["A_s12_repaired"])[:, counts == k].mean()) for k in sorted(set(counts))}
    weighted = sum(comb(12, k)*g for k, g in gap_k.items())/sum(comb(12, k) for k in gap_k)
    table = {a: {s: float(point[a][:, cols].mean()) for s, cols in scopes.items()} for a in wanted}
    per_subset = [dict(subset=n, k=int(counts[c]), n_limb=int(n_limb[c]), has_I_II=bool(has_i_ii[c]),
                       A_s12_repaired=float(point["A_s12_repaired"][:, c].mean()), B_s12_zero=float(point["B_s12_zero"][:, c].mean()))
                  for c, (n, _) in enumerate(subsets)]
    write_json(OUT/"per_subset_s12.json", per_subset)
    write_json(OUT/"summary.json", dict(completed=now(), valid_draws=len(valid), n_subsets=len(subsets), recomputed_subsets=len(changed),
                                        scope_sizes={s: int(c.sum()) for s, c in scopes.items()}, macro_auroc=table,
                                        by_retained_count=by_count, gap_by_count_and_limb=by_count_split,
                                        gap_weighted_by_binomial_count=weighted, contrasts=contrasts, weights_updated=False))
    for a, v in table.items():
        print("%-20s" % a, {s: round(x, 4) for s, x in v.items()})
    for c in contrasts:
        print("%-36s %-30s n=%3d %+.4f [%+.4f, %+.4f]" % (c["contrast"], c["scope"], c["n_subsets"], c["estimate"], *c["ci95"]))
    print("gap by k and limb count", json.dumps(by_count_split))
    print("binomial-weighted gap %.4f" % weighted)


if __name__ == "__main__":
    main()
