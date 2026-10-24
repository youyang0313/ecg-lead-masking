"""E13 evaluation: time-cutout control arm (TC) against S12-selected A and B on the test set.

TC predictions are computed here for the five protocol configurations (zero-fill, as in the
primary analysis); A and B S12-selected predictions are read from the E1/E2 inference
archive. Paired patient bootstrap, 2,000 draws, conditional on the trained seeds.
"""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"training"))
sys.path.insert(0, str(HERE))
from runtime import CLASSES, CONFIG, PROJECT, PROTOCOL, configure, now, state_hash, write_json
from metrics import multilabel_metrics
from test_statistics import WeightedAUROC, patient_weights
from infer import load_test, predict
from e13_cutout_train import OUT as TRAINED
from model import ECGClassifier
import numpy as np
import torch

OUT = PROJECT/"experiments/posthoc/e13_cutout_eval"
SEEDS = [17, 29, 43]
STANDARD = ["S12", "S6", "S4", "S3", "S2"]
PARTIAL = STANDARD[1:]
DRAWS = 2000


def main():
    device = configure()
    test = load_test()
    y = test["y"]
    configs = {c["id"]: c["available"] for c in CONFIG["configurations"]}
    trained = {r["seed"]: r for r in json.loads((TRAINED/"summary.json").read_text(encoding="utf-8"))["results"]}
    tc = np.zeros((len(SEEDS), len(STANDARD), len(y), 5), dtype=np.float32)
    for s, seed in enumerate(SEEDS):
        model = ECGClassifier(PROTOCOL).to(device)
        model.load_state_dict(torch.load(TRAINED/("TC_%d" % seed)/"s12.pt", map_location="cpu", weights_only=True)["model_state"])
        assert state_hash(model) == trained[seed]["s12"]["state_sha256"]
        for c, cid in enumerate(STANDARD):
            tc[s, c] = predict(model, test["x"], configs[cid], device)
        print(json.dumps(dict(stage="model_done", seed=seed)), flush=True)
    archive = np.load(PROJECT/"experiments/posthoc/inference/predictions.npz", allow_pickle=False)
    names, cfgs = list(archive["models"]), list(archive["configurations"])
    assert np.array_equal(archive["test_targets"], y)
    idx = [cfgs.index(c) for c in STANDARD]
    a = np.stack([archive["test"][names.index("A_s12_%d" % s)][idx] for s in SEEDS])
    b = np.stack([archive["test"][names.index("B_s12_%d" % s)][idx] for s in SEEDS])
    probs = np.stack([a, tc, b])  # [arm, seed, config, N, 5]
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(OUT/"predictions.npz", tc=tc, targets=y)
    point = np.array([[[multilabel_metrics(y, probs[g, s, c], CLASSES)["macro_auroc"] for c in range(len(STANDARD))]
                       for s in range(len(SEEDS))] for g in range(3)])
    statistic = WeightedAUROC(y, probs.reshape(-1, *y.shape))
    patients, index = np.unique(np.array([r["patient_id"] for r in test["rows"]]), return_inverse=True)
    rng = np.random.default_rng(20261023)
    draws = []
    for _ in range(DRAWS):
        value = statistic(patient_weights(index, rng.integers(0, len(patients), len(patients))))
        if value is not None:
            draws.append(value.mean(axis=-1).reshape(3, len(SEEDS), len(STANDARD)))
    draws = np.array(draws)
    arms = ["A_s12", "TC", "B_s12"]
    part = [STANDARD.index(c) for c in PARTIAL]

    def contrast(name, left, right, scope):
        l, r = arms.index(left), arms.index(right)
        cols = part if scope == "partial" else [0]
        est = float(point[l][:, cols].mean()-point[r][:, cols].mean())
        diff = draws[:, l][:, :, cols].mean(axis=(1, 2))-draws[:, r][:, :, cols].mean(axis=(1, 2))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        return dict(contrast=name, scope=scope, estimate=est, ci95=[float(lo), float(hi)],
                    per_seed=(point[l][:, cols].mean(axis=1)-point[r][:, cols].mean(axis=1)).tolist())

    contrasts = [contrast("TC - A(S12-sel)", "TC", "A_s12", "partial"), contrast("B(S12-sel) - TC", "B_s12", "TC", "partial"),
                 contrast("TC - A(S12-sel)", "TC", "A_s12", "complete"), contrast("B(S12-sel) - TC", "B_s12", "TC", "complete")]
    table = {arm: dict(per_config={c: float(point[g][:, i].mean()) for i, c in enumerate(STANDARD)},
                       partial_mean=float(point[g][:, part].mean()), partial_seed_sd=float(point[g][:, part].mean(axis=1).std(ddof=1)))
             for g, arm in enumerate(arms)}
    write_json(OUT/"summary.json", dict(completed=now(), valid_draws=len(draws), selected_epochs={s: trained[s]["s12"]["epoch"] for s in SEEDS},
                                        macro_auroc=table, contrasts=contrasts))
    for c in contrasts:
        print("%-22s %-8s %+.4f [%+.4f, %+.4f] seeds %s" % (c["contrast"], c["scope"], c["estimate"], *c["ci95"], [round(x, 3) for x in c["per_seed"]]))
    for arm, v in table.items():
        print("%-6s" % arm, {c: round(x, 4) for c, x in v["per_config"].items()}, "partial %.4f" % v["partial_mean"])


if __name__ == "__main__":
    main()
