"""E4 evaluation: configuration-specific specialists against fixed and repaired models.

Each specialist (SP) is evaluated only on its own configuration (zero-fill). Comparators are
read from earlier archives: zero-filled A and B (E1/E2 inference) and the E12 test-time
repair (limb-lead derivation + BatchNorm re-estimation) of A and B, under both selection
rules. Paired patient bootstrap, 2,000 draws, conditional on the trained seeds.
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
from e4_specialist_train import OUT as TRAINED, TARGETS
from model import ECGClassifier
import numpy as np
import torch

OUT = PROJECT/"experiments/posthoc/e4_specialist_eval"
SEEDS = [17, 29, 43]
DRAWS = 2000
ARMS = ["SP", "A_s12_zero", "A_protocol_zero", "A_s12_repaired", "A_protocol_repaired",
        "B_s12_zero", "B_protocol_zero", "B_s12_repaired", "B_protocol_repaired"]


def main():
    device = configure()
    test = load_test()
    y = test["y"]
    configs = {c["id"]: c["available"] for c in CONFIG["configurations"]}
    trained = {(r["target"], r["seed"]): r for r in json.loads((TRAINED/"summary.json").read_text(encoding="utf-8"))["results"]}
    probs = np.zeros((len(ARMS), len(SEEDS), len(TARGETS), len(y), 5), dtype=np.float32)
    for c, target in enumerate(TARGETS):
        for s, seed in enumerate(SEEDS):
            model = ECGClassifier(PROTOCOL).to(device)
            model.load_state_dict(torch.load(TRAINED/("SP_%s_%d" % (target, seed))/"selected.pt", map_location="cpu",
                                             weights_only=True)["model_state"])
            assert state_hash(model) == trained[(target, seed)]["selected"]["state_sha256"]
            probs[0, s, c] = predict(model, test["x"], configs[target], device)
    archive = np.load(PROJECT/"experiments/posthoc/inference/predictions.npz", allow_pickle=False)
    names, cfgs = list(archive["models"]), list(archive["configurations"])
    assert np.array_equal(archive["test_targets"], y)
    e12 = np.load(PROJECT/"experiments/posthoc/e12_combined/predictions.npz", allow_pickle=False)
    names12, conds12, cfgs12 = list(e12["models"]), list(e12["conditions"]), list(e12["configurations"])
    assert np.array_equal(e12["targets"], y)
    for g, arm in enumerate(ARMS[1:], start=1):
        group, condition = arm.rsplit("_", 1)
        for s, seed in enumerate(SEEDS):
            key = "%s_%d" % (group, seed)
            for c, target in enumerate(TARGETS):
                probs[g, s, c] = (archive["test"][names.index(key), cfgs.index(target)] if condition == "zero" else
                                  e12["probabilities"][names12.index(key), conds12.index("derived_bn_recal"), cfgs12.index(target)])
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(OUT/"predictions.npz", specialist=probs[0], targets=y, configurations=np.array(TARGETS))
    point = np.array([[[multilabel_metrics(y, probs[g, s, c], CLASSES)["macro_auroc"] for c in range(len(TARGETS))]
                       for s in range(len(SEEDS))] for g in range(len(ARMS))])
    statistic = WeightedAUROC(y, probs.reshape(-1, *y.shape))
    patients, index = np.unique(np.array([r["patient_id"] for r in test["rows"]]), return_inverse=True)
    rng = np.random.default_rng(20261023)
    draws = []
    for _ in range(DRAWS):
        value = statistic(patient_weights(index, rng.integers(0, len(patients), len(patients))))
        if value is not None:
            draws.append(value.mean(axis=-1).reshape(len(ARMS), len(SEEDS), len(TARGETS)))
    draws = np.array(draws)

    def contrast(left, right, cols):
        l, r = ARMS.index(left), ARMS.index(right)
        est = float(point[l][:, cols].mean()-point[r][:, cols].mean())
        diff = draws[:, l][:, :, cols].mean(axis=(1, 2))-draws[:, r][:, :, cols].mean(axis=(1, 2))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        return dict(contrast="%s - %s" % (left, right), scope="partial" if len(cols) > 1 else TARGETS[cols[0]],
                    estimate=est, ci95=[float(lo), float(hi)],
                    per_seed=(point[l][:, cols].mean(axis=1)-point[r][:, cols].mean(axis=1)).tolist())

    every = list(range(len(TARGETS)))
    contrasts = [contrast("SP", other, every) for other in ARMS[1:]]
    for c in every:
        contrasts += [contrast("SP", other, [c]) for other in ["B_s12_zero", "B_protocol_zero", "A_s12_repaired"]]
    table = {arm: dict(per_config={t: float(point[g][:, i].mean()) for i, t in enumerate(TARGETS)},
                       partial_mean=float(point[g].mean()), partial_seed_sd=float(point[g].mean(axis=1).std(ddof=1)))
             for g, arm in enumerate(ARMS)}
    write_json(OUT/"summary.json", dict(completed=now(), valid_draws=len(draws),
                                        selected_epochs={"%s_%d" % k: v["selected"]["epoch"] for k, v in trained.items()},
                                        macro_auroc=table, contrasts=contrasts))
    for arm, v in table.items():
        print("%-20s" % arm, " ".join("%s %.4f" % (t, x) for t, x in v["per_config"].items()), "partial %.4f" % v["partial_mean"])
    for c in contrasts:
        print("%-32s %-8s %+.4f [%+.4f, %+.4f] seeds %s" % (c["contrast"], c["scope"], c["estimate"], *c["ci95"],
                                                         [round(x, 3) for x in c["per_seed"]]))


if __name__ == "__main__":
    main()
