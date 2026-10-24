"""E1 (post-hoc): replay the six formal runs and add a complete-input-only checkpoint.

Deterministic replay of experiment.run_arm. Each run keeps two validation-selected
checkpoints: the original protocol rule (must reproduce the archived state hash) and
S12-only macro-AUROC (earliest epoch on ties). No test data is read here.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"training"))
from runtime import PROJECT, PROTOCOL, configure, load_development, now, provenance, set_seed, state_hash, write_json
from experiment import evaluate, train_epoch
from metrics import improves, selection_score
from model import ECGClassifier
import numpy as np
import torch

SELECTED = PROJECT/"experiments/train/20261022T183345197538Z/analysis/selected_models.json"
OUT = PROJECT/"experiments/posthoc/e1_replay"


def replay(arm, seed, data, device):
    destination = OUT/("seed_"+str(seed)+"_"+arm)
    destination.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    model = ECGClassifier(PROTOCOL).to(device)
    training = PROTOCOL["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"],
                                  betas=tuple(training["betas"]), eps=training["epsilon"])
    scaler = torch.amp.GradScaler("cuda")
    mask_generator = torch.Generator(device="cpu").manual_seed(seed+1000003)
    indices = np.arange(len(data["train"]["y"]))
    best = dict(protocol=None, s12=None)
    chosen = {}
    curve = []
    for epoch in range(training["epoch_budget"]):
        record = train_epoch(model, optimizer, scaler, data["train"], indices,
                             seed, epoch, arm, mask_generator, device)
        metrics, _, _ = evaluate(model, data["validation"], device)
        scores = dict(protocol=selection_score(metrics, PROTOCOL["selection"]["weights"]),
                      s12=metrics["S12"]["macro_auroc"])
        for rule, score in scores.items():
            if improves(score, best[rule]):
                best[rule] = score
                torch.save(dict(model_state=model.state_dict(), epoch=epoch+1, seed=seed, arm=arm,
                                rule=rule, validation_score=score), destination/(rule+".pt"))
                chosen[rule] = dict(epoch=epoch+1, score=score, state_sha256=state_hash(model))
        curve.append(dict(epoch=epoch+1, mean_bce=record["mean_bce"],
                          validation_macro_auroc={c: m["macro_auroc"] for c, m in metrics.items()}))
        print(json.dumps(dict(arm=arm, seed=seed, epoch=epoch+1, **scores)), flush=True)
    write_json(destination/"curve.json", curve)
    return chosen


def main():
    device = configure()
    data = load_development()
    original = {(c["seed"], c["arm"]): c for c in json.loads(SELECTED.read_text(encoding="utf-8"))["checkpoints"]}
    OUT.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in PROTOCOL["training"]["paired_seeds"]:
        for arm in ["A", "B"]:
            chosen = replay(arm, seed, data, device)
            reference = original[(seed, arm)]
            reproduced = (chosen["protocol"]["state_sha256"] == reference["selected_state_sha256"]
                          and chosen["protocol"]["epoch"] == reference["selected_epoch"])
            results.append(dict(seed=seed, arm=arm, protocol=chosen["protocol"], s12=chosen["s12"],
                                protocol_checkpoint_reproduces_archive=reproduced))
            write_json(OUT/"summary.json", dict(started=now(), results=results, environment=provenance()))
            if not reproduced:
                raise AssertionError("Replay diverged from archived checkpoint: seed %d arm %s" % (seed, arm))
    write_json(OUT/"summary.json", dict(completed=now(), status="complete", results=results, environment=provenance()))


if __name__ == "__main__":
    main()
