"""Post-hoc inference for E1/E2/E6/E7 with fixed checkpoints; no weight updates.

Models: the six archived protocol-selected checkpoints plus the six E1 S12-selected ones.
Test configurations: five protocol sets, drop-one/keep-one leads and random subsets per
retained count. E2 test-time variants (standard sets only): 12/k rescaling and BatchNorm
re-estimation from masked training records. BatchNorm input shift is logged per layer.
"""
from pathlib import Path
from itertools import combinations
import copy
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"training"))
from runtime import (CLASSES, CONFIG, LEADS, PREPARED, PROJECT, PROTOCOL, batch_tensor, configure,
                     hash_file, load_development, now, state_hash, write_json)
from model import ECGClassifier
import numpy as np
import torch
from torch import nn

SELECTED = PROJECT/"experiments/train/20261022T183345197538Z/analysis/selected_models.json"
E1 = PROJECT/"experiments/posthoc/e1_replay"
TEST = PREPARED.parent/"test_prepared"
OUT = PROJECT/"experiments/posthoc/inference"
BATCH = PROTOCOL["training"]["batch_size"]
RECALIBRATION_RECORDS = 4096


def configurations():
    standard = [(c["id"], c["available"]) for c in CONFIG["configurations"]]
    drop = [("drop_"+lead, [l for l in LEADS if l != lead]) for lead in LEADS]
    keep = [("keep_"+lead, [lead]) for lead in LEADS]
    rng = np.random.default_rng(20261024)
    subsets = []
    for k in range(1, 12):
        pool = list(combinations(range(12), k))
        picks = rng.choice(len(pool), size=min(20, len(pool)), replace=False)
        subsets += [("rand_k%02d_%02d" % (k, j), [LEADS[i] for i in pool[p]]) for j, p in enumerate(sorted(picks))]
    return standard, standard+drop+keep+subsets


def keep_vector(available, device):
    return torch.tensor([lead in available for lead in LEADS], device=device).view(1, 12, 1)


@torch.inference_mode()
def predict(model, x, available, device, scale=1.0):
    model.eval()
    keep = keep_vector(available, device)
    out = []
    for start in range(0, len(x), BATCH):
        batch = batch_tensor(x, np.arange(start, min(start+BATCH, len(x))), device)*keep*scale
        out.append(torch.sigmoid(model(batch)).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def recalibrated(model, train_x, records, available, device):
    """Copy with BatchNorm running statistics re-estimated (cumulative average) under one mask."""
    clone = copy.deepcopy(model)
    for module in clone.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None
    clone.train()
    keep = keep_vector(available, device)
    for start in range(0, len(records), BATCH):
        clone(batch_tensor(train_x, records[start:start+BATCH], device)*keep)
    clone.eval()
    return clone


@torch.inference_mode()
def bn_shift_summary(model, x, available, device):
    """Mean over channels of |data mean - running mean| / running SD at each BatchNorm input."""
    model.eval()
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm1d)]
    first = {n: torch.zeros(m.num_features, device=device, dtype=torch.float64) for n, m in layers}
    elements = {n: 0 for n, _ in layers}

    def hook(name):
        def record(_module, inputs, _output):
            value = inputs[0].double()
            first[name].add_(value.sum(dim=(0, 2)))
            elements[name] += value.shape[0]*value.shape[2]
        return record

    hooks = [m.register_forward_hook(hook(n)) for n, m in layers]
    keep = keep_vector(available, device)
    for start in range(0, len(x), BATCH):
        model(batch_tensor(x, np.arange(start, min(start+BATCH, len(x))), device)*keep)
    for h in hooks:
        h.remove()
    summary = []
    for n, m in layers:
        mean = first[n]/elements[n]
        shift = (mean-m.running_mean.double()).abs()/torch.sqrt(m.running_var.double()+m.eps)
        summary.append(dict(layer=n, mean_standardized_shift=float(shift.mean())))
    return summary


def load_test():
    ready = json.loads((TEST/"READY.json").read_text(encoding="utf-8"))
    assert ready["status"] == "complete" and ready["audit_sha256"] == hash_file(TEST/"audit.json")
    for entry in ready["outputs"]:
        assert hash_file(TEST/entry["file"]) == entry["sha256"], entry["file"]
    rows = json.loads((TEST/"test.records.json").read_text(encoding="utf-8"))
    return dict(x=np.load(TEST/"test.signals.npy", mmap_mode="r", allow_pickle=False),
                y=np.load(TEST/"test.labels.npy", allow_pickle=False), rows=rows)


def models():
    archived = json.loads(SELECTED.read_text(encoding="utf-8"))["checkpoints"]
    e1 = {(r["seed"], r["arm"]): r for r in json.loads((E1/"summary.json").read_text(encoding="utf-8"))["results"]}
    assert len(e1) == 6 and all(r["protocol_checkpoint_reproduces_archive"] for r in e1.values())
    listing = []
    for item in archived:
        listing.append(dict(key="%s_protocol_%d" % (item["arm"], item["seed"]), arm=item["arm"], seed=item["seed"],
                            rule="protocol", path=item["checkpoint_path"], state_sha256=item["selected_state_sha256"],
                            epoch=item["selected_epoch"]))
    for (seed, arm), r in sorted(e1.items()):
        listing.append(dict(key="%s_s12_%d" % (arm, seed), arm=arm, seed=seed, rule="s12",
                            path=str(E1/("seed_%d_%s" % (seed, arm))/"s12.pt"),
                            state_sha256=r["s12"]["state_sha256"], epoch=r["s12"]["epoch"]))
    return listing


def main():
    device = configure()
    development = load_development()
    test = load_test()
    standard, everything = configurations()
    listing = models()
    records = np.sort(np.random.default_rng(20261025).choice(len(development["train"]["y"]),
                                                             RECALIBRATION_RECORDS, replace=False))
    OUT.mkdir(parents=True, exist_ok=False)
    test_p = np.zeros((len(listing), len(everything), len(test["y"]), 5), dtype=np.float32)
    val_p = np.zeros((len(listing), len(standard), len(development["validation"]["y"]), 5), dtype=np.float32)
    variant_p = np.zeros((len(listing), 2, len(standard), len(test["y"]), 5), dtype=np.float32)
    shifts = []
    for m, item in enumerate(listing):
        model = ECGClassifier(PROTOCOL).to(device)
        model.load_state_dict(torch.load(item["path"], map_location="cpu", weights_only=True)["model_state"])
        assert state_hash(model) == item["state_sha256"], item["key"]
        for c, (_, available) in enumerate(everything):
            test_p[m, c] = predict(model, test["x"], available, device)
        for c, (cid, available) in enumerate(standard):
            val_p[m, c] = predict(model, development["validation"]["x"], available, device)
            variant_p[m, 0, c] = predict(model, test["x"], available, device, scale=12/len(available))
            variant_p[m, 1, c] = predict(recalibrated(model, development["train"]["x"], records, available, device),
                                         test["x"], available, device)
            shifts.append(dict(model=item["key"], configuration=cid,
                               layers=bn_shift_summary(model, test["x"], available, device)))
        assert state_hash(model) == item["state_sha256"], "inference mutated "+item["key"]
        print(json.dumps(dict(stage="model_done", model=item["key"])), flush=True)
    np.savez_compressed(OUT/"predictions.npz", test=test_p, validation=val_p, variants=variant_p,
                        test_targets=test["y"], validation_targets=development["validation"]["y"],
                        patient_ids=np.array([r["patient_id"] for r in test["rows"]]),
                        models=np.array([i["key"] for i in listing]),
                        configurations=np.array([c for c, _ in everything]),
                        standard=np.array([c for c, _ in standard]),
                        variant_names=np.array(["scale_12_over_k", "bn_recalibration"]), classes=np.array(CLASSES))
    write_json(OUT/"manifest.json", dict(completed=now(), models=listing,
                                         configurations={c: a for c, a in everything},
                                         recalibration_training_records=RECALIBRATION_RECORDS,
                                         recalibration_record_seed=20261025, subset_seed=20261024,
                                         weights_updated=False, bn_input_shift=shifts))


if __name__ == "__main__":
    main()
