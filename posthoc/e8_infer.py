"""Post-hoc inference for E8/E9/E10; fixed checkpoints, no weight updates.

Models: A/B S12-selected (E1), B protocol-selected (archived), GN_A, GN_B, HO_B.
E10 replaces zero-filled unavailable leads in the four partial sets with other
representations (identical draws for every model): low/typical-amplitude Gaussian noise,
baseline wander, or the same lead copied from a different test record.
"""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"training"))
sys.path.insert(0, str(HERE))
from runtime import CLASSES, PROJECT, PROTOCOL, batch_tensor, configure, now, state_hash, write_json
from infer import BATCH, E1, SELECTED, configurations, keep_vector, load_test, predict
from e8_train import OUT as TRAINED, build
import numpy as np
import torch

OUT = PROJECT/"experiments/posthoc/e8_e10_inference"
SEEDS = [17, 29, 43]
FILLS = ["noise_0.1", "noise_1.0", "baseline_wander", "other_record"]


def listing():
    archived = {(c["seed"], c["arm"]): c for c in json.loads(SELECTED.read_text(encoding="utf-8"))["checkpoints"]}
    e1 = {(r["seed"], r["arm"]): r for r in json.loads((E1/"summary.json").read_text(encoding="utf-8"))["results"]}
    trained = {(r["variant"], r["seed"]): r for r in json.loads((TRAINED/"summary.json").read_text(encoding="utf-8"))["results"]}
    items = []
    for seed in SEEDS:
        items.append(dict(key="A_s12_%d" % seed, variant="BN", path=str(E1/("seed_%d_A" % seed)/"s12.pt"),
                          state_sha256=e1[(seed, "A")]["s12"]["state_sha256"]))
        items.append(dict(key="B_s12_%d" % seed, variant="BN", path=str(E1/("seed_%d_B" % seed)/"s12.pt"),
                          state_sha256=e1[(seed, "B")]["s12"]["state_sha256"]))
        items.append(dict(key="B_protocol_%d" % seed, variant="BN", path=archived[(seed, "B")]["checkpoint_path"],
                          state_sha256=archived[(seed, "B")]["selected_state_sha256"]))
        for variant in ["GN_A", "GN_B", "HO_B"]:
            items.append(dict(key="%s_%d" % (variant, seed), variant=variant,
                              path=str(TRAINED/("%s_%d" % (variant, seed))/"s12.pt"),
                              state_sha256=trained[(variant, seed)]["s12"]["state_sha256"]))
    return items


def fill_batch(kind, x_cpu, start, config_index, donors):
    """Replacement values for every channel of one CPU batch; only missing leads are used."""
    g = torch.Generator().manual_seed(7919*(config_index+1)+start)
    n = x_cpu.shape[0]
    if kind.startswith("noise"):
        return float(kind.split("_")[1])*torch.randn(x_cpu.shape, generator=g)
    if kind == "baseline_wander":
        t = torch.arange(1000, dtype=torch.float32)/100.0
        freq = 0.1+0.4*torch.rand((n, 12, 1), generator=g)
        phase = 2*np.pi*torch.rand((n, 12, 1), generator=g)
        return torch.sin(2*np.pi*freq*t+phase)
    if kind == "other_record":
        return torch.from_numpy(np.array(donors, dtype=np.float32, copy=True))
    raise ValueError(kind)


@torch.inference_mode()
def predict_fill(model, x, available, kind, config_index, device, permutation):
    model.eval()
    keep = keep_vector(available, device).float()
    out = []
    for start in range(0, len(x), BATCH):
        index = np.arange(start, min(start+BATCH, len(x)))
        clean = batch_tensor(x, index, device)
        donors = x[permutation[index]] if kind == "other_record" else None
        replacement = fill_batch(kind, clean.cpu(), start, config_index, donors)
        batch = clean*keep+replacement.to(device)*(1-keep)
        out.append(torch.sigmoid(model(batch)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    device = configure()
    test = load_test()
    _, everything = configurations()
    partial = [c for c in everything[:5] if c[0] != "S12"]
    items = listing()
    n = len(test["y"])
    # Donor record for "other_record": a fixed derangement-style shift of a random permutation.
    order = np.random.default_rng(20261026).permutation(n)
    permutation = np.empty(n, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    assert not np.any(permutation == np.arange(n))
    OUT.mkdir(parents=True, exist_ok=False)
    test_p = np.zeros((len(items), len(everything), n, 5), dtype=np.float32)
    fill_p = np.zeros((len(items), len(FILLS), len(partial), n, 5), dtype=np.float32)
    for m, item in enumerate(items):
        model = build(item["variant"]).to(device)
        model.load_state_dict(torch.load(item["path"], map_location="cpu", weights_only=True)["model_state"])
        assert state_hash(model) == item["state_sha256"], item["key"]
        for c, (_, available) in enumerate(everything):
            test_p[m, c] = predict(model, test["x"], available, device)
        for f, kind in enumerate(FILLS):
            for c, (_, available) in enumerate(partial):
                fill_p[m, f, c] = predict_fill(model, test["x"], available, kind, c, device, permutation)
        assert state_hash(model) == item["state_sha256"]
        print(json.dumps(dict(stage="model_done", model=item["key"])), flush=True)
    np.savez_compressed(OUT/"predictions.npz", test=test_p, fills=fill_p, test_targets=test["y"],
                        patient_ids=np.array([r["patient_id"] for r in test["rows"]]),
                        models=np.array([i["key"] for i in items]), configurations=np.array([c for c, _ in everything]),
                        partial=np.array([c for c, _ in partial]), fill_names=np.array(FILLS), classes=np.array(CLASSES))
    write_json(OUT/"manifest.json", dict(completed=now(), models=items, fills=FILLS, donor_seed=20261026,
                                         weights_updated=False,
                                         fill_units="standardized training units (1.0 = training SD of that lead)"))


if __name__ == "__main__":
    main()
