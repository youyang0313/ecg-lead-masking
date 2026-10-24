"""E13 (post-hoc training): generic input-removal control arm (time cutout).

Identical to arm B except for *how* information is removed. With probability 0.5 a
presentation is complete; otherwise k is drawn uniformly from 1..11 as in arm B and a
contiguous time segment of round(1000 * (12 - k) / 12) samples, at a uniformly random
start, is zeroed in all 12 leads. The removed fraction of the input therefore has the
same distribution as under lead masking, but every lead stays available. Same seeds,
initialization, record order, optimizer, schedule and precision as the formal runs;
checkpoints chosen by S12 validation macro-AUROC only. No test data is read here.
"""
from collections import Counter
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"training"))
from runtime import (PROJECT, PROTOCOL, batch_tensor, configure, epoch_lr, epoch_order, load_development,
                     now, provenance, set_seed, state_hash, write_json)
from experiment import evaluate
from metrics import improves
from model import ECGClassifier
import numpy as np
import torch
from torch import nn

OUT = PROJECT/"experiments/posthoc/e13_cutout_training"
LENGTH = 1000


def cutout_mask(batch_size, generator):
    """[N, 1, T] keep-mask; 50% complete, else one all-lead time segment removed."""
    masks = torch.ones((batch_size, 1, LENGTH), dtype=torch.bool)
    partial = torch.nonzero(torch.rand(batch_size, generator=generator) >= 0.5).flatten()
    removed = []
    for row in partial.tolist():
        k = int(torch.randint(1, 12, (1,), generator=generator))
        width = int(round(LENGTH*(12-k)/12))
        start = int(torch.randint(0, LENGTH-width+1, (1,), generator=generator))
        masks[row, 0, start:start+width] = False
        removed.append(12-k)
    return masks, removed


def train_epoch(model, optimizer, scaler, data, indices, seed, epoch, generator, device):
    model.train()
    order = epoch_order(indices, seed, epoch)
    lr = epoch_lr(epoch)
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    batch_size = PROTOCOL["training"]["batch_size"]
    total, removed_twelfths, overflow = 0.0, Counter(), 0
    for start in range(0, len(order), batch_size):
        chosen = order[start:start+batch_size]
        x = batch_tensor(data["x"], chosen, device)
        y = batch_tensor(data["y"], chosen, device)
        keep, removed = cutout_mask(len(chosen), generator)
        removed_twelfths.update(removed)
        x = x*keep.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.sqrt(sum(p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None))
        overflow += int(not torch.isfinite(norm))
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())*len(chosen)
    return dict(epoch=epoch+1, learning_rate=lr, mean_bce=total/len(order), overflow_skipped_steps=overflow,
                removed_twelfths_histogram={str(k): v for k, v in sorted(removed_twelfths.items())})


def run(seed, data, device):
    destination = OUT/("TC_%d" % seed)
    destination.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    model = ECGClassifier(PROTOCOL).to(device)
    initial = state_hash(model)
    training = PROTOCOL["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"],
                                  betas=tuple(training["betas"]), eps=training["epsilon"])
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator(device="cpu").manual_seed(seed+1000003)
    indices = np.arange(len(data["train"]["y"]))
    best, chosen, curve = None, None, []
    for epoch in range(training["epoch_budget"]):
        record = train_epoch(model, optimizer, scaler, data["train"], indices, seed, epoch, generator, device)
        metrics, _, _ = evaluate(model, data["validation"], device)
        score = metrics["S12"]["macro_auroc"]
        if improves(score, best):
            best = score
            torch.save(dict(model_state=model.state_dict(), epoch=epoch+1, seed=seed, variant="TC", rule="s12",
                            validation_score=score), destination/"s12.pt")
            chosen = dict(epoch=epoch+1, score=score, state_sha256=state_hash(model))
        record["validation_macro_auroc"] = {c: m["macro_auroc"] for c, m in metrics.items()}
        curve.append(record)
        print(json.dumps(dict(variant="TC", seed=seed, epoch=epoch+1, s12=score, s2=metrics["S2"]["macro_auroc"])), flush=True)
    write_json(destination/"curve.json", curve)
    return dict(variant="TC", seed=seed, initial_state_sha256=initial, s12=chosen)


def main():
    device = configure()
    data = load_development()
    OUT.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in PROTOCOL["training"]["paired_seeds"]:
        results.append(run(seed, data, device))
        write_json(OUT/"summary.json", dict(status="running", results=results))
    write_json(OUT/"summary.json", dict(completed=now(), status="complete", selection="S12 validation macro-AUROC only",
                                        results=results, environment=provenance()))


if __name__ == "__main__":
    main()
