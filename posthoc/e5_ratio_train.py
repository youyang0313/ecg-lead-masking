"""E5 (post-hoc training): sensitivity to the masking ratio.

Identical to arm B except for the probability that a presentation is masked: p in
{0.25, 0.75, 1.0} instead of 0.5. Masked presentations draw the retained count uniformly
from 1..11 and the subset uniformly given the count, as in arm B. Same seeds,
initialization, record order, mask-stream seed, optimizer, schedule and precision as the
formal runs; checkpoints chosen by S12 validation macro-AUROC only. No test data is read.
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

OUT = PROJECT/"experiments/posthoc/e5_ratio_training"
RATIOS = [0.25, 0.75, 1.0]


def sample_mask(batch_size, ratio, generator):
    """Masked with probability `ratio`; else complete. Same sampler as arm B otherwise."""
    masks = torch.ones((batch_size, 12), dtype=torch.bool)
    partial = torch.nonzero(torch.rand(batch_size, generator=generator) >= 1-ratio).flatten()
    counts = torch.randint(1, 12, (len(partial),), generator=generator)
    for row, count in zip(partial.tolist(), counts.tolist()):
        masks[row] = False
        masks[row, torch.randperm(12, generator=generator)[:count]] = True
    return masks


def train_epoch(model, optimizer, scaler, data, indices, seed, epoch, ratio, generator, device):
    model.train()
    order = epoch_order(indices, seed, epoch)
    lr = epoch_lr(epoch)
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    batch_size = PROTOCOL["training"]["batch_size"]
    total, counts, overflow = 0.0, Counter(), 0
    for start in range(0, len(order), batch_size):
        chosen = order[start:start+batch_size]
        x = batch_tensor(data["x"], chosen, device)
        y = batch_tensor(data["y"], chosen, device)
        keep = sample_mask(len(chosen), ratio, generator)
        counts.update(keep.sum(dim=1).tolist())
        x = x*keep.to(device, non_blocking=True).unsqueeze(-1)
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
                retained_count_histogram={str(k): v for k, v in sorted(counts.items())})


def run(ratio, seed, data, device):
    destination = OUT/("R%03d_%d" % (round(ratio*100), seed))
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
        record = train_epoch(model, optimizer, scaler, data["train"], indices, seed, epoch, ratio, generator, device)
        metrics, _, _ = evaluate(model, data["validation"], device)
        score = metrics["S12"]["macro_auroc"]
        if improves(score, best):
            best = score
            torch.save(dict(model_state=model.state_dict(), epoch=epoch+1, seed=seed, ratio=ratio, rule="s12",
                            validation_score=score), destination/"s12.pt")
            chosen = dict(epoch=epoch+1, score=score, state_sha256=state_hash(model))
        record["validation_macro_auroc"] = {c: m["macro_auroc"] for c, m in metrics.items()}
        curve.append(record)
        print(json.dumps(dict(ratio=ratio, seed=seed, epoch=epoch+1, s12=score, s2=metrics["S2"]["macro_auroc"])), flush=True)
    write_json(destination/"curve.json", curve)
    return dict(ratio=ratio, seed=seed, initial_state_sha256=initial, s12=chosen)


def main():
    device = configure()
    data = load_development()
    OUT.mkdir(parents=True, exist_ok=False)
    results = []
    for ratio in RATIOS:
        for seed in PROTOCOL["training"]["paired_seeds"]:
            results.append(run(ratio, seed, data, device))
            write_json(OUT/"summary.json", dict(status="running", results=results))
    write_json(OUT/"summary.json", dict(completed=now(), status="complete", selection="S12 validation macro-AUROC only",
                                        results=results, environment=provenance()))


if __name__ == "__main__":
    main()
