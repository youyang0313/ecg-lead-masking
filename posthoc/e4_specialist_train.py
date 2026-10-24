"""E4 (post-hoc training): configuration-specific specialist models.

One model per partial-input configuration (S6, S4, S3, S2) and paired seed. Every training
presentation shows exactly that configuration's leads (the others zero-filled after
standardization); complete input is never shown. Same seeds, initialization, record order,
optimizer, schedule and precision as the formal runs. Each checkpoint is chosen by the
validation macro-AUROC of its own configuration, since that is the only input it is meant
for. No test data is read here.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"training"))
from runtime import (CONFIG, LEADS, PROJECT, PROTOCOL, batch_tensor, configure, epoch_lr, epoch_order,
                     load_development, now, provenance, set_seed, state_hash, write_json)
from experiment import evaluate
from metrics import improves
from model import ECGClassifier
import numpy as np
import torch
from torch import nn

OUT = PROJECT/"experiments/posthoc/e4_specialist_training"
TARGETS = ["S6", "S4", "S3", "S2"]


def train_epoch(model, optimizer, scaler, data, indices, seed, epoch, keep, device):
    model.train()
    order = epoch_order(indices, seed, epoch)
    lr = epoch_lr(epoch)
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    batch_size = PROTOCOL["training"]["batch_size"]
    total, overflow = 0.0, 0
    for start in range(0, len(order), batch_size):
        chosen = order[start:start+batch_size]
        x = batch_tensor(data["x"], chosen, device)*keep
        y = batch_tensor(data["y"], chosen, device)
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
    return dict(epoch=epoch+1, learning_rate=lr, mean_bce=total/len(order), overflow_skipped_steps=overflow)


def run(target, seed, data, device):
    destination = OUT/("SP_%s_%d" % (target, seed))
    destination.mkdir(parents=True, exist_ok=False)
    available = {c["id"]: c["available"] for c in CONFIG["configurations"]}[target]
    keep = torch.tensor([lead in available for lead in LEADS], device=device).view(1, 12, 1)
    set_seed(seed)
    model = ECGClassifier(PROTOCOL).to(device)
    initial = state_hash(model)
    training = PROTOCOL["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"],
                                  betas=tuple(training["betas"]), eps=training["epsilon"])
    scaler = torch.amp.GradScaler("cuda")
    indices = np.arange(len(data["train"]["y"]))
    best, chosen, curve = None, None, []
    for epoch in range(training["epoch_budget"]):
        record = train_epoch(model, optimizer, scaler, data["train"], indices, seed, epoch, keep, device)
        metrics, _, _ = evaluate(model, data["validation"], device)
        score = metrics[target]["macro_auroc"]
        if improves(score, best):
            best = score
            torch.save(dict(model_state=model.state_dict(), epoch=epoch+1, seed=seed, variant="SP_"+target,
                            rule="own configuration", validation_score=score), destination/"selected.pt")
            chosen = dict(epoch=epoch+1, score=score, state_sha256=state_hash(model))
        record["validation_macro_auroc"] = {c: m["macro_auroc"] for c, m in metrics.items()}
        curve.append(record)
        print(json.dumps(dict(variant="SP_"+target, seed=seed, epoch=epoch+1, own=score)), flush=True)
    write_json(destination/"curve.json", curve)
    return dict(target=target, seed=seed, initial_state_sha256=initial, selected=chosen)


def main():
    device = configure()
    data = load_development()
    OUT.mkdir(parents=True, exist_ok=False)
    results = []
    for target in TARGETS:
        for seed in PROTOCOL["training"]["paired_seeds"]:
            results.append(run(target, seed, data, device))
            write_json(OUT/"summary.json", dict(status="running", results=results))
    write_json(OUT/"summary.json", dict(completed=now(), status="complete",
                                        selection="validation macro-AUROC of the model's own configuration",
                                        results=results, environment=provenance()))


if __name__ == "__main__":
    main()
