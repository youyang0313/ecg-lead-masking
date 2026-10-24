"""E8/E9 (post-hoc training): normalization ablation and held-out lead family.

E8  GN_A / GN_B: identical protocol, but every BatchNorm1d replaced by GroupNorm(8, C).
    Convolutions receive the same seeded initialization as the BN models.
E9  HO_B: arm-B masking, rejecting partial masks that retain only limb leads (I..aVF),
    so S6, S2 and every other limb-only input are never seen in training.
Checkpoints are chosen by S12 validation macro-AUROC only, so held-out configurations
never influence selection. No test data is read here.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"training"))
from runtime import PROJECT, PROTOCOL, configure, load_development, now, provenance, set_seed, state_hash, write_json
import experiment
from experiment import evaluate, train_epoch
from metrics import improves
from model import ECGClassifier
import numpy as np
import torch
from torch import nn

OUT = PROJECT/"experiments/posthoc/e8_e9_training"
GROUPS = 8
LIMB = 6  # canonical indices 0..5 are I, II, III, aVR, aVL, aVF


def to_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.GroupNorm(GROUPS, child.num_features))
        else:
            to_groupnorm(child)
    return module


def build(variant):
    model = ECGClassifier(PROTOCOL)
    return to_groupnorm(model) if variant.startswith("GN") else model


def limb_holdout_mask(batch_size, generator):
    """Arm-B distribution conditioned on retaining at least one precordial lead when partial."""
    masks = torch.ones((batch_size, 12), dtype=torch.bool)
    partial = torch.nonzero(torch.rand(batch_size, generator=generator) >= 0.5).flatten()
    for row in partial.tolist():
        while True:
            count = int(torch.randint(1, 12, (1,), generator=generator))
            keep = torch.randperm(12, generator=generator)[:count]
            if (keep >= LIMB).any():
                break
        masks[row] = False
        masks[row, keep] = True
    return masks


def run(variant, seed, data, device):
    arm = "A" if variant == "GN_A" else "B"
    destination = OUT/("%s_%d" % (variant, seed))
    destination.mkdir(parents=True, exist_ok=False)
    experiment.sample_lead_mask = limb_holdout_mask if variant == "HO_B" else original_sampler
    set_seed(seed)
    model = build(variant).to(device)
    training = PROTOCOL["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"],
                                  betas=tuple(training["betas"]), eps=training["epsilon"])
    scaler = torch.amp.GradScaler("cuda")
    mask_generator = torch.Generator(device="cpu").manual_seed(seed+1000003)
    indices = np.arange(len(data["train"]["y"]))
    best, chosen, curve = None, None, []
    for epoch in range(training["epoch_budget"]):
        record = train_epoch(model, optimizer, scaler, data["train"], indices,
                             seed, epoch, arm, mask_generator, device)
        metrics, _, _ = evaluate(model, data["validation"], device)
        score = metrics["S12"]["macro_auroc"]
        if improves(score, best):
            best = score
            torch.save(dict(model_state=model.state_dict(), epoch=epoch+1, seed=seed, variant=variant,
                            rule="s12", validation_score=score), destination/"s12.pt")
            chosen = dict(epoch=epoch+1, score=score, state_sha256=state_hash(model))
        curve.append(dict(epoch=epoch+1, mean_bce=record["mean_bce"],
                          retained_lead_count_histogram=record["retained_lead_count_histogram"],
                          validation_macro_auroc={c: m["macro_auroc"] for c, m in metrics.items()}))
        print(json.dumps(dict(variant=variant, seed=seed, epoch=epoch+1, s12=score,
                              s2=metrics["S2"]["macro_auroc"])), flush=True)
    write_json(destination/"curve.json", curve)
    return dict(variant=variant, seed=seed, arm=arm, s12=chosen,
                parameters=sum(p.numel() for p in model.parameters()))


original_sampler = experiment.sample_lead_mask


def main():
    device = configure()
    data = load_development()
    OUT.mkdir(parents=True, exist_ok=False)
    results = []
    for variant in ["GN_A", "GN_B", "HO_B"]:
        for seed in PROTOCOL["training"]["paired_seeds"]:
            results.append(run(variant, seed, data, device))
            write_json(OUT/"summary.json", dict(status="running", results=results))
    experiment.sample_lead_mask = original_sampler
    write_json(OUT/"summary.json", dict(completed=now(), status="complete", groups=GROUPS,
                                        selection="S12 validation macro-AUROC only", results=results,
                                        environment=provenance()))


if __name__ == "__main__":
    main()
