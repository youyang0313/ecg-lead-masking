"""Loading, model code and formatting, kept out of the lecture file on purpose.

Lectrace traces only the lecture file, so everything defined here runs without producing a
step: stepping through the talk never lands inside a data loader or a forward hook. The model
class is the archived one from `training/model.py`, and the checkpoints are the archived
S12-selected models of seed 17 (`experiments/posthoc/e1_replay`).
"""
import ast
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ASSETS = HERE/"assets"
sys.path.insert(0, str(PROJECT/"training"))
from model import ECGClassifier  # noqa: E402  (the archived architecture)

PROTOCOL = json.loads((PROJECT/"report/protocol_v1.json").read_text(encoding="utf-8"))
CONFIG = json.loads((PROJECT/"report/lead_configurations.json").read_text(encoding="utf-8"))
LEADS = CONFIG["canonical_names_for_this_document"]
CLASSES = PROTOCOL["dataset"]["classes"]
CONFIGS = {c["id"]: c["available"] for c in CONFIG["configurations"]}
LIMB = ["I", "II", "III", "aVR", "aVL", "aVF"]
# Each limb lead as a combination of (I, II): Einthoven and Goldberger.
COEF = {"I": (1.0, 0.0), "II": (0.0, 1.0), "III": (-1.0, 1.0),
        "aVR": (-0.5, -0.5), "aVL": (1.0, -0.5), "aVF": (-0.5, 1.0)}
PREPARED = PROJECT/"data/ptbxl_1_0_3/prepared"
TEST = PROJECT/"data/ptbxl_1_0_3/test_prepared"
SCALER = json.loads((PREPARED/"scaler.json").read_text(encoding="utf-8"))
MU = torch.tensor(SCALER["mean_mV"], dtype=torch.float32).view(1, 12, 1)
SD = torch.tensor(SCALER["applied_scale_mV"], dtype=torch.float32).view(1, 12, 1)
BATCH = PROTOCOL["training"]["batch_size"]
torch.set_grad_enabled(False)


def results() -> dict:
    return json.loads((ASSETS/"results.json").read_text(encoding="utf-8"))


def per_subset() -> list:
    return json.loads((ASSETS/"per_subset_s12.json").read_text(encoding="utf-8"))


def fig(name: str) -> str:
    return str(ASSETS/name)


def d3(x: float) -> str:
    return f"{x:.3f}"


def signed(x: float) -> str:
    return f"{x:+.3f}"


def ci(c: list) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def pct(x: float) -> str:
    return f"{100*x:.0f}%"


# ------------------------------------------------------------------ data
def test_set():
    rows = json.loads((TEST/"test.records.json").read_text(encoding="utf-8"))
    x = np.load(TEST/"test.signals.npy", mmap_mode="r")
    y = np.load(TEST/"test.labels.npy")
    return x, y, rows


def demo_record_index(rows) -> int:
    """First test record (lowest ECG id) whose only diagnostic finding is inferior MI at
    likelihood 100. Chosen from metadata alone, before any model output is looked at."""
    with open(PROJECT/"data_review/metadata/scp_statements.csv", encoding="utf-8") as f:
        subclass = {r[""]: r["diagnostic_subclass"] for r in csv.DictReader(f) if r["diagnostic"] == "1.0"}
    with open(PROJECT/"data_review/metadata/ptbxl_database.csv", encoding="utf-8") as f:
        codes = {int(float(r["ecg_id"])): ast.literal_eval(r["scp_codes"]) for r in csv.DictReader(f)}
    for i in sorted(range(len(rows)), key=lambda k: rows[k]["ecg_id"]):
        diag = {c: v for c, v in codes[rows[i]["ecg_id"]].items() if c in subclass}
        if diag and all(subclass[c] == "IMI" and v == 100.0 for c, v in diag.items()):
            return i
    raise LookupError("no inferior-MI record")


def train_signals():
    return np.load(PREPARED/"train.signals.npy", mmap_mode="r")


def recalibration_records(n_train: int) -> np.ndarray:
    """The same 4,096 training records as in the paper (seed 20261025)."""
    return np.sort(np.random.default_rng(20261025).choice(n_train, 4096, replace=False))


# ------------------------------------------------------------------ inputs
def physical(z: torch.Tensor) -> torch.Tensor:
    """Standardized input back to mV."""
    return z*SD+MU


def zero_fill(z: torch.Tensor, available: list) -> torch.Tensor:
    keep = torch.tensor([lead in available for lead in LEADS], dtype=torch.float32).view(1, 12, 1)
    return z*keep


def derive_limb_leads(z: torch.Tensor, available: list) -> torch.Tensor:
    """Zero-fill, then compute every missing limb lead from the available ones (least squares
    for I and II; exact for any two limb leads). Measured leads keep their values."""
    out = zero_fill(z, available)
    limb = [l for l in available if l in LIMB]
    if len(limb) < 2:
        return out
    c = torch.tensor([COEF[l] for l in limb])
    phys = physical(z)
    basis = torch.einsum("kl,nlt->nkt", torch.linalg.pinv(c), torch.stack([phys[:, LEADS.index(l)] for l in limb], 1))
    for lead in LIMB:
        if lead not in available:
            k = LEADS.index(lead)
            a, b = COEF[lead]
            out[:, k] = (a*basis[:, 0]+b*basis[:, 1]-MU[0, k])/SD[0, k]
    return out


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate-truth)/np.linalg.norm(truth))


# ------------------------------------------------------------------ models
def load_model(arm: str, seed: int = 17) -> nn.Module:
    model = ECGClassifier(PROTOCOL)
    path = PROJECT/f"experiments/posthoc/e1_replay/seed_{seed}_{arm}/s12.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"])
    return model.eval()


def predict(model: nn.Module, x, available: list, derive: bool = False) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(x), 256):
        z = torch.from_numpy(np.array(x[s:s+256], dtype=np.float32))
        z = derive_limb_leads(z, available) if derive else zero_fill(z, available)
        out.append(torch.sigmoid(model(z)).numpy())
    return np.concatenate(out)


def bn_shift(model: nn.Module, x, available: list) -> list:
    """For every BatchNorm layer after the stem: mean over channels of
    |test-data mean - running mean| / running SD, under the given lead set."""
    layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)][1:]
    sums = [torch.zeros(m.num_features, dtype=torch.float64) for m in layers]
    elements = [0]*len(layers)

    def accumulate(i):
        def hook(mod, inp, out):  # a forward hook must return None, or it replaces the output
            sums[i].add_(inp[0].double().sum(dim=(0, 2)))
            elements[i] += inp[0].shape[0]*inp[0].shape[2]
        return hook

    hooks = [m.register_forward_hook(accumulate(i)) for i, m in enumerate(layers)]
    model.eval()
    for s in range(0, len(x), 256):
        model(zero_fill(torch.from_numpy(np.array(x[s:s+256], dtype=np.float32)), available))
    for h in hooks:
        h.remove()
    shifts = []
    for m, total, n in zip(layers, sums, elements):
        mean = total/n
        shifts.append(float(((mean-m.running_mean.double()).abs()/torch.sqrt(m.running_var.double()+m.eps)).mean()))
    return [round(v, 2) for v in shifts]


def reestimate_batchnorm(model: nn.Module, train_x, available: list, derive: bool = True) -> nn.Module:
    """Copy of the model with BatchNorm statistics re-estimated on the paper's 4,096 training
    records, presented with the same lead set (and limb-lead derivation). No weight changes."""
    clone = copy.deepcopy(model)
    for m in clone.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None
    clone.train()
    records = recalibration_records(len(train_x))
    for s in range(0, len(records), BATCH):
        z = torch.from_numpy(np.array(train_x[records[s:s+BATCH]], dtype=np.float32))
        clone(derive_limb_leads(z, available) if derive else zero_fill(z, available))
    return clone.eval()


# ------------------------------------------------------------------ metrics
def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j+1 < len(p) and sorted_p[j+1] == sorted_p[i]:
            j += 1
        ranks[order[i:j+1]] = (i+j)/2+1
        i = j+1
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg))


def macro_auroc(y: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean([auroc(y[:, j], probs[:, j]) for j in range(y.shape[1])]))


# ------------------------------------------------------------------ charts
def lead_chart(series: dict, seconds: float = 2.5, title: str = "") -> dict:
    """Vega-Lite chart of named 1-D signals in mV over the first `seconds`. Layers are drawn in
    the order given, so a thick first trace stays visible under thinner later ones."""
    n = int(seconds*100)
    names = list(series)
    colors = ["#9e9e9e", "#0072B2", "#D55E00", "#009E73"][:len(names)]
    widths = [4, 1.6, 1.6, 1.6][:len(names)]
    dashes = [[1, 0], [6, 3], [1, 0], [2, 2]][:len(names)]
    layers = []
    for name, width, dash in zip(names, widths, dashes):
        layers.append({
            "data": {"values": [{"time (s)": t/100, "mV": float(series[name][t]), "trace": name} for t in range(n)]},
            "mark": {"type": "line", "strokeWidth": width, "strokeDash": dash},
            "encoding": {"x": {"field": "time (s)", "type": "quantitative"},
                         "y": {"field": "mV", "type": "quantitative"},
                         "color": {"field": "trace", "type": "nominal",
                                   "scale": {"domain": names, "range": colors}}},
        })
    return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": title, "width": 620, "height": 220, "layer": layers}


def bn_shift_chart(shifts: dict) -> dict:
    """Vega-Lite grouped bars: BatchNorm input shift per layer for several model/input pairs."""
    values = [{"layer": i+1, "shift (running SDs)": v, "model": name}
              for name, row in shifts.items() for i, v in enumerate(row)]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": 620, "height": 220,
        "data": {"values": values},
        "mark": {"type": "line", "point": True},
        "encoding": {"x": {"field": "layer", "type": "ordinal", "title": "BatchNorm layer after the stem"},
                     "y": {"field": "shift (running SDs)", "type": "quantitative"},
                     "color": {"field": "model", "type": "nominal"}},
    }


# ------------------------------------------------------------------ slide tables
def ladder_rows() -> list:
    """Repair ladder: mean macro-AUROC over S6/S4/S3/S2, three seeds, S12-only selection."""
    lad = results()["ladder"]
    rows = [("A, zero-fill", "A_zero"), ("A, GroupNorm retrained (E7)", "A_groupnorm"),
            ("A, derived limb leads (E10)", "A_derived"), ("A, BatchNorm re-estimated (E2)", "A_bn"),
            ("A, derived + re-estimated (E11)", "A_derived_bn"), ("B, zero-fill", "B_zero"),
            ("Specialist per configuration (E3)", "specialists")]
    return [{"model": name, "partial-input macro-AUROC": d3(lad[key]["partial_mean"])} for name, key in rows]


def gap_closed() -> dict:
    """Share of the A-to-B gap closed by each repair, from the three-seed means."""
    lad = results()["ladder"]
    a0, b0 = lad["A_zero"]["partial_mean"], lad["B_zero"]["partial_mean"]
    return {step: round((lad[key]["partial_mean"]-a0)/(b0-a0), 2)
            for step, key in [("derive limb leads", "A_derived"), ("re-estimate BatchNorm", "A_bn"),
                              ("both", "A_derived_bn")]}


def config_rows() -> list:
    r = results()
    return [{"input": c, "A": d3(v["A"]), "B": d3(v["B"]), "B - A": signed(v["diff"]), "95% interval": ci(v["ci"])}
            for c, v in r["by_config"].items()]


def config_caption() -> str:
    r = results()
    return (f"Independent test: {r['cohort']['records']:,} records, {r['cohort']['patients']:,} patients; "
            f"primary effect {signed(r['primary']['estimate'])} {ci(r['primary']['ci'])}.")


def gaps_by_limb_leads() -> dict:
    """B minus repaired A for each of the 207 subsets, grouped by how many limb leads remain."""
    rows = per_subset()
    return {"two or more limb leads": [s["B_s12_zero"]-s["A_s12_repaired"] for s in rows if s["n_limb"] >= 2],
            "fewer than two": [s["B_s12_zero"]-s["A_s12_repaired"] for s in rows if s["n_limb"] < 2]}
