"""Experiment provenance and prepared-data access. No test data path is exposed."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import platform
import random
import sys

# Required before CUDA context creation for deterministic CUDA BLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PREPARED = PROJECT/"data"/"ptbxl_1_0_3"/"prepared"
PROTOCOL_PATH = PROJECT/"report"/"protocol_v1.json"
CONFIG_PATH = PROJECT/"report"/"lead_configurations.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CLASSES = PROTOCOL["dataset"]["classes"]
LEADS = CONFIG["canonical_names_for_this_document"]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def hash_file(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def state_hash(model):
    h = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def parameter_hash(model):
    """Trainable weights only; excludes BatchNorm running-statistic buffers."""
    h = sha256()
    for name, tensor in sorted(model.named_parameters()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure():
    if not torch.cuda.is_available():
        raise RuntimeError("This run requires the verified CUDA environment")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda:0")


def epoch_order(indices, seed, epoch):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, 20261023])).permutation(indices)


def epoch_lr(epoch):
    training = PROTOCOL["training"]
    budget = training["epoch_budget"]
    if not 0 <= epoch < budget:
        raise ValueError("Epoch outside the fixed protocol budget")
    hi, lo = training["learning_rate"], training["minimum_learning_rate"]
    return lo+0.5*(hi-lo)*(1+math.cos(math.pi*epoch/(budget-1)))


def load_development():
    ready = json.loads((PREPARED/"READY.json").read_text(encoding="utf-8"))
    audit_path = PROJECT/"data_review"/"preprocessing_audit.json"
    if ready["status"] != "complete" or hash_file(audit_path) != ready["audit_sha256"]:
        raise ValueError("Prepared-data ready marker/audit mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for relative, expected in audit["input_sha256"].items():
        if hash_file(PROJECT/relative) != expected:
            raise ValueError("Dataset/design input changed since preprocessing: "+relative)
    for item in ready["outputs"]:
        if hash_file(PREPARED/item["file"]) != item["sha256"]:
            raise ValueError("Prepared-data hash mismatch: "+item["file"])
    splits = {}
    for split in ["train", "validation"]:
        x = np.load(PREPARED/(split+".signals.npy"), mmap_mode="r", allow_pickle=False)
        y = np.load(PREPARED/(split+".labels.npy"), allow_pickle=False)
        rows = json.loads((PREPARED/(split+".records.json")).read_text(encoding="utf-8"))
        if x.shape != (len(rows), 12, 1000) or y.shape != (len(rows), 5):
            raise ValueError("Prepared shape mismatch")
        if not np.array_equal(y, [[c in r["labels"] for c in CLASSES] for r in rows]):
            raise ValueError("Prepared labels/row alignment mismatch")
        splits[split] = dict(x=x, y=y, rows=rows)
    if {r["patient_id"] for r in splits["train"]["rows"]} & {r["patient_id"] for r in splits["validation"]["rows"]}:
        raise ValueError("Patient overlap")
    return splits


def batch_tensor(array, indices, device):
    # Copy protects the read-only memory map; masks never mutate saved arrays.
    cpu = torch.from_numpy(np.array(array[indices], dtype=np.float32, copy=True))
    return cpu.pin_memory().to(device, non_blocking=True)


def provenance():
    return dict(recorded_on=now(), python=sys.version, executable=sys.executable,
                platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
                cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                deterministic_algorithms=True, cudnn_benchmark=False, tf32=False,
                training_precision="float16 CUDA autocast + dynamic GradScaler; float32 parameters",
                evaluation_precision="float32, autocast disabled",
                batch_size=PROTOCOL["training"]["batch_size"],
                input_and_protocol_sha256={"protocol_v1.json": hash_file(PROTOCOL_PATH),
                                           "lead_configurations.json": hash_file(CONFIG_PATH),
                                           "prepared_READY.json": hash_file(PREPARED/"READY.json")},
                source_sha256={p.name: hash_file(p) for p in sorted(ROOT.glob("*.py"))},
                test_waveforms_used=False, test_annotations_parsed=False)
