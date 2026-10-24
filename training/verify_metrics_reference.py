"""Run in the existing base Python with scikit-learn, without importing torch."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
from metrics import binary_metrics, multilabel_metrics

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261023)
largest_error = 0.0
binary_cases = 0
for i in range(300):
    n = int(rng.integers(3, 500))
    target = rng.integers(0, 2, n)
    target[:2] = [0, 1]
    scores = rng.normal(size=n)
    if i % 3 == 0:
        scores = np.round(scores, 1)
    elif i % 3 == 1:
        scores[:] = .5
    result = binary_metrics(target, scores)
    errors = [abs(result["auroc"]-roc_auc_score(target, scores)),
              abs(result["average_precision"]-average_precision_score(target, scores))]
    largest_error = max(largest_error, *errors)
    if max(errors) > 1e-12:
        raise AssertionError("Ranking metric mismatch: "+str(errors))
    binary_cases += 1
for i in range(20):
    target = rng.integers(0, 2, (200, 5))
    scores = np.round(rng.random((200, 5)), 2)
    result = multilabel_metrics(target, scores, list("abcde"))
    errors = [abs(result["macro_auroc"]-roc_auc_score(target, scores, average="macro")),
              abs(result["macro_average_precision"]-average_precision_score(target, scores, average="macro"))]
    largest_error = max(largest_error, *errors)
    if max(errors) > 1e-12:
        raise AssertionError("Macro metric mismatch")
result = dict(checked_on=datetime.now(timezone.utc).isoformat(), status="passed",
              reference="scikit-learn", reference_version=sklearn.__version__,
              numpy_version=np.__version__, binary_cases=binary_cases, multilabel_cases=20,
              ties_included=True, largest_absolute_error=largest_error,
              implementation_sha256=sha256((ROOT/"metrics.py").read_bytes()).hexdigest(),
              source_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html",
                           "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html"])
(ROOT/"metric_reference_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
