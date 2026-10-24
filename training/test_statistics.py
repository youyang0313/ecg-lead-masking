"""Frozen-protocol patient bootstrap, conditional on the trained seed set.

Weighted AUROC exactly represents duplicating all recordings of sampled patients.
Scores are sorted once; equal-score groups receive half credit for tied pairs.
"""
import numpy as np


class WeightedAUROC:
    def __init__(self, targets, probabilities):
        self.y = np.asarray(targets)
        p = np.asarray(probabilities)
        if self.y.ndim != 2 or p.shape[-2:] != self.y.shape:
            raise ValueError('Expected probabilities [..., recordings, labels]')
        if not np.isin(self.y, [0, 1]).all() or not np.isfinite(p).all():
            raise ValueError('Invalid labels/probabilities')
        scores = np.swapaxes(p, -1, -2)
        self.order = np.argsort(scores, axis=-1, kind='stable')
        sorted_scores = np.take_along_axis(scores, self.order, axis=-1)
        self.sorted_y = np.take_along_axis(np.broadcast_to(self.y.T, scores.shape), self.order, axis=-1)
        # Group start/end positions shared by all records with an equal score.
        n = len(self.y)
        positions = np.broadcast_to(np.arange(n), scores.shape)
        starts = np.concatenate([np.ones(scores.shape[:-1]+(1,), dtype=bool),
                                 np.diff(sorted_scores, axis=-1) != 0], axis=-1)
        ends = np.concatenate([np.diff(sorted_scores, axis=-1) != 0,
                               np.ones(scores.shape[:-1]+(1,), dtype=bool)], axis=-1)
        self.before = np.maximum.accumulate(np.where(starts, positions, 0), axis=-1)
        self.after = np.minimum.accumulate(np.where(ends, positions+1, n)[..., ::-1], axis=-1)[..., ::-1]

    def __call__(self, weights):
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self.y),) or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError('Invalid record multiplicities')
        positive = (w[:, None]*self.y).sum(axis=0)
        negative = (w[:, None]*(1-self.y)).sum(axis=0)
        if np.any(positive == 0) or np.any(negative == 0):
            return None
        sorted_w = w[self.order]
        cumulative_negative = np.concatenate([
            np.zeros(self.order.shape[:-1]+(1,)),
            np.cumsum(sorted_w*(1-self.sorted_y), axis=-1)], axis=-1)
        below = np.take_along_axis(cumulative_negative, self.before, axis=-1)
        through = np.take_along_axis(cumulative_negative, self.after, axis=-1)
        return (sorted_w*self.sorted_y*(below+0.5*(through-below))).sum(axis=-1)/(positive*negative)


def patient_weights(patient_index, sampled_patient_indices):
    patient_index = np.asarray(patient_index)
    return np.bincount(sampled_patient_indices, minlength=int(patient_index.max())+1)[patient_index]


def effects(auroc, model_keys, paired_seeds, configurations):
    # auroc shape: models, configurations, labels. Never ensemble predictions.
    macro = auroc.mean(axis=-1)
    differences = np.stack([macro[model_keys.index((s, 'B'))]-macro[model_keys.index((s, 'A'))]
                            for s in paired_seeds])
    partial = [configurations.index(c) for c in ['S6', 'S4', 'S3', 'S2']]
    return dict(primary=float(differences[:, partial].mean()),
                complete=float(differences[:, configurations.index('S12')].mean()),
                per_seed_partial=differences[:, partial].mean(axis=1),
                per_configuration=differences.mean(axis=0))


def self_check():
    """Independent explicit-pair reference with ties, weights and patient clusters."""
    rng = np.random.default_rng(9917)
    cases, maximum = 0, 0.0
    for n in [9, 23, 51]:
        y = rng.integers(0, 2, size=(n, 5))
        y[0], y[1] = 0, 1
        scores = rng.integers(0, 7, size=(2, 3, n, 5))/6
        statistic = WeightedAUROC(y, scores)
        for _ in range(12):
            w = rng.integers(0, 4, size=n)
            w[:2] = 1
            actual = statistic(w)
            expected = np.empty((2, 3, 5))
            for a in range(2):
                for c in range(3):
                    for label in range(5):
                        pos, neg = y[:, label] == 1, y[:, label] == 0
                        p, q = scores[a, c, pos, label], scores[a, c, neg, label]
                        credit = (p[:, None] > q).astype(float)+0.5*(p[:, None] == q)
                        pairs = w[pos, None]*w[None, neg]
                        expected[a, c, label] = (pairs*credit).sum()/pairs.sum()
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
            assert np.allclose(actual, expected, rtol=0, atol=1e-14)
            cases += actual.size
        assert statistic(np.zeros(n)) is None
    index = np.array([0, 0, 1, 2, 2, 2])
    assert np.array_equal(patient_weights(index, [2, 0, 2]), [1, 1, 0, 2, 2, 2])
    # Explicit repeated-record expansion must equal multiplicity weighting.
    weights = patient_weights(index, [2, 0, 2])
    y = np.array([[0], [1], [0], [1], [0], [1]])
    p = np.array([0.2, 0.2, 0.8, 0.8, 0.4, 0.4])[:, None]
    expanded = np.repeat(np.arange(len(index)), weights)
    assert np.allclose(WeightedAUROC(y, p)(weights),
                       WeightedAUROC(y[expanded], p[expanded])(np.ones(len(expanded))))
    return dict(status='passed', explicit_pair_auroc_comparisons=cases,
                max_absolute_error=maximum, patient_cluster_multiplicity_checked=True,
                equal_scores_and_zero_weights_checked=True, undefined_target_checked=True)


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(), indent=2))
