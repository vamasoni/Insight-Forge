"""Metrics shared by the evaluation suites."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any


def execution_match(pred_rows: Sequence[tuple] | None, gold_rows: Sequence[tuple]) -> bool:
    """BIRD's official EX: the two result *sets* are equal (row order and duplicates ignored)."""
    if pred_rows is None:
        return False
    return set(map(_hashable, pred_rows)) == set(map(_hashable, gold_rows))


def _hashable(row: Any) -> tuple:
    return tuple(tuple(v) if isinstance(v, list) else v for v in row)


def mcnemar(base_correct: Sequence[bool], new_correct: Sequence[bool]) -> dict:
    """Exact McNemar test on paired per-question outcomes.

    b = base right, new wrong (regressions); c = base wrong, new right (fixes).
    Returns the two-sided exact p-value (binomial on the discordant pairs).
    """
    b = sum(1 for x, y in zip(base_correct, new_correct) if x and not y)
    c = sum(1 for x, y in zip(base_correct, new_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"regressions": 0, "fixes": 0, "p_value": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"regressions": b, "fixes": c, "p_value": min(1.0, 2 * p)}


def cohens_kappa(a: Sequence[Any], b: Sequence[Any], weights: str | None = None) -> float:
    """Cohen's kappa for two raters. weights=None (nominal), 'linear' or 'quadratic' (ordinal labels)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    k, n = len(labels), len(a)
    if k == 1:
        return 1.0
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ca, cb = Counter(a), Counter(b)

    def w(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        d = abs(i - j) / (k - 1)
        return d if weights == "linear" else d * d

    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w(i, j) * ca[labels[i]] * cb[labels[j]] for i in range(k) for j in range(k)) / (n * n)
    return 1.0 - po / pe if pe else 1.0


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
