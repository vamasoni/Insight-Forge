"""Statistical tests the analysis step can run. Each returns a JSON-able dict that the critic can check.

Every result carries `method`, `n`, and, where relevant, `p_value` and an `interpretation_limits` note,
so the report can't quietly upgrade an association into a causal claim.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

OBSERVATIONAL = "Observational data: shows association, not causation."


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"column {col!r} not in result; available: {list(df.columns)}")
    return pd.to_numeric(df[col], errors="coerce")


def describe(df: pd.DataFrame, value: str, group: str | None = None) -> dict[str, Any]:
    s = _num(df, value)
    if group:
        g = df.assign(_v=s).dropna(subset=["_v"]).groupby(group)["_v"]
        table = g.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
        return {"method": "describe", "value": value, "group": group, "n": int(g.size().sum()),
                "by_group": table.round(4).to_dict(orient="records")}
    s = s.dropna()
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"method": "describe", "value": value, "n": int(len(s)), "mean": float(s.mean()),
            "std": float(s.std()), "min": float(s.min()), "max": float(s.max()),
            "quantiles": {str(k): float(v) for k, v in q.items()}}


def t_test(df: pd.DataFrame, value: str, group: str, a: str | None = None, b: str | None = None) -> dict[str, Any]:
    """Welch's t-test (unequal variances) between two groups, with Cohen's d and a 95% CI of the difference."""
    d = df.assign(_v=_num(df, value)).dropna(subset=["_v", group])
    counts = d[group].value_counts()
    note = None
    if a is None or b is None:
        if len(counts) < 2:
            raise ValueError(f"t_test needs two groups in {group!r}, found {len(counts)}")
        a, b = counts.index[:2].tolist()
        if len(counts) > 2:
            note = f"{len(counts)} groups present; compared the two largest ({a!r}, {b!r})."
    x = d.loc[d[group].astype(str) == str(a), "_v"]
    y = d.loc[d[group].astype(str) == str(b), "_v"]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("each group needs at least 2 observations")
    t, p = stats.ttest_ind(x, y, equal_var=False)
    vx, vy, nx, ny = x.var(ddof=1), y.var(ddof=1), len(x), len(y)
    se = math.sqrt(vx / nx + vy / ny)
    dof = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    diff = float(x.mean() - y.mean())
    tcrit = stats.t.ppf(0.975, dof)
    pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)) or float("nan")
    out = {"method": "welch_t_test", "value": value, "group": group, "groups": [str(a), str(b)],
           "n": int(nx + ny), "n_a": int(nx), "n_b": int(ny), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
           "difference": diff, "ci95": [diff - tcrit * se, diff + tcrit * se], "t": float(t), "p_value": float(p),
           "cohens_d": diff / pooled if pooled else None, "significant_at_0.05": bool(p < 0.05),
           "interpretation_limits": OBSERVATIONAL}
    if note:
        out["note"] = note
    return out


def chi_square(df: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    ct = pd.crosstab(df[a], df[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("chi-square needs at least a 2x2 table")
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = int(ct.to_numpy().sum())
    v = math.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    small = float((expected < 5).mean())
    out = {"method": "chi_square_independence", "a": a, "b": b, "n": n, "chi2": float(chi2), "dof": int(dof),
           "p_value": float(p), "cramers_v": v, "significant_at_0.05": bool(p < 0.05),
           "table": {str(k): {str(kk): int(vv) for kk, vv in row.items()} for k, row in ct.to_dict("index").items()},
           "interpretation_limits": OBSERVATIONAL}
    if small > 0.2:
        out["warning"] = f"{small:.0%} of expected counts < 5; chi-square approximation unreliable."
    return out


def correlation(df: pd.DataFrame, x: str, y: str, method: str = "spearman") -> dict[str, Any]:
    d = pd.DataFrame({"x": _num(df, x), "y": _num(df, y)}).dropna()
    n = len(d)
    if n < 3:
        raise ValueError("correlation needs at least 3 rows")
    r, p = (stats.pearsonr if method == "pearson" else stats.spearmanr)(d.x, d.y)
    r = float(r)
    ci = None
    if n > 3 and abs(r) < 1:
        z, se = np.arctanh(r), 1 / math.sqrt(n - 3)
        ci = [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
    return {"method": f"{method}_correlation", "x": x, "y": y, "n": n, "r": r, "ci95": ci, "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05), "interpretation_limits": OBSERVATIONAL}


def regression(df: pd.DataFrame, y: str, x: list[str] | str) -> dict[str, Any]:
    """OLS with categorical predictors one-hot encoded; robust (HC3) standard errors."""
    import statsmodels.formula.api as smf

    xs = [x] if isinstance(x, str) else list(x)
    cols = [y, *xs]
    d = df[cols].copy()
    d[y] = pd.to_numeric(d[y], errors="coerce")
    terms = []
    for c in xs:
        num = pd.to_numeric(d[c], errors="coerce")
        if num.notna().mean() > 0.95:
            d[c] = num
            terms.append(f"Q('{c}')")
        else:
            terms.append(f"C(Q('{c}'))")
    d = d.dropna()
    if len(d) < len(xs) + 5:
        raise ValueError("not enough rows for regression")
    model = smf.ols(f"Q('{y}') ~ " + " + ".join(terms), data=d).fit(cov_type="HC3")
    coefs = {k.replace("Q('", "").replace("')", ""): {"coef": float(v), "p_value": float(model.pvalues[k]),
                                                      "ci95": [float(a) for a in model.conf_int().loc[k]]}
             for k, v in model.params.items()}
    return {"method": "ols_regression_hc3", "y": y, "x": xs, "n": int(model.nobs), "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj), "coefficients": coefs,
            "interpretation_limits": OBSERVATIONAL + " Coefficients are conditional associations."}


def anomalies(df: pd.DataFrame, value: str, time: str | None = None, z: float = 3.5) -> dict[str, Any]:
    """Robust z-score (median/MAD) outliers."""
    s = _num(df, value)
    med = s.median()
    mad = (s - med).abs().median()
    if not mad or np.isnan(mad):
        return {"method": "robust_zscore", "value": value, "n": int(s.notna().sum()), "anomalies": [],
                "note": "MAD is zero; no spread to detect anomalies against."}
    rz = 0.6745 * (s - med) / mad
    flagged = df.assign(robust_z=rz.round(3)).loc[rz.abs() > z]
    if time and time in df.columns:
        flagged = flagged.sort_values(time)
    rows = flagged.head(50)
    return {"method": "robust_zscore", "value": value, "threshold": z, "n": int(s.notna().sum()),
            "median": float(med), "mad": float(mad), "n_anomalies": int(len(flagged)),
            "anomalies": rows.astype(object).where(rows.notna(), None).to_dict(orient="records")}


REGISTRY = {"describe": describe, "t_test": t_test, "chi_square": chi_square, "correlation": correlation,
            "regression": regression, "anomalies": anomalies}
