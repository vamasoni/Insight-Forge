"""Regression gate: compare an eval summary against a committed baseline and fail the build on a real drop.

A plain "accuracy went down" rule is noisy: on 150 BIRD questions the 95% CI is roughly +/-7 points, so
two identical runs can differ by 2-3 points from LLM nondeterminism alone. The gate therefore uses:
  - a hard floor: fail if the metric drops by more than --max-drop points, and
  - a paired test: fail if McNemar's exact test on per-question outcomes says the new run is worse
    (more regressions than fixes, p < --alpha), even if the drop is below the floor.

    python -m evals.regression --current results/bird_retrieval_repair_summary.json \
        --baseline evals/baselines/bird_retrieval_repair.json --metric ex
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from evals.metrics import mcnemar


def compare(current: dict, baseline: dict, metric: str, max_drop: float, alpha: float,
            higher_is_better: bool = True) -> dict:
    cur, base = current.get(metric), baseline.get(metric)
    if cur is None or base is None:
        return {"metric": metric, "status": "skip", "reason": f"{metric} missing"}
    delta = cur - base if higher_is_better else base - cur
    res = {"metric": metric, "baseline": base, "current": cur, "delta": round(cur - base, 3)}
    pq_c, pq_b = current.get("per_question") or {}, baseline.get("per_question") or {}
    shared = sorted(set(pq_c) & set(pq_b))
    if shared:
        res["paired"] = mcnemar([bool(pq_b[q]) for q in shared], [bool(pq_c[q]) for q in shared])
        res["paired"]["n_shared"] = len(shared)
        if len(shared) < 0.9 * max(len(pq_b), 1):
            res["warning"] = "current run covers a different question subset; paired test is partial"
    failed = delta < -max_drop
    reason = f"dropped {abs(delta):.2f} > {max_drop}" if failed else ""
    p = res.get("paired")
    if p and p["regressions"] > p["fixes"] and p["p_value"] < alpha:
        failed, reason = True, (reason + "; " if reason else "") + (
            f"paired test: {p['regressions']} regressions vs {p['fixes']} fixes, p={p['p_value']:.3g}")
    res["status"] = "fail" if failed else "pass"
    res["reason"] = reason
    return res


def _markdown(results: list[dict], title: str) -> str:
    L = [f"### {title}", "", "| Metric | Baseline | Current | Delta | Paired (reg/fix, p) | Status |",
         "|---|---|---|---|---|---|"]
    for r in results:
        p = r.get("paired")
        ps = f"{p['regressions']}/{p['fixes']}, {p['p_value']:.3g}" if p else "-"
        icon = {"pass": "✅ pass", "fail": "❌ FAIL", "skip": "⏭ skip"}[r["status"]]
        L.append(f"| {r['metric']} | {r.get('baseline', '-')} | {r.get('current', '-')} | {r.get('delta', '-')} "
                 f"| {ps} | {icon} {r.get('reason', '')} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--metric", nargs="+", default=["ex"])
    ap.add_argument("--lower-is-better", nargs="*", default=[], help="metrics where smaller is better")
    ap.add_argument("--max-drop", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--update-baseline", action="store_true")
    a = ap.parse_args(argv)

    cur_p, base_p = Path(a.current), Path(a.baseline)
    current = json.loads(cur_p.read_text())
    if a.update_baseline:
        base_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cur_p, base_p)
        print(f"baseline updated: {base_p}")
        return 0
    if not base_p.exists():
        print(f"::warning::no baseline at {base_p}; passing. Commit one with --update-baseline.")
        return 0
    baseline = json.loads(base_p.read_text())
    results = [compare(current, baseline, m, a.max_drop, a.alpha, m not in a.lower_is_better) for m in a.metric]
    md = _markdown(results, f"Regression gate: {cur_p.name}")
    print(md)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)
    return 1 if any(r["status"] == "fail" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
