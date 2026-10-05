"""LLM-as-judge for report faithfulness, calibrated against your own labels with Cohen's kappa.

Workflow:
  1. Generate reports:            python -m evals.olist.run_olist --mode agent      (saves runs/*/report.json)
  2. Export claims for labelling: python -m evals.judge.judge export --runs runs --out results/labels.csv
  3. Fill the human_score column yourself (0/1/2, same rubric as the judge, see prompts.JUDGE_SYSTEM).
  4. Run the judge:               python -m evals.judge.judge score --labels results/labels.csv
  5. Calibrate:                   python -m evals.judge.judge calibrate --labels results/labels.csv

Label at least ~100 claims, and include the critic's rejected claims so all three scores appear;
kappa on a set that's 95% "2" is meaningless.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.metrics import cohens_kappa
from insightforge import prompts
from insightforge.evidence import EvidenceStore
from insightforge.llm import LLMProvider

FIELDS = ["run_id", "claim_id", "critic_status", "claim", "evidence", "human_score", "judge_score", "judge_reason"]


def export(runs_dir: Path, out: Path, limit: int | None = None) -> int:
    rows = []
    for rep_path in sorted(runs_dir.glob("*/report.json")):
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        store = EvidenceStore.from_dict({"run_id": rep["run_id"], "evidence": rep["evidence"]})
        for c in rep["claims"] + rep["rejected"]:
            ev = "\n\n".join(store.describe(e) for e in c["evidence_ids"]) or "(no evidence cited)"
            rows.append({"run_id": rep["run_id"], "claim_id": c["id"], "critic_status": c["status"],
                         "claim": c["text"], "evidence": ev, "human_score": "", "judge_score": "",
                         "judge_reason": ""})
    rows = rows[:limit] if limit else rows
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def judge_one(llm: LLMProvider, claim: str, evidence: str) -> tuple[int | None, str]:
    try:
        d = llm.complete_json(prompts.JUDGE_SYSTEM, prompts.JUDGE_USER.format(claim=claim, evidence=evidence),
                              role="judge", max_tokens=200)
        score = int(d["score"])
        return (score if score in (0, 1, 2) else None), str(d.get("reason", ""))
    except Exception as e:
        return None, f"judge error: {e}"


def score(labels: Path, llm: LLMProvider, workers: int = 4) -> None:
    rows = list(csv.DictReader(labels.open(encoding="utf-8")))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda r: judge_one(llm, r["claim"], r["evidence"]), rows))
    for r, (s, why) in zip(rows, res):
        r["judge_score"], r["judge_reason"] = ("" if s is None else s), why
    with labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def calibrate(labels: Path) -> dict:
    rows = [r for r in csv.DictReader(labels.open(encoding="utf-8"))
            if str(r["human_score"]).strip() != "" and str(r["judge_score"]).strip() != ""]
    if len(rows) < 2:
        raise SystemExit("need rows with both human_score and judge_score filled in")
    h = [int(r["human_score"]) for r in rows]
    j = [int(r["judge_score"]) for r in rows]
    hb, jb = [x == 2 for x in h], [x == 2 for x in j]  # binary: fully supported vs not
    confusion = Counter(zip(h, j))
    k_w = cohens_kappa(h, j, weights="quadratic")
    out = {
        "n": len(rows), "exact_agreement": round(sum(a == b for a, b in zip(h, j)) / len(rows), 3),
        "kappa_nominal": round(cohens_kappa(h, j), 3), "kappa_quadratic": round(k_w, 3),
        "kappa_binary_supported": round(cohens_kappa(hb, jb), 3),
        "human_label_distribution": dict(Counter(h)), "judge_label_distribution": dict(Counter(j)),
        "confusion_human_x_judge": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
        "verdict": ("judge usable as a CI gate (kappa >= 0.6)" if k_w >= 0.6 else
                    "judge NOT reliable enough to gate CI; revise the rubric/prompt and re-label"),
    }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--runs", default="runs")
    e.add_argument("--out", default="results/labels.csv")
    e.add_argument("--limit", type=int)
    s = sub.add_parser("score")
    s.add_argument("--labels", default="results/labels.csv")
    c = sub.add_parser("calibrate")
    c.add_argument("--labels", default="results/labels.csv")
    c.add_argument("--out", default="results/judge_calibration.json")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(f"exported {export(Path(a.runs), Path(a.out), a.limit)} claims to {a.out}; fill in human_score")
    elif a.cmd == "score":
        from insightforge.llm import get_llm

        score(Path(a.labels), get_llm())
        print(f"judge scores written to {a.labels}")
    else:
        res = calibrate(Path(a.labels))
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
