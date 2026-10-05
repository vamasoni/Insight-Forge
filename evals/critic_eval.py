"""Measure how many unsupported claims the critic catches, using controlled perturbations.

For each Olist question we build a correct claim from the gold result, then corrupt it in known ways:
  number_shift       the number is changed by 8-40%
  wrong_citation     cites a different, unrelated query
  missing_citation   cites an evidence id that doesn't exist
  causal             adds causal language to an observational fact
  false_significance says "statistically significant" with no test behind it
  unit_confusion     a share like 0.37 is reported as 3.7%

Recall = fraction of corrupted claims flagged; false-positive rate = clean claims flagged.
Runs deterministic-only by default (free, used in CI); --llm adds the LLM layer.

    python -m evals.critic_eval --db data/olist_synth.duckdb
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evals.olist.run_olist import load_questions
from insightforge.agent.critic import Critic
from insightforge.db import open_database
from insightforge.evidence import EvidenceStore

PERTURBATIONS = ["number_shift", "wrong_citation", "missing_citation", "causal", "false_significance",
                 "unit_confusion"]


def _fmt(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}"


def build_cases(db_path: str, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    db = open_database(db_path)
    qs = load_questions()
    results = {q["id"]: db.execute(q["gold_sql"]) for q in qs}
    cases = []
    for q in qs:
        res = results[q["id"]]
        if not res.rows:
            continue
        # pick a numeric cell and (optionally) its label
        row = res.rows[0]
        num_idx = next((i for i, v in enumerate(row) if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if num_idx is None or row[num_idx] is None:
            continue
        val = float(row[num_idx])
        if abs(val) < 13 and float(val).is_integer():  # the checker ignores tiny integers by design
            continue
        label = next((str(v) for i, v in enumerate(row) if i != num_idx and isinstance(v, str)), None)
        subject = f"For {label}, the" if label else "The"
        clean = f"{subject} result of '{q['question'].rstrip('?')}' is {_fmt(val)}."
        other = rng.choice([x for x in qs if x["id"] != q["id"]])
        base = {"qid": q["id"], "value": val, "other_qid": other["id"]}
        cases.append(base | {"type": "clean", "text": clean, "cite": "own"})
        shifted = val * rng.choice([0.6, 0.75, 0.92, 1.08, 1.25, 1.4])
        if _fmt(shifted) != _fmt(val):  # a shift that rounds to the same text is not a corruption
            cases.append(base | {"type": "number_shift", "text": clean.replace(_fmt(val), _fmt(shifted)), "cite": "own"})
        cases.append(base | {"type": "wrong_citation", "text": clean, "cite": "other"})
        cases.append(base | {"type": "missing_citation", "text": clean, "cite": "missing"})
        cases.append(base | {"type": "causal", "text": clean.rstrip(".") + ", which drives overall sales growth.",
                             "cite": "own"})
        cases.append(base | {"type": "false_significance",
                             "text": clean.rstrip(".") + ", a statistically significant difference.", "cite": "own"})
        if 0 < val < 1:
            cases.append(base | {"type": "unit_confusion", "text": clean.replace(_fmt(val), f"{val * 10:.1f}%"),
                                 "cite": "own"})
    return cases


def run(db_path: str, use_llm: bool = False, llm=None, seed: int = 0) -> dict:
    db = open_database(db_path)
    qs = {q["id"]: q for q in load_questions()}
    critic = Critic(llm, use_llm=use_llm)
    rows = []
    for case in build_cases(db_path, seed):
        store = EvidenceStore(run_id="critic-eval")
        own = store.add_query(sub_question=qs[case["qid"]]["question"], sql=qs[case["qid"]]["gold_sql"],
                              dialect=db.dialect, df=db.execute(qs[case["qid"]]["gold_sql"]).df, truncated=False,
                              elapsed_s=0, attempts=1, tables=[])
        other = store.add_query(sub_question=qs[case["other_qid"]]["question"], sql=qs[case["other_qid"]]["gold_sql"],
                                dialect=db.dialect, df=db.execute(qs[case["other_qid"]]["gold_sql"]).df,
                                truncated=False, elapsed_s=0, attempts=1, tables=[])
        cite = {"own": own.id, "other": other.id, "missing": "Q99"}[case["cite"]]
        claim = store.add_claim(case["text"], [cite])
        critic.review(store)
        rows.append({"type": case["type"], "qid": case["qid"], "text": case["text"], "status": claim.status,
                     "flagged": claim.status != "supported", "notes": claim.critic_notes})
    # wrong_citation can be a true "supported" if the other query happens to contain the same number
    by_type = {}
    for t in ["clean", *PERTURBATIONS]:
        sub = [r for r in rows if r["type"] == t]
        if sub:
            by_type[t] = {"n": len(sub), "flag_rate": round(100 * sum(r["flagged"] for r in sub) / len(sub), 2)}
    corrupted = [r for r in rows if r["type"] != "clean"]
    clean = [r for r in rows if r["type"] == "clean"]
    return {
        "mode": "deterministic+llm" if use_llm else "deterministic",
        "n_clean": len(clean), "n_corrupted": len(corrupted),
        "recall": round(100 * sum(r["flagged"] for r in corrupted) / max(1, len(corrupted)), 2),
        "false_positive_rate": round(100 * sum(r["flagged"] for r in clean) / max(1, len(clean)), 2),
        "by_type": by_type, "misses": [r for r in corrupted if not r["flagged"]][:20],
        "false_positives": [r for r in clean if r["flagged"]][:20],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/olist.duckdb")
    ap.add_argument("--llm", action="store_true", help="also run the LLM critic layer")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    llm = None
    if a.llm:
        from insightforge.llm import get_llm

        llm = get_llm()
    summ = run(a.db, a.llm, llm)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "critic_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("misses", "false_positives")}, indent=2))
    for m in summ["misses"][:5]:
        print("MISS", m["type"], "|", m["text"][:110])
    for m in summ["false_positives"][:5]:
        print("FP  ", m["notes"], "|", m["text"][:110])


if __name__ == "__main__":
    main()
