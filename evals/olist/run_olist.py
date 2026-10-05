"""End-to-end evaluation on the Olist question set.

Two modes:
  --mode sql    : SQL tool only; relaxed execution match against gold (extra columns allowed)
  --mode agent  : full agent; checks the answer is in the evidence AND stated correctly in the report,
                  plus deterministic contract checks (schema-valid outputs, tool-call correctness).

    python -m evals.olist.run_olist --mode agent --db data/olist.duckdb
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pandas as pd

from evals.metrics import percentile
from insightforge.agent import InsightForgeAgent
from insightforge.agent.critic import extract_numbers, number_supported
from insightforge.config import get_settings
from insightforge.db import open_database
from insightforge.llm import LLMProvider, get_llm, track_usage
from insightforge.tools.sql_tool import SQLTool

QUESTIONS = Path(__file__).with_name("questions.jsonl")


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _norm(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(f"{float(v):.6g}")
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(v).isoformat()
    return str(v)


def relaxed_match(pred: pd.DataFrame, gold_rows: list[tuple]) -> bool:
    """True if some choice of pred columns reproduces the gold result set. Allows extra columns."""
    if not gold_rows:
        return pred.empty
    width = len(gold_rows[0])
    gold = {tuple(_norm(v) for v in r) for r in gold_rows}
    if pred.shape[1] < width or pred.shape[1] > 10:
        return False
    rows = [tuple(_norm(v) for v in r) for r in pred.itertuples(index=False, name=None)]
    for cols in itertools.permutations(range(pred.shape[1]), width):
        if {tuple(r[c] for c in cols) for r in rows} == gold:
            return True
    return False


def scalar_value(gold_rows: list[tuple]) -> float | None:
    try:
        v = gold_rows[0][0]
        return float(v) if v is not None else None
    except (IndexError, TypeError, ValueError):
        return None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-6, 0.005 * abs(b))


def eval_sql(tool: SQLTool, q: dict, gold: list[tuple]) -> dict:
    with track_usage() as u:
        run = tool.run(q["question"], enforce_limit=False)
    ok = run.ok and relaxed_match(run.result.df, gold)
    return {"id": q["id"], "correct": bool(ok), "executed": run.ok, "attempts": run.n_attempts, "sql": run.sql,
            "error": run.error, "cost_usd": u.cost_usd, "latency_s": run.latency_s}


def eval_agent(agent: InsightForgeAgent, q: dict, gold: list[tuple]) -> dict:
    t0 = time.perf_counter()
    rep = agent.ask(q["question"], save=False)
    store = rep.store
    in_evidence, in_report = False, None
    if q["answer_type"] == "scalar":
        g = scalar_value(gold)
        all_ids = list(store.items)
        in_evidence = g is not None and any(_close(n, g) for n in store.numbers_for(all_ids))
        text = " ".join([rep.summary] + [c.text for c in rep.claims])
        in_report = g is not None and any(number_supported(n, [g]) for n in extract_numbers(text))
    else:
        in_evidence = any(relaxed_match(df, gold) for df in store.frames.values())
        if len(gold) == 1 and len(gold[0]) == 1:  # single label answers, e.g. "SP"
            text = " ".join([rep.summary] + [c.text for c in rep.claims]).lower()
            in_report = str(_norm(gold[0][0])).lower().split("t00:00")[0] in text
    schema_errors = [f for f in rep.failures if f.get("question") == "(schema)"]
    sub_failures = [f for f in rep.failures if f.get("question") != "(schema)"]
    needs = [s for s in rep.plan if s.get("needs_analysis")]
    tool_calls_ok = (not sub_failures and len(store.queries()) >= len(rep.plan)
                     and len(store.calcs()) >= len(needs)
                     and all(store.get(e) is not None for c in rep.claims for e in c.evidence_ids))
    m = rep.metrics()
    return {"id": q["id"], "answer_in_evidence": bool(in_evidence), "answer_in_report": in_report,
            "schema_valid": not schema_errors, "tool_calls_ok": bool(tool_calls_ok),
            "claims_total": m["claims_total"], "claims_rejected": m["claims_rejected"],
            "rejected_by_status": m["rejected_by_status"], "revisions": m["revisions"],
            "cost_usd": m["cost_usd"], "latency_s": round(time.perf_counter() - t0, 3), "run_id": rep.run_id}


def summarize(rows: list[dict], mode: str) -> dict:
    n = len(rows)
    pct = lambda k: round(100 * sum(1 for r in rows if r.get(k)) / n, 2) if n else 0.0  # noqa: E731
    base = {"mode": mode, "n": n, "cost_per_question_usd": round(sum(r["cost_usd"] for r in rows) / max(n, 1), 6),
            "latency_p50_s": round(percentile([r["latency_s"] for r in rows], 0.5), 3),
            "latency_p95_s": round(percentile([r["latency_s"] for r in rows], 0.95), 3)}
    if mode == "sql":
        return base | {"ex_relaxed": pct("correct"), "valid_sql_rate": pct("executed"),
                       "per_question": {r["id"]: r["correct"] for r in rows}}
    graded = [r for r in rows if r["answer_in_report"] is not None]
    claims = sum(r["claims_total"] for r in rows)
    return base | {
        "answer_in_evidence": pct("answer_in_evidence"),
        "answer_in_report": round(100 * sum(bool(r["answer_in_report"]) for r in graded) / len(graded), 2)
        if graded else None,
        "schema_valid_rate": pct("schema_valid"), "tool_call_correctness": pct("tool_calls_ok"),
        "claims_total": claims, "critic_rejection_rate": round(100 * sum(r["claims_rejected"] for r in rows)
                                                               / claims, 2) if claims else 0.0,
        "per_question": {r["id"]: r["answer_in_evidence"] for r in rows},
    }


def run(db_path: str, mode: str, llm: LLMProvider, workers: int = 2, limit: int | None = None,
        ids: list[str] | None = None) -> tuple[list[dict], dict]:
    s = get_settings(db_path=db_path)
    db = open_database(db_path)
    qs = load_questions()
    if ids:
        qs = [q for q in qs if q["id"] in ids]
    qs = qs[:limit] if limit else qs
    golds = {q["id"]: db.execute(q["gold_sql"]).rows for q in qs}
    if mode == "sql":
        tool = SQLTool(db, llm, max_rows=None, use_retrieval=s.use_schema_retrieval, max_repairs=s.max_repair_attempts)
        fn = lambda q: eval_sql(tool, q, golds[q["id"]])  # noqa: E731
    else:
        agent = InsightForgeAgent(db, llm, s)
        fn = lambda q: eval_agent(agent, q, golds[q["id"]])  # noqa: E731
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(fn, qs))
    return rows, summarize(rows, mode)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/olist.duckdb")
    ap.add_argument("--mode", choices=["sql", "agent"], default="agent")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    rows, summ = run(a.db, a.mode, get_llm(), a.workers, a.limit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"olist_{a.mode}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    (out / f"olist_{a.mode}_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_question"}, indent=2))


if __name__ == "__main__":
    main()
