"""BIRD dev-set execution-accuracy evaluation, with ablations for schema retrieval and self-repair.

Layout expected (the official dev.zip, unzipped):
    data/bird/dev/dev.json
    data/bird/dev/dev_databases/<db_id>/<db_id>.sqlite
    data/bird/dev/dev_databases/<db_id>/database_description/*.csv

Examples:
    python -m evals.bird.run_bird --limit 150 --configs retrieval+repair
    python -m evals.bird.run_bird --configs full full+repair retrieval retrieval+repair --limit 300
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from evals.metrics import execution_match, percentile, wilson_ci
from insightforge.config import get_settings
from insightforge.db import SQLiteDatabase
from insightforge.llm import LLMProvider, get_llm, track_usage
from insightforge.tools.sql_tool import SQLTool

CONFIGS = {
    "full": dict(use_retrieval=False, max_repairs=0, repair_on_empty=False),
    "full+repair": dict(use_retrieval=False, max_repairs=2, repair_on_empty=True),
    "retrieval": dict(use_retrieval=True, max_repairs=0, repair_on_empty=False),
    "retrieval+repair": dict(use_retrieval=True, max_repairs=2, repair_on_empty=True),
}


@dataclass
class BirdItem:
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str


def load_bird(root: Path) -> list[BirdItem]:
    data = json.loads((root / "dev.json").read_text(encoding="utf-8"))
    return [BirdItem(d.get("question_id", i), d["db_id"], d["question"], d.get("evidence", ""), d["SQL"],
                     d.get("difficulty", "unknown")) for i, d in enumerate(data)]


def stratified_sample(items: list[BirdItem], n: int | None, seed: int) -> list[BirdItem]:
    """Same seed + n -> same subset, with difficulty proportions preserved. Keeps CI runs comparable."""
    if not n or n >= len(items):
        return items
    rng = random.Random(seed)
    by_diff: dict[str, list[BirdItem]] = defaultdict(list)
    for it in items:
        by_diff[it.difficulty].append(it)
    out = []
    for diff, group in sorted(by_diff.items()):
        k = round(n * len(group) / len(items))
        out += rng.sample(group, min(k, len(group)))
    return sorted(out, key=lambda x: x.question_id)[:n]


class BirdRunner:
    def __init__(self, root: Path, llm: LLMProvider, config: str, top_tables: int = 4, timeout_s: float = 30.0):
        self.root = root
        self.llm = llm
        self.cfg = CONFIGS[config]
        self.config = config
        self.top_tables = top_tables
        self.timeout_s = timeout_s
        self._tools: dict[str, SQLTool] = {}
        self._lock = threading.Lock()
        self.gold_cache_dir = root / ".gold_cache"
        self.gold_cache_dir.mkdir(exist_ok=True)

    def db(self, db_id: str) -> SQLiteDatabase:
        return self.tool(db_id).db  # type: ignore[return-value]

    def tool(self, db_id: str) -> SQLTool:
        with self._lock:
            if db_id not in self._tools:
                base = self.root / "dev_databases" / db_id
                db = SQLiteDatabase(str(base / f"{db_id}.sqlite"), descriptions_dir=str(base / "database_description"))
                db.schema()
                self._tools[db_id] = SQLTool(db, self.llm, max_rows=None, top_tables=self.top_tables,
                                             strict_columns=True, query_timeout_s=self.timeout_s, **self.cfg)
            return self._tools[db_id]

    def gold(self, it: BirdItem) -> list[tuple] | None:
        f = self.gold_cache_dir / f"{it.question_id}.pkl"
        if f.exists():
            return pickle.loads(f.read_bytes())
        try:
            rows = self.db(it.db_id).execute(it.gold_sql, timeout_s=self.timeout_s * 2).rows
        except Exception:
            return None
        f.write_bytes(pickle.dumps(rows))
        return rows

    def run_one(self, it: BirdItem) -> dict:
        tool = self.tool(it.db_id)
        t0 = time.perf_counter()
        with track_usage() as u:
            run = tool.run(it.question, hint=it.evidence, enforce_limit=False)
        gold = self.gold(it)
        correct = gold is not None and run.ok and execution_match(run.result.rows, gold)
        first_fail = next((a.stage for a in run.attempts if a.stage != "ok"), None)
        return {
            "question_id": it.question_id, "db_id": it.db_id, "difficulty": it.difficulty, "correct": bool(correct),
            "executed": run.ok, "gold_error": gold is None, "attempts": run.n_attempts, "repaired": run.repaired,
            "first_failure_stage": first_fail, "pred_sql": run.sql, "gold_sql": it.gold_sql,
            "tables": run.retrieval.tables if run.retrieval else None,
            "gold_tables_recalled": _tables_recalled(it.gold_sql, run.retrieval.tables if run.retrieval else []),
            "error": run.error, "latency_s": round(time.perf_counter() - t0, 3), **_usage(u),
        }


def _tables_recalled(gold_sql: str, retrieved: list[str]) -> bool | None:
    """Did retrieval include every table the gold query uses? (retrieval recall, per question)"""
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return None
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    need = {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes}
    return need <= {r.lower() for r in retrieved}


def _usage(u) -> dict:
    return {"cost_usd": round(u.cost_usd, 6), "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "llm_calls": u.calls}


def summarize(rows: list[dict], config: str, model: str) -> dict:
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    by_diff = {}
    for d in sorted({r["difficulty"] for r in rows}):
        sub = [r for r in rows if r["difficulty"] == d]
        by_diff[d] = {"n": len(sub), "ex": round(100 * sum(r["correct"] for r in sub) / len(sub), 2)}
    needed_repair = [r for r in rows if r["attempts"] > 1]
    recall = [r["gold_tables_recalled"] for r in rows if r["gold_tables_recalled"] is not None]
    lo, hi = wilson_ci(k, n)
    return {
        "config": config, "model": model, "n": n, "ex": round(100 * k / n, 2) if n else 0.0,
        "ex_ci95": [round(100 * lo, 2), round(100 * hi, 2)], "by_difficulty": by_diff,
        "valid_sql_rate": round(100 * sum(r["executed"] for r in rows) / n, 2) if n else 0.0,
        "needed_repair_rate": round(100 * len(needed_repair) / n, 2) if n else 0.0,
        "repair_success_rate": round(100 * sum(r["correct"] for r in needed_repair) / len(needed_repair), 2)
        if needed_repair else None,
        "table_recall": round(100 * sum(recall) / len(recall), 2) if recall else None,
        "first_failure_stages": dict(Counter(r["first_failure_stage"] for r in rows if r["first_failure_stage"])),
        "gold_errors": sum(r["gold_error"] for r in rows),
        "cost_per_query_usd": round(sum(r["cost_usd"] for r in rows) / n, 6) if n else 0.0,
        "latency_p50_s": round(percentile([r["latency_s"] for r in rows], 0.5), 3),
        "latency_p95_s": round(percentile([r["latency_s"] for r in rows], 0.95), 3),
        "avg_llm_calls": round(sum(r["llm_calls"] for r in rows) / n, 2) if n else 0.0,
        "per_question": {str(r["question_id"]): r["correct"] for r in rows},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/bird/dev")
    ap.add_argument("--configs", nargs="+", default=["retrieval+repair"], choices=list(CONFIGS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--top-tables", type=int, default=4)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = stratified_sample(load_bird(root), a.limit, a.seed)
    s = get_settings()
    llm = get_llm(s)
    print(f"BIRD dev: {len(items)} questions, model={s.provider_label}")

    for cfg in a.configs:
        runner = BirdRunner(root, llm, cfg, top_tables=a.top_tables)
        rows, t0 = [], time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(runner.run_one, it): it for it in items}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    rows.append(f.result())
                except Exception as e:  # never lose a whole run to one bad question
                    it = futs[f]
                    rows.append({"question_id": it.question_id, "db_id": it.db_id, "difficulty": it.difficulty,
                                 "correct": False, "executed": False, "gold_error": False, "attempts": 0,
                                 "repaired": False, "first_failure_stage": "crash", "error": repr(e),
                                 "gold_tables_recalled": None, "latency_s": 0, "cost_usd": 0, "input_tokens": 0,
                                 "output_tokens": 0, "llm_calls": 0})
                if i % 25 == 0:
                    acc = 100 * sum(r["correct"] for r in rows) / len(rows)
                    print(f"  [{cfg}] {i}/{len(items)}  running EX={acc:.1f}%  ({time.time() - t0:.0f}s)")
        rows.sort(key=lambda r: r["question_id"])
        tag = cfg.replace("+", "_")
        with (out / f"bird_{tag}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summ = summarize(rows, cfg, s.provider_label)
        (out / f"bird_{tag}_summary.json").write_text(json.dumps(summ, indent=2))
        print(f"[{cfg}] EX={summ['ex']}% (95% CI {summ['ex_ci95']}), by difficulty {summ['by_difficulty']}, "
              f"repair success={summ['repair_success_rate']}, table recall={summ['table_recall']}, "
              f"${summ['cost_per_query_usd']}/query, p50={summ['latency_p50_s']}s")


if __name__ == "__main__":
    main()
