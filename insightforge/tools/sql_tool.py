"""SQL tool: schema retrieval -> generation -> validation -> execution -> repair loop."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd

from insightforge import prompts
from insightforge.db import Database, QueryResult
from insightforge.evidence.store import EvidenceStore, QueryEvidence
from insightforge.llm import LLMProvider
from insightforge.retrieval import RetrievalResult, SchemaRetriever
from insightforge.tools.sql_validator import SQLValidator
from insightforge.tracing import get_tracer

_FLOAT_CAST = {"sqlite": "CAST(x AS REAL)", "duckdb": "CAST(x AS DOUBLE) or x * 1.0"}


@dataclass
class Attempt:
    sql: str
    stage: str  # ok | parse | statement | readonly | function | table | explain | execute | empty | no_sql
    error: str = ""


@dataclass
class SQLRun:
    question: str
    ok: bool
    sql: str | None
    result: QueryResult | None
    attempts: list[Attempt] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    evidence: QueryEvidence | None = None
    error: str = ""
    latency_s: float = 0.0

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def repaired(self) -> bool:
        return self.ok and self.n_attempts > 1

    def to_dict(self) -> dict:
        return {
            "question": self.question, "ok": self.ok, "sql": self.sql, "error": self.error,
            "attempts": [a.__dict__ for a in self.attempts], "latency_s": round(self.latency_s, 3),
            "tables": self.retrieval.tables if self.retrieval else None,
            "row_count": self.result.row_count if self.result else None,
            "evidence_id": self.evidence.id if self.evidence else None,
        }


class SQLTool:
    def __init__(self, db: Database, llm: LLMProvider, *, max_rows: int | None = 5000, max_repairs: int = 2,
                 use_retrieval: bool = True, top_tables: int = 4, repair_on_empty: bool = True,
                 strict_columns: bool = False, query_timeout_s: float = 30.0):
        self.db = db
        self.llm = llm
        self.validator = SQLValidator(db, max_rows=max_rows)
        self.retriever = SchemaRetriever(db)
        self.max_repairs = max_repairs
        self.use_retrieval = use_retrieval
        self.top_tables = top_tables
        self.repair_on_empty = repair_on_empty
        self.strict_columns = strict_columns
        self.query_timeout_s = query_timeout_s
        self.max_rows = max_rows

    def _system(self) -> str:
        d = self.db.dialect
        return prompts.SQL_SYSTEM.format(
            dialect="SQLite" if d == "sqlite" else "DuckDB", float_cast=_FLOAT_CAST.get(d, "CAST(x AS REAL)"),
            column_rule=prompts.STRICT_COLUMNS if self.strict_columns else prompts.LOOSE_COLUMNS)

    def run(self, question: str, *, hint: str = "", store: EvidenceStore | None = None,
            enforce_limit: bool = True, max_rows: int | None = None) -> SQLRun:
        """max_rows overrides the default row cap for this call (analysis inputs need full data).
        Thread-safe: no per-call state is stored on the tool."""
        cap = max_rows if max_rows is not None else self.max_rows
        tracer = get_tracer()
        t0 = time.perf_counter()
        with tracer.span("sql_tool", as_type="tool", input={"question": question, "hint": hint}) as span:
            with tracer.span("schema_retrieval", as_type="retriever", input=question) as rs:
                retrieval = (self.retriever.retrieve(question, hint, top_k=self.top_tables) if self.use_retrieval
                             else self.retriever.full())
                rs["output"] = {"tables": retrieval.tables, "values": retrieval.matched_values[:10]}

            user = prompts.SQL_USER.format(
                schema=self.db.schema().render(retrieval.tables),
                value_hints=("\nValue hints:\n" + retrieval.value_hints() + "\n") if retrieval.matched_values else "",
                hint_block=f"Hint: {hint}\n" if hint else "",
                question=question)
            messages = [{"role": "user", "content": user}]
            system = self._system()
            run = SQLRun(question, False, None, None, retrieval=retrieval)
            empty_retry_used = False
            first_empty: tuple[str, QueryResult] | None = None

            for i in range(self.max_repairs + 1):
                reply = self.llm.complete(system, messages, role="sql" if i == 0 else "sql_repair", max_tokens=800).text
                sql = extract_sql(reply)
                messages.append({"role": "assistant", "content": reply})
                if not sql:
                    run.attempts.append(Attempt("", "no_sql", "no SQL found in reply"))
                    messages.append({"role": "user", "content": "Reply with the query in a ```sql``` block."})
                    continue

                v = self.validator.validate(sql, enforce_limit=enforce_limit, max_rows=cap)
                if not v.ok:
                    run.attempts.append(Attempt(sql, v.stage, v.error_message))
                    messages.append({"role": "user", "content": prompts.SQL_REPAIR.format(
                        sql=v.sql, stage=v.stage, error=v.error_message)})
                    continue
                try:
                    result = self.db.execute(v.sql, timeout_s=self.query_timeout_s)
                except Exception as e:
                    err = str(e).strip().replace("\n", " ")[:400]
                    run.attempts.append(Attempt(v.sql, "execute", err))
                    messages.append({"role": "user", "content": prompts.SQL_REPAIR.format(
                        sql=v.sql, stage="execution", error=err)})
                    continue

                if result.row_count == 0 and self.repair_on_empty and not empty_retry_used and i < self.max_repairs:
                    empty_retry_used = True
                    first_empty = (v.sql, result)
                    run.attempts.append(Attempt(v.sql, "empty", "query returned 0 rows"))
                    messages.append({"role": "user", "content": prompts.SQL_EMPTY_REPAIR.format(sql=v.sql)})
                    continue

                run.attempts.append(Attempt(v.sql, "ok"))
                run.ok, run.sql, run.result = True, v.sql, result
                break

            if not run.ok and first_empty is not None:  # repair didn't help: an empty answer is still an answer
                run.ok, run.sql, run.result = True, first_empty[0], first_empty[1]
            if not run.ok:
                run.error = run.attempts[-1].error if run.attempts else "no attempts"

            if run.ok and store is not None:
                df = run.result.df
                run.evidence = store.add_query(
                    sub_question=question, sql=run.sql, dialect=self.db.dialect, df=df,
                    truncated=bool(enforce_limit and cap and len(df) >= cap),
                    elapsed_s=run.result.elapsed_s, attempts=run.n_attempts,
                    tables=self.validator.validate(run.sql, enforce_limit=False).tables)
            run.latency_s = time.perf_counter() - t0
            span["output"] = run.to_dict()
        return run

    def run_sql_direct(self, sql: str, store: EvidenceStore | None = None, label: str = "direct query",
                       max_rows: int | None = None) -> SQLRun:
        """Validated execution of caller-supplied SQL (used by the API and MCP server)."""
        cap = max_rows if max_rows is not None else self.max_rows
        v = self.validator.validate(sql, max_rows=cap)
        if not v.ok:
            return SQLRun(label, False, sql, None, [Attempt(sql, v.stage, v.error_message)], error=v.error_message)
        try:
            result = self.db.execute(v.sql, timeout_s=self.query_timeout_s)
        except Exception as e:
            return SQLRun(label, False, v.sql, None, [Attempt(v.sql, "execute", str(e))], error=str(e))
        run = SQLRun(label, True, v.sql, result, [Attempt(v.sql, "ok")])
        if store is not None:
            run.evidence = store.add_query(sub_question=label, sql=v.sql, dialect=self.db.dialect, df=result.df,
                                           truncated=bool(cap and result.row_count >= cap),
                                           elapsed_s=result.elapsed_s, attempts=1, tables=v.tables)
        return run


_SQL_FENCE = re.compile(r"```(?:sql|sqlite|duckdb)?\s*\n?(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str | None:
    blocks = [b.strip() for b in _SQL_FENCE.findall(text) if b.strip()]
    if blocks:
        return blocks[-1].rstrip(";").strip()
    m = re.search(r"\b(WITH|SELECT)\b.*", text, re.S | re.I)
    return m.group(0).strip().rstrip(";").strip() if m else None


def df_or_empty(run: SQLRun) -> pd.DataFrame:
    return run.result.df if run.result is not None else pd.DataFrame()
