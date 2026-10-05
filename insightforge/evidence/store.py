"""Evidence store: every number in a report must trace back to something stored here.

IDs:  Q<n> = executed SQL query,  C<n> = calculation / statistical test,  V<n> = chart,  F<n> = claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ClaimKind = Literal["descriptive", "comparative", "statistical", "causal", "other"]
ClaimStatus = Literal["pending", "supported", "unsupported", "causal_overreach", "unverified_number", "needs_hedge"]


@dataclass
class QueryEvidence:
    id: str
    sub_question: str
    sql: str
    dialect: str
    row_count: int
    columns: list[str]
    preview: list[dict]
    result_hash: str
    truncated: bool
    elapsed_s: float
    attempts: int
    tables: list[str]
    kind: str = "query"


@dataclass
class CalcEvidence:
    id: str
    description: str
    method: str
    inputs: list[str]
    outputs: dict[str, Any]
    code: str = ""
    kind: str = "calc"


@dataclass
class ChartEvidence:
    id: str
    title: str
    inputs: list[str]
    figure_json: str
    kind: str = "chart"


@dataclass
class Claim:
    id: str
    text: str
    evidence_ids: list[str]
    kind: ClaimKind = "descriptive"
    status: ClaimStatus = "pending"
    critic_notes: list[str] = field(default_factory=list)


class EvidenceStore:
    PREVIEW_ROWS = 200

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:6]
        self.items: dict[str, QueryEvidence | CalcEvidence | ChartEvidence] = {}
        self.frames: dict[str, pd.DataFrame] = {}  # full results, in memory only
        self.claims: dict[str, Claim] = {}
        self.rejected_log: list[Claim] = []  # claims the critic rejected in earlier drafts
        self._n = {"Q": 0, "C": 0, "V": 0, "F": 0}

    def _next(self, p: str) -> str:
        self._n[p] += 1
        return f"{p}{self._n[p]}"

    # ---- add ---------------------------------------------------------------------------------
    def add_query(self, *, sub_question: str, sql: str, dialect: str, df: pd.DataFrame, truncated: bool,
                  elapsed_s: float, attempts: int, tables: list[str]) -> QueryEvidence:
        qid = self._next("Q")
        ev = QueryEvidence(qid, sub_question, sql, dialect, len(df), [str(c) for c in df.columns],
                           _records(df.head(self.PREVIEW_ROWS)), _hash_df(df), truncated, round(elapsed_s, 4),
                           attempts, tables)
        self.items[qid] = ev
        self.frames[qid] = df
        return ev

    def add_calc(self, *, description: str, method: str, inputs: list[str], outputs: dict, code: str = "") -> CalcEvidence:
        ev = CalcEvidence(self._next("C"), description, method, inputs, _jsonable(outputs), code)
        self.items[ev.id] = ev
        return ev

    def add_chart(self, *, title: str, inputs: list[str], figure_json: str) -> ChartEvidence:
        ev = ChartEvidence(self._next("V"), title, inputs, figure_json)
        self.items[ev.id] = ev
        return ev

    def add_claim(self, text: str, evidence_ids: list[str], kind: ClaimKind = "descriptive") -> Claim:
        c = Claim(self._next("F"), text.strip(), [e for e in evidence_ids if e], kind)
        self.claims[c.id] = c
        return c

    def reset_claims(self) -> None:
        """Start a new draft. Rejected claims are archived (with draft-scoped ids) so the report can show them."""
        for c in self.claims.values():
            if c.status not in ("pending", "supported"):
                c.id = f"{c.id}'"
                self.rejected_log.append(c)
        self.claims.clear()
        self._n["F"] = 0

    # ---- read --------------------------------------------------------------------------------
    def get(self, eid: str):
        return self.items.get(eid)

    def queries(self) -> list[QueryEvidence]:
        return [e for e in self.items.values() if isinstance(e, QueryEvidence)]

    def calcs(self) -> list[CalcEvidence]:
        return [e for e in self.items.values() if isinstance(e, CalcEvidence)]

    def charts(self) -> list[ChartEvidence]:
        return [e for e in self.items.values() if isinstance(e, ChartEvidence)]

    def numbers_for(self, evidence_ids: list[str]) -> list[float]:
        """All numeric values an evidence item can vouch for (full result frames, calc outputs, row counts)."""
        nums: list[float] = []
        for eid in evidence_ids:
            ev = self.items.get(eid)
            if ev is None:
                continue
            if isinstance(ev, QueryEvidence):
                nums.append(float(ev.row_count))
                df = self.frames.get(eid)
                if df is not None:
                    for col in df.columns:
                        s = pd.to_numeric(df[col], errors="coerce").dropna()
                        nums.extend(s.head(5000).astype(float).tolist())
                else:
                    nums.extend(_flatten_numbers(ev.preview))
            elif isinstance(ev, CalcEvidence):
                nums.extend(_flatten_numbers(ev.outputs))
        return [n for n in nums if math.isfinite(n)]

    def describe(self, eid: str, max_rows: int = 15) -> str:
        """Text summary of an evidence item, used in synthesis/critic prompts."""
        ev = self.items.get(eid)
        if ev is None:
            return f"[{eid}] (missing)"
        if isinstance(ev, QueryEvidence):
            rows = ev.preview[:max_rows]
            more = f" (showing {len(rows)} of {ev.row_count})" if ev.row_count > len(rows) else ""
            return (f"[{ev.id}] query for: {ev.sub_question}\nSQL: {ev.sql}\nrows={ev.row_count}{more}"
                    f"{' TRUNCATED at row limit' if ev.truncated else ''}\n{_table_text(rows, ev.columns)}")
        if isinstance(ev, CalcEvidence):
            return (f"[{ev.id}] {ev.method}: {ev.description} (inputs: {', '.join(ev.inputs)})\n"
                    f"{json.dumps(ev.outputs, default=str)[:1500]}")
        return f"[{ev.id}] chart: {ev.title} (inputs: {', '.join(ev.inputs)})"

    # ---- persist -----------------------------------------------------------------------------
    def to_dict(self, include_figures: bool = True) -> dict:
        items = {}
        for k, v in self.items.items():
            d = asdict(v)
            if isinstance(v, ChartEvidence) and not include_figures:
                d.pop("figure_json")
            items[k] = d
        return {"run_id": self.run_id, "evidence": items, "claims": {k: asdict(c) for k, c in self.claims.items()},
                "rejected_drafts": [asdict(c) for c in self.rejected_log]}

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceStore":
        """Rebuild a store from a saved report/evidence JSON (no full frames; previews are used instead)."""
        st = cls(run_id=d.get("run_id"))
        kinds = {"query": QueryEvidence, "calc": CalcEvidence, "chart": ChartEvidence}
        for k, v in (d.get("evidence") or {}).items():
            k_cls = kinds[v.get("kind", "query")]
            fields_ = {f for f in k_cls.__dataclass_fields__}
            if k_cls is ChartEvidence:
                v = {"figure_json": "", **v}
            st.items[k] = k_cls(**{f: v[f] for f in fields_ if f in v})
        return st

    def save(self, runs_dir: Path) -> Path:
        d = Path(runs_dir) / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "evidence.json").write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        for qid, df in self.frames.items():
            df.to_csv(d / f"{qid}.csv", index=False)
        return d


# ---- helpers ---------------------------------------------------------------------------------
def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes()).hexdigest()[:12]


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


def _flatten_numbers(x: Any) -> list[float]:
    out: list[float] = []
    if isinstance(x, bool):
        return out
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(_flatten_numbers(v))
    elif isinstance(x, (list, tuple)):
        for v in x:
            out.extend(_flatten_numbers(v))
    elif isinstance(x, str) and re.fullmatch(r"-?\d+(\.\d+)?", x.strip()):
        out.append(float(x))
    return out


def _table_text(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    lines = [" | ".join(columns)]
    for r in rows:
        lines.append(" | ".join(_cell(r.get(c)) for c in columns))
    return "\n".join(lines)


def _cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)
