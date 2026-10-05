"""The agent, as a LangGraph state machine.

    plan -> sql -> analyze -> synthesize -> critic --(bad claims & budget left)--> synthesize
                                                    \\-> finalize -> END

Four roles: Planner (plan + synthesize), SQL tool, Analysis tool, Critic.
"""
from __future__ import annotations

import difflib
import time
from typing import Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator

from insightforge import prompts
from insightforge.agent.critic import Critic
from insightforge.agent.report import Report
from insightforge.config import Settings, get_settings
from insightforge.db import Database
from insightforge.evidence import EvidenceStore
from insightforge.llm import LLMProvider, track_usage
from insightforge.tools.analysis_tool import AnalysisTool
from insightforge.tools.sql_tool import SQLTool
from insightforge.tools.stats_lib import REGISTRY
from insightforge.tracing import get_tracer


# ---- schemas for LLM outputs (the "schema-valid outputs" check in the eval suite uses these) -----
class SubQuestion(BaseModel):
    id: str
    question: str
    needs_analysis: bool = False
    analysis: Literal["t_test", "chi_square", "correlation", "regression", "anomalies", "describe"] | None = None
    analysis_args: dict[str, Any] = Field(default_factory=dict)
    chart: Literal["bar", "line", "scatter", "none"] | None = "none"

    @field_validator("chart", mode="before")
    @classmethod
    def _chart(cls, v):
        return v if v in ("bar", "line", "scatter", "none") else "none"


class Plan(BaseModel):
    sub_questions: list[SubQuestion] = Field(min_length=1)


class ClaimOut(BaseModel):
    text: str = Field(min_length=3)
    evidence_ids: list[str] = Field(default_factory=list)
    kind: Literal["descriptive", "comparative", "statistical", "causal", "other"] = "descriptive"

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, v):
        return v if v in ("descriptive", "comparative", "statistical", "causal", "other") else "other"


class Synthesis(BaseModel):
    summary: str = ""
    claims: list[ClaimOut] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    question: str
    store: EvidenceStore
    plan: list[dict]
    sub_results: dict[str, dict]  # sub-question id -> {"evidence_id", "ok", "error"}
    failures: list[dict]
    synthesis: dict
    revisions: int
    schema_errors: list[str]
    report: Report


class InsightForgeAgent:
    def __init__(self, db: Database, llm: LLMProvider, settings: Settings | None = None, use_critic_llm: bool = True):
        self.s = settings or get_settings()
        self.db = db
        self.llm = llm
        self.sql = SQLTool(db, llm, max_rows=self.s.max_rows, max_repairs=self.s.max_repair_attempts,
                           use_retrieval=self.s.use_schema_retrieval, top_tables=self.s.retrieval_top_tables,
                           query_timeout_s=self.s.query_timeout_s)
        self.analysis = AnalysisTool(timeout_s=self.s.analysis_timeout_s)
        self.critic = Critic(llm, use_llm=use_critic_llm)
        self.graph = self._build()

    # ---- nodes -------------------------------------------------------------------------------
    def _plan(self, state: AgentState) -> AgentState:
        tables = ", ".join(f"{t.name} ({t.description})" if t.description else t.name
                           for t in self.db.schema().tables.values())
        system = prompts.PLANNER_SYSTEM.format(max_sub=self.s.max_sub_questions, tables=tables)
        errors: list[str] = []
        try:
            raw = self.llm.complete_json(system, f"Question: {state['question']}", role="planner", max_tokens=1000)
            plan = Plan.model_validate(raw)
            subs = plan.sub_questions[: self.s.max_sub_questions]
        except (ValidationError, ValueError) as e:
            errors.append(f"planner output invalid: {str(e)[:200]}")
            subs = [SubQuestion(id="S1", question=state["question"])]
        for i, sq in enumerate(subs, 1):  # normalise ids so downstream never trusts model-made ids
            sq.id = f"S{i}"
        return {"plan": [sq.model_dump() for sq in subs], "schema_errors": errors}

    def _run_sql(self, state: AgentState) -> AgentState:
        store = state["store"]
        results, failures = {}, []
        for sq in state["plan"]:
            rows = self.s.analysis_max_rows if sq.get("needs_analysis") else None
            run = self.sql.run(sq["question"], store=store, max_rows=rows)
            if run.ok:
                results[sq["id"]] = {"evidence_id": run.evidence.id, "ok": True, "attempts": run.n_attempts}
            else:
                results[sq["id"]] = {"evidence_id": None, "ok": False, "error": run.error}
                failures.append({"id": sq["id"], "question": sq["question"], "error": run.error})
        return {"sub_results": results, "failures": failures}

    def _analyze(self, state: AgentState) -> AgentState:
        store = state["store"]
        failures = list(state.get("failures", []))
        for sq in state["plan"]:
            res = state["sub_results"].get(sq["id"], {})
            qid = res.get("evidence_id")
            if not qid:
                continue
            df = store.frames[qid]
            if sq.get("chart") and sq["chart"] != "none":
                self._chart(sq, qid, df, store)
            if sq.get("needs_analysis") and sq.get("analysis") in REGISTRY:
                args = _fit_args(sq["analysis"], sq.get("analysis_args") or {}, list(df.columns))
                run = self.analysis.run_helper(sq["analysis"], args, {qid: df}, store=store,
                                               description=f"{sq['analysis']} for: {sq['question']}")
                if run.ok:
                    for fj in run.figures:
                        store.add_chart(title=f"{sq['analysis']}: {sq['question']}", inputs=[run.evidence.id],
                                        figure_json=fj)
                else:
                    failures.append({"id": sq["id"], "question": sq["question"],
                                     "error": f"analysis {sq['analysis']} failed: {run.error}"})
        return {"failures": failures}

    def _chart(self, sq: dict, qid: str, df: pd.DataFrame, store: EvidenceStore) -> None:
        import plotly.express as px

        if df.empty or len(df.columns) < 2 or len(df) > 2000:
            return
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num:
            return
        x, y = df.columns[0], next((c for c in num if c != df.columns[0]), None)
        if y is None:
            return
        try:
            kind = sq["chart"]
            if kind == "line" or pd.api.types.is_datetime64_any_dtype(df[x]):
                fig = px.line(df.sort_values(x), x=x, y=y, title=sq["question"])
            elif kind == "scatter" and pd.api.types.is_numeric_dtype(df[x]):
                fig = px.scatter(df, x=x, y=y, title=sq["question"])
            else:
                fig = px.bar(df.head(30), x=x, y=y, title=sq["question"])
            store.add_chart(title=sq["question"], inputs=[qid], figure_json=fig.to_json())
        except Exception:
            pass

    def _synthesize(self, state: AgentState) -> AgentState:
        store = state["store"]
        ids = [e.id for e in store.queries()] + [e.id for e in store.calcs()]
        evidence = "\n\n".join(store.describe(e) for e in ids) or "(no evidence: every query failed)"
        fb = ""
        if state.get("revisions", 0) > 0:
            fb = prompts.SYNTH_FEEDBACK.format(notes=Critic.feedback(store))
        errors = list(state.get("schema_errors", []))
        try:
            raw = self.llm.complete_json(prompts.SYNTH_SYSTEM,
                                         prompts.SYNTH_USER.format(question=state["question"], evidence=evidence,
                                                                   feedback=fb),
                                         role="synthesizer", max_tokens=2000)
            syn = Synthesis.model_validate(raw)
        except (ValidationError, ValueError) as e:
            errors.append(f"synthesis output invalid: {str(e)[:200]}")
            syn = Synthesis(summary="The agent could not produce a valid report for this question.",
                            caveats=["Report generation failed schema validation."])
        store.reset_claims()
        for c in syn.claims:
            store.add_claim(c.text, [e.strip().strip("[]") for e in c.evidence_ids], c.kind)
        return {"synthesis": syn.model_dump(), "schema_errors": errors}

    def _critic(self, state: AgentState) -> AgentState:
        with get_tracer().span("critic", as_type="evaluator") as span:
            claims = self.critic.review(state["store"])
            span["output"] = {c.id: c.status for c in claims}
        return {}

    def _route_after_critic(self, state: AgentState) -> str:
        bad = [c for c in state["store"].claims.values() if c.status != "supported"]
        if bad and state.get("revisions", 0) < self.s.max_critic_revisions:
            return "revise"
        return "finalize"

    def _revise(self, state: AgentState) -> AgentState:
        return {"revisions": state.get("revisions", 0) + 1}

    def _finalize(self, state: AgentState) -> AgentState:
        store = state["store"]
        syn = state.get("synthesis", {})
        claims = list(store.claims.values())
        report = Report(question=state["question"], summary=syn.get("summary", ""),
                        claims=[c for c in claims if c.status == "supported"],
                        rejected=store.rejected_log + [c for c in claims if c.status != "supported"],
                        caveats=syn.get("caveats", []), store=store, plan=state.get("plan", []),
                        failures=state.get("failures", []), revisions=state.get("revisions", 0))
        for eid in {e for c in report.claims for e in c.evidence_ids}:
            for note in (n for c in report.claims for n in c.critic_notes if eid in n):
                if note not in report.caveats:
                    report.caveats.append(note)
        return {"report": report}

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("sql", self._run_sql)
        g.add_node("analyze", self._analyze)
        g.add_node("synthesize", self._synthesize)
        g.add_node("critic", self._critic)
        g.add_node("revise", self._revise)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "plan")
        g.add_edge("plan", "sql")
        g.add_edge("sql", "analyze")
        g.add_edge("analyze", "synthesize")
        g.add_edge("synthesize", "critic")
        g.add_conditional_edges("critic", self._route_after_critic, {"revise": "revise", "finalize": "finalize"})
        g.add_edge("revise", "synthesize")
        g.add_edge("finalize", END)
        return g.compile()

    # ---- entry point -------------------------------------------------------------------------
    def ask(self, question: str, save: bool = True) -> Report:
        t0 = time.perf_counter()
        tracer = get_tracer()
        with track_usage() as usage, tracer.span("insightforge.ask", as_type="agent", input=question) as span:
            final = self.graph.invoke({"question": question, "store": EvidenceStore(), "revisions": 0})
            report: Report = final["report"]
            report.usage = usage.to_dict()
            report.latency_s = time.perf_counter() - t0
            report.failures += [{"id": "-", "question": "(schema)", "error": e} for e in final.get("schema_errors", [])]
            span["output"] = report.metrics()
        tracer.flush()
        if save:
            report.save(self.s.runs_dir)
        return report


_COLUMN_ARGS = {"value", "group", "x", "y", "time", "a", "b"}


def _fit_args(method: str, args: dict, columns: list[str]) -> dict:
    """Normalize and fuzzy-match planner-generated analysis arguments.

    The planner may occasionally use generic names such as column_x/column_y
    instead of the canonical arguments expected by stats_lib. Normalize those
    aliases first, then map the resulting column names to the actual SQL
    result columns.
    """
    lower = {c.lower(): c for c in columns}

    def fit(v):
        if not isinstance(v, str) or v in columns:
            return v
        if v.lower() in lower:
            return lower[v.lower()]
        m = difflib.get_close_matches(
            v.lower(),
            list(lower),
            n=1,
            cutoff=0.6,
        )
        return lower[m[0]] if m else v

    # Normalize common LLM/planner aliases to the canonical stats_lib names.
    aliases = {
        "column_x": "value" if method in {"t_test", "describe", "anomalies"} else "x",
        "column_y": "group" if method in {"t_test", "describe"} else "y",
        "column_a": "a",
        "column_b": "b",
        "value_column": "value",
        "group_column": "group",
    }

    normalized = {}
    for key, value in args.items():
        canonical_key = aliases.get(key, key)

        # Don't overwrite an explicitly supplied canonical argument.
        if canonical_key in normalized:
            continue

        normalized[canonical_key] = value

    col_keys = _COLUMN_ARGS

    out = {}
    for key, value in normalized.items():
        if key not in col_keys:
            out[key] = value
        elif isinstance(value, list):
            out[key] = [fit(x) for x in value]
        else:
            out[key] = fit(value)

    return out