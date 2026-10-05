"""Final report: findings with claim IDs, and a provenance section a reader can check by hand."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from insightforge.evidence import CalcEvidence, Claim, EvidenceStore, QueryEvidence


@dataclass
class Report:
    question: str
    summary: str
    claims: list[Claim]
    rejected: list[Claim]
    caveats: list[str]
    store: EvidenceStore
    plan: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    revisions: int = 0

    @property
    def run_id(self) -> str:
        return self.store.run_id

    def metrics(self) -> dict:
        total = len(self.claims) + len(self.rejected)
        return {
            "claims_total": total, "claims_supported": len(self.claims), "claims_rejected": len(self.rejected),
            "rejected_by_status": _count(c.status for c in self.rejected), "revisions": self.revisions,
            "queries": len(self.store.queries()), "calcs": len(self.store.calcs()),
            "sub_question_failures": len(self.failures), "latency_s": round(self.latency_s, 2),
            "cost_usd": self.usage.get("cost_usd", 0.0), "llm_calls": self.usage.get("calls", 0),
        }

    def to_markdown(self) -> str:
        L = [f"# {self.question}", "", self.summary or "_No summary produced._", ""]
        if self.claims:
            L += ["## Findings", ""] + [f"- {c.text} **[{c.id}]**" for c in self.claims] + [""]
        if self.caveats or self.failures:
            L += ["## Caveats", ""] + [f"- {c}" for c in self.caveats]
            L += [f"- Could not answer sub-question \"{f['question']}\": {f['error']}" for f in self.failures]
            L.append("")
        charts = self.store.charts()
        if charts:
            L += ["## Charts", ""] + [f"- **{v.id}** {v.title} (from {', '.join(v.inputs)})" for v in charts] + [""]
        if self.claims:
            L += ["## Provenance", "", "| Claim | Evidence |", "|---|---|"]
            for c in self.claims:
                L.append(f"| {c.id} | " + "; ".join(self._ev_label(e) for e in c.evidence_ids) + " |")
            L.append("")
            cited = self._cited_with_inputs()
            for eid in cited:
                ev = self.store.get(eid)
                if isinstance(ev, QueryEvidence):
                    L += [f"### {ev.id}: {ev.sub_question}",
                          f"{ev.row_count:,} rows{' (truncated)' if ev.truncated else ''}, "
                          f"{ev.attempts} attempt(s), tables: {', '.join(ev.tables)}",
                          "```sql", ev.sql, "```", ""]
                elif isinstance(ev, CalcEvidence):
                    L += [f"### {ev.id}: {ev.method}", f"Inputs: {', '.join(ev.inputs)}", "```json",
                          json.dumps(_brief(ev.outputs), indent=2, default=str), "```", ""]
        if self.rejected:
            L += ["## Removed by the critic", ""]
            L += [f"- ~~{c.text}~~ [{c.status}] {'; '.join(c.critic_notes)}" for c in self.rejected] + [""]
        m = self.metrics()
        L.append(f"_Run {self.run_id}: {m['queries']} queries, {m['calcs']} calculations, "
                 f"{m['llm_calls']} LLM calls, ${m['cost_usd']:.4f}, {m['latency_s']}s._")
        return "\n".join(L)

    def _cited_with_inputs(self) -> list[str]:
        """Cited evidence plus the queries that fed any cited calculation, queries first."""
        ids = {e for c in self.claims for e in c.evidence_ids}
        for e in list(ids):
            ev = self.store.get(e)
            if isinstance(ev, CalcEvidence):
                ids.update(ev.inputs)
        return sorted((i for i in ids if self.store.get(i) is not None),
                      key=lambda s: ({"Q": 0, "C": 1, "V": 2}.get(s[0], 3), int(s[1:])))

    def _ev_label(self, eid: str) -> str:
        ev = self.store.get(eid)
        if isinstance(ev, QueryEvidence):
            return f"{eid} (SQL, {ev.row_count:,} rows)"
        if isinstance(ev, CalcEvidence):
            p = ev.outputs.get("p_value")
            return f"{eid} ({ev.method}" + (f", p={p:.3g}" if isinstance(p, (int, float)) else "") + ")"
        return eid

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "question": self.question, "summary": self.summary,
                "claims": [c.__dict__ for c in self.claims], "rejected": [c.__dict__ for c in self.rejected],
                "caveats": self.caveats, "plan": self.plan, "failures": self.failures, "metrics": self.metrics(),
                "usage": self.usage, "evidence": self.store.to_dict(include_figures=False)["evidence"],
                "charts": [{"id": v.id, "title": v.title, "figure_json": v.figure_json} for v in self.store.charts()]}

    def save(self, runs_dir: Path) -> Path:
        d = self.store.save(runs_dir)
        (d / "report.md").write_text(self.to_markdown(), encoding="utf-8")
        (d / "report.json").write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return d


def _count(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _brief(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("table", "anomalies", "by_group")} | (
        {"by_group": d["by_group"][:10]} if "by_group" in d else {})
