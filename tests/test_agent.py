from pathlib import Path

from fake_llm import make_fake_llm

from insightforge.agent import InsightForgeAgent
from insightforge.config import get_settings


def test_end_to_end(db, tmp_path):
    agent = InsightForgeAgent(db, make_fake_llm(bad_first_sql=True), get_settings(runs_dir=tmp_path))
    rep = agent.ask("Do late deliveries get worse reviews?")
    statuses = {c.status for c in rep.rejected}
    assert {"causal_overreach", "unverified_number"} <= statuses
    assert rep.claims and all(c.status == "supported" for c in rep.claims)
    assert all(rep.store.get(e) for c in rep.claims for e in c.evidence_ids)
    md = rep.to_markdown()
    assert "## Provenance" in md and "```sql" in md and "Removed by the critic" in md
    q2 = rep.store.get("Q2")
    assert not q2.truncated and q2.row_count > 2000  # analysis input not capped at the display limit
    assert rep.revisions == 1 and rep.metrics()["llm_calls"] >= 6
    assert (Path(tmp_path) / rep.run_id / "report.json").exists()


def test_invalid_planner_output_falls_back(db, tmp_path):
    from insightforge.llm import FakeProvider

    def respond(system, messages, role):
        if role == "planner":
            return "not json at all"
        if role.startswith("sql"):
            return "```sql\nSELECT COUNT(*) AS n_orders FROM orders\n```"
        if role == "synthesizer":
            return '{"summary": "s", "claims": [], "caveats": []}'
        return '{"verdicts": []}'

    rep = InsightForgeAgent(db, FakeProvider(responder=respond), get_settings(runs_dir=tmp_path)).ask("how many?")
    assert len(rep.plan) == 1 and any("planner output invalid" in f["error"] for f in rep.failures)
