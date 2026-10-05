from insightforge.evidence import EvidenceStore
from insightforge.llm import FakeProvider, track_usage
from insightforge.tools.sql_tool import SQLTool, extract_sql


def test_extract_sql():
    assert extract_sql("x\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Here: SELECT a FROM b") == "SELECT a FROM b"
    assert extract_sql("no query") is None


def test_repair_from_binder_error(db):
    llm = FakeProvider(queue=["```sql\nSELECT payment_typ, COUNT(*) FROM payments GROUP BY 1\n```",
                              "```sql\nSELECT payment_type, COUNT(*) AS n FROM payments GROUP BY 1\n```"])
    store = EvidenceStore()
    with track_usage() as u:
        run = SQLTool(db, llm).run("orders per payment type", store=store)
    assert run.ok and run.repaired and [a.stage for a in run.attempts] == ["explain", "ok"]
    assert "payment_typ" in llm.calls[1]["messages"][-1]["content"]  # error fed back to the model
    assert u.by_role["sql_repair"]["calls"] == 1
    assert run.evidence.id == "Q1" and store.get("Q1").row_count == 4


def test_gives_up_after_budget(db):
    llm = FakeProvider(queue=["```sql\nSELECT nope FROM orders\n```"] * 3)
    run = SQLTool(db, llm, max_repairs=2).run("q")
    assert not run.ok and run.n_attempts == 3 and "nope" in run.error


def test_empty_result_retry_then_keep(db):
    empty = "```sql\nSELECT COUNT(*) AS n FROM orders WHERE order_status = 'Delivered' HAVING COUNT(*) > 0\n```"
    llm = FakeProvider(queue=[empty, empty])
    run = SQLTool(db, llm, max_repairs=2).run("delivered orders")
    assert run.ok and run.result.row_count == 0 and [a.stage for a in run.attempts][0] == "empty"


def test_no_retrieval_uses_full_schema(db):
    llm = FakeProvider(queue=["```sql\nSELECT 1\n```"])
    run = SQLTool(db, llm, use_retrieval=False).run("anything")
    assert set(run.retrieval.tables) == set(db.schema().tables)
