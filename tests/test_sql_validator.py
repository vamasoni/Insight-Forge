import pytest

from insightforge.tools.sql_validator import SQLValidator


@pytest.fixture(scope="module")
def v(db):
    return SQLValidator(db, max_rows=100)


@pytest.mark.parametrize("sql,stage", [
    ("DELETE FROM orders", "readonly"),
    ("DROP TABLE orders", "readonly"),
    ("INSERT INTO orders SELECT * FROM orders", "readonly"),
    ("SELECT 1; DROP TABLE orders", "statement"),
    ("SELECT * FROM read_csv('/etc/passwd')", "function"),
    ("SELECT getenv('HOME')", "function"),
    ("SELECT * FROM ordrs", "table"),
    ("SELECT no_such_col FROM orders", "explain"),
    ("SELEC 1", "parse"),
])
def test_rejects(v, sql, stage):
    r = v.validate(sql)
    assert not r.ok and r.stage == stage, r


def test_unknown_table_message_lists_tables(v):
    assert "Available tables" in v.validate("SELECT * FROM ordrs").error_message


def test_limit_injected_and_capped(v):
    assert v.validate("SELECT * FROM orders").sql.endswith("LIMIT 100")
    assert v.validate("SELECT * FROM orders LIMIT 10").sql.endswith("LIMIT 10")
    assert v.validate("SELECT * FROM orders LIMIT 99999").sql.endswith("LIMIT 100")
    assert "LIMIT" not in v.validate("SELECT * FROM orders", enforce_limit=False).sql.upper()


def test_cte_and_union_ok(v):
    assert v.validate("WITH x AS (SELECT * FROM orders) SELECT COUNT(*) FROM x").ok
    r = v.validate("SELECT order_status FROM orders UNION SELECT seller_state FROM sellers")
    assert r.ok and r.sql.endswith("LIMIT 100")


def test_connection_is_read_only(db):
    with pytest.raises(Exception):
        db.execute("DELETE FROM orders")
    with pytest.raises(Exception):
        db.execute("SELECT * FROM read_csv_auto('/etc/hosts')")
