import json
import sqlite3

from evals.metrics import cohens_kappa, execution_match, mcnemar
from evals.regression import compare
from insightforge.llm import FakeProvider


def test_execution_match():
    assert execution_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a"), (1, "a")])
    assert not execution_match([(1,)], [(2,)]) and not execution_match(None, [])


def test_kappa_and_mcnemar():
    assert round(cohens_kappa([0, 1, 2, 2, 1, 0, 2, 2], [0, 1, 2, 1, 1, 0, 2, 2]), 3) == 0.81  # sklearn: 0.8095
    r = mcnemar([True] * 50 + [False] * 50, [True] * 40 + [False] * 58 + [True] * 2)
    assert r["regressions"] == 10 and r["fixes"] == 2 and r["p_value"] < 0.05


def test_regression_gate():
    base = {"ex": 60.0, "per_question": {str(i): i < 60 for i in range(100)}}
    same = compare(dict(base), base, "ex", 3.0, 0.05)
    assert same["status"] == "pass"
    worse = {"ex": 58.0, "per_question": {str(i): (i < 60 and i >= 12) or i in (70, 71) or i < 2 for i in range(100)}}
    assert compare(worse, base, "ex", 3.0, 0.05)["status"] == "fail"  # small drop, but paired test catches it
    assert compare({"ex": 55.0}, base, "ex", 3.0, 0.05)["status"] == "fail"


def test_bird_runner_on_mini_fixture(tmp_path):
    from evals.bird.run_bird import BirdRunner, load_bird, stratified_sample, summarize

    root = tmp_path / "dev"
    dbdir = root / "dev_databases" / "shop"
    (dbdir / "database_description").mkdir(parents=True)
    con = sqlite3.connect(dbdir / "shop.sqlite")
    con.executescript("CREATE TABLE client(id INTEGER PRIMARY KEY, city TEXT);"
                      "CREATE TABLE sale(id INTEGER, client_id INTEGER REFERENCES client(id), amount REAL);"
                      "INSERT INTO client VALUES (1,'Pune'),(2,'Hyderabad');"
                      "INSERT INTO sale VALUES (1,1,10),(2,1,5),(3,2,7);")
    con.commit()
    con.close()
    (dbdir / "database_description" / "sale.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "amount,sale amount,amount in INR,real,\n")
    items = [{"question_id": 0, "db_id": "shop", "question": "Total sales in Pune?", "evidence": "",
              "SQL": "SELECT SUM(amount) FROM sale JOIN client ON client.id = sale.client_id WHERE city = 'Pune'",
              "difficulty": "simple"},
             {"question_id": 1, "db_id": "shop", "question": "How many clients?", "evidence": "",
              "SQL": "SELECT COUNT(*) FROM client", "difficulty": "moderate"}]
    (root / "dev.json").write_text(json.dumps(items))

    answers = {"Pune": items[0]["SQL"], "clients": "SELECT COUNT(*) FROM sale"}  # 2nd is wrong on purpose

    def respond(system, messages, role):
        q = messages[0]["content"]
        assert "amount in INR" in q or "clients" in q  # BIRD descriptions reach the prompt
        return "```sql\n" + next(v for k, v in answers.items() if k in q) + "\n```"

    runner = BirdRunner(root, FakeProvider(responder=respond), "retrieval+repair")
    rows = [runner.run_one(it) for it in stratified_sample(load_bird(root), None, 0)]
    s = summarize(rows, "retrieval+repair", "fake")
    assert [r["correct"] for r in rows] == [True, False] and s["ex"] == 50.0
    assert rows[0]["gold_tables_recalled"] is True


def test_olist_sql_mode_with_gold_answers(olist_db):
    from evals.olist.run_olist import load_questions, run

    gold = {q["question"]: q["gold_sql"] for q in load_questions()}

    def respond(system, messages, role):
        q = messages[0]["content"].split("Question: ")[-1].strip()
        return f"```sql\n{gold[q]}\n```"

    rows, summ = run(olist_db, "sql", FakeProvider(responder=respond), workers=1)
    assert summ["n"] == 50 and summ["ex_relaxed"] == 100.0, [r for r in rows if not r["correct"]]
