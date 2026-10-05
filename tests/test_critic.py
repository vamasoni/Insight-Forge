import pandas as pd

from insightforge.agent.critic import Critic, deterministic_check, extract_numbers
from insightforge.evidence import EvidenceStore


def _store():
    st = EvidenceStore()
    st.add_query(sub_question="q", sql="SELECT 1", dialect="duckdb", truncated=False, elapsed_s=0, attempts=1,
                 tables=[], df=pd.DataFrame({"state": ["SP", "RJ"], "share": [0.4213, 0.1301], "n": [3370, 1041]}))
    st.add_calc(description="t", method="welch_t_test", inputs=["Q1"], outputs={"p_value": 0.2, "mean_a": 4.18})
    return st


def test_extract_numbers():
    vals = [n.value for n in extract_numbers("42.1% of 3,370 orders (R$ 1.2M) in 2018, Q1")]
    assert vals == [42.1, 3370.0, 1.2e6, 2018.0]


def test_rounding_and_percent():
    st = _store()
    ok = st.add_claim("SP accounts for 42.1% of orders (3,370).", ["Q1"])
    assert deterministic_check(ok, st).status == "supported"
    bad = st.add_claim("SP accounts for 45% of orders.", ["Q1"])
    assert deterministic_check(bad, st).status == "unverified_number"


def test_causal_and_significance():
    st = _store()
    assert deterministic_check(st.add_claim("Being in SP causes more orders.", ["Q1"]), st).status == "causal_overreach"
    assert deterministic_check(st.add_claim("This does not mean SP causes it.", ["Q1"]), st).status == "supported"
    sig = st.add_claim("Mean score 4.18 is significantly higher.", ["C1"])  # p = 0.2
    assert deterministic_check(sig, st).status == "unsupported"
    assert deterministic_check(st.add_claim("SP is significant.", ["Q1"]), st).status == "needs_hedge"


def test_missing_evidence():
    st = _store()
    assert deterministic_check(st.add_claim("3,370 orders.", []), st).status == "unsupported"
    assert deterministic_check(st.add_claim("3,370 orders.", ["Q7"]), st).status == "unsupported"


def test_critic_eval_thresholds(olist_db):
    from evals.critic_eval import run

    s = run(olist_db)
    assert s["recall"] >= 95 and s["false_positive_rate"] <= 5, s


def test_llm_layer_only_sees_passing_claims():
    from insightforge.llm import FakeProvider

    st = _store()
    st.add_claim("SP has 3,370 orders.", ["Q1"])
    st.add_claim("SP causes orders.", ["Q1"])
    llm = FakeProvider(queue=['{"verdicts": [{"id": "F1", "verdict": "needs_hedge", "reason": "r"}]}'])
    Critic(llm).review(st)
    assert st.claims["F1"].status == "needs_hedge" and st.claims["F2"].status == "causal_overreach"
    assert "F2" not in llm.calls[0]["messages"][0]["content"]
