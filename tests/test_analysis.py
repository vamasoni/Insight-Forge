import numpy as np
import pandas as pd
import pytest
from scipy import stats

from insightforge.evidence import EvidenceStore
from insightforge.tools.analysis_tool import AnalysisTool, UnsafeCode, check_code
from insightforge.tools.stats_lib import chi_square, correlation, t_test


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    g = np.where(rng.random(400) < 0.5, "a", "b")
    return pd.DataFrame({"g": g, "y": rng.normal(0, 1, 400) + (g == "a") * 0.5, "x": rng.normal(0, 1, 400)})


def test_t_test_matches_scipy(df):
    out = t_test(df, "y", "g", "a", "b")
    ref = stats.ttest_ind(df.y[df.g == "a"], df.y[df.g == "b"], equal_var=False)
    assert out["p_value"] == pytest.approx(ref.pvalue) and out["ci95"][0] < out["difference"] < out["ci95"][1]


def test_chi_square_and_corr(df):
    assert chi_square(df.assign(h=df.x > 0), "g", "h")["dof"] == 1
    assert abs(correlation(df, "x", "y")["r"]) < 0.2


def test_sandbox_helper_records_evidence(df):
    st = EvidenceStore()
    r = AnalysisTool().run_helper("t_test", {"value": "y", "group": "g"}, {"Q1": df}, store=st)
    assert r.ok and r.evidence.id == "C1" and st.get("C1").inputs == ["Q1"] and r.figures


@pytest.mark.parametrize("code", ["import os", "open('x')", "x.__class__", "df.to_csv('x')",
                                  "import subprocess", "eval('1')", "pd.read_csv('/etc/passwd')"])
def test_unsafe_code_rejected(code):
    with pytest.raises(UnsafeCode):
        check_code(code)


def test_custom_code_and_timeout(df):
    t = AnalysisTool(timeout_s=3)
    ok = t.run_code("result = {'m': float(Q1.y.mean())}", {"Q1": df})
    assert ok.ok and ok.outputs["m"] == pytest.approx(df.y.mean())
    bad = t.run_code("x = 0\nwhile True:\n    x += 1", {"Q1": df})
    assert not bad.ok
