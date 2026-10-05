import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(autouse=True)
def _runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    from insightforge import tracing

    tracing.set_tracer(None)


@pytest.fixture(scope="session")
def olist_db(tmp_path_factory) -> str:
    d = tmp_path_factory.mktemp("olist")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_olist.py"), "--out", str(d / "csv"),
                    "--orders", "3000"], check=True, capture_output=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/load_olist.py"), "--csv-dir", str(d / "csv"),
                    "--out", str(d / "olist.duckdb")], check=True, capture_output=True)
    return str(d / "olist.duckdb")


@pytest.fixture(scope="session")
def db(olist_db):
    from insightforge.db import open_database

    return open_database(olist_db)
