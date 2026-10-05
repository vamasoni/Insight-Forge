"""CLI:  python -m insightforge ask "Do late deliveries get worse reviews?"
        python -m insightforge sql "How many orders per payment type?"
        python -m insightforge schema"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="insightforge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("sql").add_argument("question")
    sub.add_parser("schema")
    a = ap.parse_args(argv)

    from insightforge import runtime

    if a.cmd == "schema":
        print(runtime.database().schema().render())
    elif a.cmd == "sql":
        run = runtime.agent().sql.run(a.question)
        print(run.sql or "", "\n")
        print(run.result.df.head(30).to_string() if run.ok else f"FAILED: {run.error}")
        print(f"\nattempts={run.n_attempts} tables={run.retrieval.tables if run.retrieval else None}")
    else:
        rep = runtime.agent().ask(a.question)
        print(rep.to_markdown())
        print(f"\nsaved to {runtime.settings().runs_dir / rep.run_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
