"""Child-process entry point for the analysis sandbox. Not imported by the main process."""
from __future__ import annotations

import builtins
import contextlib
import io
import json
import pickle
import sys
from pathlib import Path


def _limit(memory_mb: int, cpu_s: int) -> None:
    try:
        import resource  # POSIX only; on Windows the parent's timeout is the guard
    except ImportError:
        return
    mem = memory_mb * 1024 * 1024
    for lim, val in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, cpu_s),
                     (resource.RLIMIT_FSIZE, 50 * 1024 * 1024)):
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(lim, (val, val))


_SAFE_BUILTINS = ["abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max", "min",
                  "range", "round", "set", "sorted", "str", "sum", "tuple", "zip", "isinstance", "print", "map",
                  "filter", "reversed", "Exception", "ValueError", "KeyError", "TypeError", "ZeroDivisionError",
                  "None", "True", "False", "divmod", "pow", "frozenset", "slice"]


def _default_figure(method: str, df, args: dict, out: dict):
    import plotly.express as px

    try:
        if method == "t_test":
            d = df[df[args["group"]].astype(str).isin(out["groups"])]
            return px.box(d, x=args["group"], y=args["value"], title=f"{args['value']} by {args['group']}")
        if method == "correlation":
            return px.scatter(df, x=args["x"], y=args["y"], title=f"{args['y']} vs {args['x']} (r={out['r']:.2f})")
        if method == "describe" and args.get("group"):
            import pandas as pd

            t = pd.DataFrame(out["by_group"])
            return px.bar(t, x=args["group"], y="mean", title=f"Mean {args['value']} by {args['group']}")
        if method == "anomalies" and args.get("time"):
            return px.line(df.sort_values(args["time"]), x=args["time"], y=args["value"], title="Series with anomalies")
    except Exception:
        return None
    return None


def main(workdir: str, memory_mb: str, cpu_s: str) -> None:
    wd = Path(workdir)
    job = json.loads((wd / "job.json").read_text())
    with (wd / "frames.pkl").open("rb") as f:
        frames = pickle.load(f)
    _limit(int(memory_mb), int(cpu_s))
    stdout = io.StringIO()
    result: dict = {"ok": False}
    try:
        from insightforge.tools import stats_lib

        with contextlib.redirect_stdout(stdout):
            if job["mode"] == "helper":
                df = frames[job["input"]]
                outputs = stats_lib.REGISTRY[job["method"]](df, **job["args"])
                fig = _default_figure(job["method"], df, job["args"], outputs)
                figs = [fig] if fig is not None else []
            else:
                from insightforge.tools.analysis_tool import ALLOWED_IMPORTS

                real_import = builtins.__import__

                def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                    top = name if name in ALLOWED_IMPORTS else None
                    if top is None and globals is not None and globals.get("__name__") == "__sandbox__":
                        raise ImportError(f"import of {name!r} is not allowed")
                    return real_import(name, globals, locals, fromlist, level)

                import numpy as np
                import pandas as pd
                import plotly.express as px
                import plotly.graph_objects as go
                from scipy import stats

                safe = {k: getattr(builtins, k) for k in _SAFE_BUILTINS if hasattr(builtins, k)}
                safe["__import__"] = guarded_import
                ns = {"__name__": "__sandbox__", "__builtins__": safe, "pd": pd, "np": np, "stats": stats,
                      "px": px, "go": go, "sl": stats_lib, **{k: v.copy() for k, v in frames.items()}}
                exec(compile(job["code"], "<analysis>", "exec"), ns)  # noqa: S102 - AST-checked above
                outputs = ns.get("result")
                if not isinstance(outputs, dict):
                    raise ValueError("code must assign a dict to `result`")
                f = ns.get("fig")
                figs = [] if f is None else (list(f) if isinstance(f, (list, tuple)) else [f])
        result = {"ok": True, "outputs": outputs, "figures": [g.to_json() for g in figs]}
    except MemoryError:
        result = {"ok": False, "error": "analysis ran out of memory"}
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    result["stdout"] = stdout.getvalue()[-4000:]
    (wd / "result.json").write_text(json.dumps(result, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
