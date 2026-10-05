"""Download and unzip the BIRD dev set into data/bird/dev.

    python scripts/download_bird.py

If the URL has moved, get dev.zip from https://bird-bench.github.io/ and unzip it so that
data/bird/dev/dev.json and data/bird/dev/dev_databases/ exist.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
OUT = Path("data/bird")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zpath = OUT / "dev.zip"
    if not zpath.exists():
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(OUT)
    # normalise folder name (archives have shipped as dev/ and dev_20240627/)
    cands = [p for p in OUT.iterdir() if p.is_dir() and p.name.startswith("dev") and (p / "dev.json").exists()]
    if not cands:
        sys.exit("dev.json not found after unzip; check the archive layout")
    if cands[0].name != "dev":
        shutil.move(str(cands[0]), OUT / "dev")
    dev = OUT / "dev"
    inner = dev / "dev_databases.zip"
    if inner.exists() and not (dev / "dev_databases").exists():
        with zipfile.ZipFile(inner) as z:
            z.extractall(dev)
    n = len(list((dev / "dev_databases").glob("*/*.sqlite")))
    print(f"ready: {dev} ({n} databases)")


if __name__ == "__main__":
    main()
