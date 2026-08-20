"""Fetch/update public matrix-multiplication corpora useful for rank-23 comparison.

This helper intentionally keeps third-party repositories outside the project source tree by
default. It uses git when available. The JKU 17,376-scheme repository is an interactive site;
we save its landing page when direct HTTP access works, but do not pretend that this alone is
a complete mirror. The equivalence scanner can consume any downloaded JSON trees directly.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

REPOS = {
    "perminov": "https://github.com/dronperminov/FastMatrixMultiplication.git",
    "matmulcatalog": "https://github.com/solven-eu/matmulcatalog.git",
    "kauers": "https://github.com/mkauers/matrix-multiplication.git",
}
JKU = "https://www.algebra.uni-linz.ac.at/research/matrix-multiplication/"


def git_sync(url: str, dst: Path):
    if (dst / ".git").exists():
        subprocess.run(["git", "-C", str(dst), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("external/rank23_corpora"))
    ap.add_argument("--only", choices=[*REPOS, "all"], default="all")
    ap.add_argument("--skip-jku", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = REPOS if args.only == "all" else {args.only: REPOS[args.only]}
    status = {}
    for name, url in chosen.items():
        try:
            git_sync(url, args.out / name)
            status[name] = "ok"
        except Exception as e:
            status[name] = f"failed: {e}"
    if not args.skip_jku:
        try:
            req = Request(JKU, headers={"User-Agent": "rank23-equivalence-research/0.1"})
            data = urlopen(req, timeout=60).read()
            (args.out / "jku_landing.html").write_bytes(data)
            status["jku_landing"] = f"ok ({len(data)} bytes; interactive repository, not full mirror)"
        except Exception as e:
            status["jku_landing"] = f"failed: {e}"
    (args.out / "fetch_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
