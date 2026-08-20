"""Fetch/update public matrix-multiplication corpora useful for rank-23 comparison.

GitHub corpora are cloned with git.  The full JKU 3x3 repository is a separate
interactive web corpus, so ``--mirror-jku`` invokes the host-scoped crawler in
``jku_mirror.py``.  If your system rejects the JKU TLS chain, pass
``--insecure-jku`` explicitly; that disables certificate verification only for
``algebra.uni-linz.ac.at`` and does not modify global Python/git TLS settings.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

from jku_mirror import mirror_jku

REPOS = {
    "perminov": "https://github.com/dronperminov/FastMatrixMultiplication.git",
    "matmulcatalog": "https://github.com/solven-eu/matmulcatalog.git",
    "kauers": "https://github.com/mkauers/matrix-multiplication.git",
}


def git_sync(url: str, dst: Path):
    if (dst / ".git").exists():
        subprocess.run(["git", "-C", str(dst), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("external/rank23_corpora"))
    ap.add_argument("--only", choices=[*REPOS, "all"], default="all")
    ap.add_argument("--mirror-jku", action="store_true",
                    help="mirror the separate JKU 17k-scheme web repository")
    ap.add_argument("--insecure-jku", action="store_true",
                    help="disable TLS verification only for algebra.uni-linz.ac.at")
    ap.add_argument("--jku-max-pages", type=int, default=50000)
    ap.add_argument("--jku-max-files", type=int, default=0)
    ap.add_argument("--jku-delay", type=float, default=0.03)
    ap.add_argument("--jku-timeout", type=float, default=60.0)
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
    if args.mirror_jku:
        try:
            summary = mirror_jku(
                args.out / "jku",
                insecure_jku=args.insecure_jku,
                max_pages=args.jku_max_pages,
                max_files=args.jku_max_files,
                delay=args.jku_delay,
                timeout=args.jku_timeout,
            )
            status["jku"] = summary
        except Exception as e:
            status["jku"] = f"failed: {e}"
    (args.out / "fetch_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
