#!/usr/bin/env python3
"""
Mirror an Apache AutoIndex tree without following its ?C=... sort links.

Designed for:
  http://www.algebra.uni-linz.ac.at/people/mkauers/matrix-mult/solutions/

It stores only real leaf files, not directory index pages, so the resulting
tree can be handed directly to corpus_audit.py / rank23_equivalence_search.py.

Standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import html.parser
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from pathlib import Path

DEFAULT_BASE = "http://www.algebra.uni-linz.ac.at/people/mkauers/matrix-mult/solutions/"


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)


def canonical_url(url: str) -> str:
    """Drop query/fragment and normalise path without changing trailing slash."""
    p = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(urllib.parse.unquote(p.path), safe="/:@!$&'()*+,;=-._~")
    return urllib.parse.urlunsplit((p.scheme, p.netloc, path, "", ""))


def is_under_base(url: str, base: str) -> bool:
    u = urllib.parse.urlsplit(url)
    b = urllib.parse.urlsplit(base)
    return (
        u.scheme == b.scheme
        and u.netloc == b.netloc
        and u.path.startswith(b.path)
    )


def relative_leaf_path(url: str, base: str) -> Path:
    up = urllib.parse.urlsplit(url)
    bp = urllib.parse.urlsplit(base)
    rel = urllib.parse.unquote(up.path[len(bp.path):]).lstrip("/")
    return Path(rel)


def fetch(url: str, timeout: float, user_agent: str) -> tuple[str, bytes, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "*/*",
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        ctype = resp.headers.get_content_type() or "application/octet-stream"
        final_url = canonical_url(resp.geturl())
        return final_url, data, ctype


def parse_directory_links(page_url: str, body: bytes) -> list[str]:
    text = body.decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(text)

    out: list[str] = []
    for href in parser.hrefs:
        href = href.strip()

        # The entire reason this script exists.
        if not href or href.startswith(("?", "#", "mailto:", "javascript:")):
            continue

        # AutoIndex parent link.
        if href in ("../", ".."):
            continue

        child = canonical_url(urllib.parse.urljoin(page_url, href))
        out.append(child)

    return out


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: dict, lock: threading.Lock) -> None:
    line = json.dumps(obj, sort_keys=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_manifest(path: Path) -> tuple[set[str], set[str]]:
    done_dirs: set[str] = set()
    done_files: set[str] = set()
    if not path.exists():
        return done_dirs, done_files
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") == "dir_ok":
                done_dirs.add(row["url"])
            elif row.get("status") in {"file_ok", "file_exists"}:
                done_files.add(row["url"])
    return done_dirs, done_files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.02,
                    help="small per-request politeness delay in seconds")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore previous manifest (existing leaf files are still reused)")
    args = ap.parse_args()

    base = canonical_url(args.base)
    if not base.endswith("/"):
        base += "/"

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "mirror_manifest.jsonl"
    summary_path = out / "mirror_summary.json"

    if args.fresh and manifest.exists():
        manifest.rename(out / f"mirror_manifest.{int(time.time())}.jsonl")

    done_dirs, done_files = load_manifest(manifest)
    lock = threading.Lock()
    stop = threading.Event()

    stats = Counter()
    stats["dirs_from_manifest"] = len(done_dirs)
    stats["files_from_manifest"] = len(done_files)

    def on_signal(signum, frame):
        stop.set()
        print("\nStopping cleanly after current requests...", file=sys.stderr)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    seen_dirs = set(done_dirs)
    seen_files = set(done_files)
    pending_dirs = deque([base] if base not in done_dirs else [])

    # Resume subtlety: a completed directory's child links are not stored separately.
    # If base is marked done, replay downloaded manifest tree from scratch by refetching
    # only directory pages. This is cheap and avoids silently stopping on resume.
    if not pending_dirs:
        pending_dirs.append(base)
        seen_dirs.discard(base)

    user_agent = "rank23-jku-mirror/1.0 (research corpus mirroring; low concurrency)"

    def fetch_retry(url: str):
        last = None
        for attempt in range(args.retries):
            if stop.is_set():
                raise RuntimeError("stopping")
            try:
                if args.delay:
                    time.sleep(args.delay)
                return fetch(url, args.timeout, user_agent)
            except Exception as e:
                last = e
                time.sleep(min(2 ** attempt, 4))
        raise last  # type: ignore[misc]

    def process_dir(url: str):
        final_url, body, ctype = fetch_retry(url)
        # Apache index should be HTML. If a slash URL unexpectedly gives a file,
        # preserve it with a synthetic name rather than losing data.
        links = parse_directory_links(final_url, body)
        return url, final_url, body, ctype, links

    def process_file(url: str):
        rel = relative_leaf_path(url, base)
        target = out / "files" / rel

        if target.exists() and target.stat().st_size > 0:
            return url, url, None, None, target, "exists"

        final_url, body, ctype = fetch_retry(url)
        # If a "file" redirects outside base, reject.
        if not is_under_base(final_url, base):
            raise RuntimeError(f"redirect outside base: {final_url}")
        atomic_write(target, body)
        return url, final_url, body, ctype, target, "downloaded"

    active: dict[cf.Future, tuple[str, str]] = {}

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        while (pending_dirs or active) and not stop.is_set():
            # Fill worker pool with directories first. Directory discovery is the
            # frontier; file downloads share the same pool as they are discovered.
            while pending_dirs and len(active) < args.workers and not stop.is_set():
                u = pending_dirs.popleft()
                if u in seen_dirs:
                    continue
                seen_dirs.add(u)
                fut = ex.submit(process_dir, u)
                active[fut] = ("dir", u)
                stats["dirs_submitted"] += 1

            if not active:
                break

            done, _ = cf.wait(active, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                kind, requested = active.pop(fut)
                try:
                    result = fut.result()
                except Exception as e:
                    stats[f"{kind}_errors"] += 1
                    append_jsonl(
                        manifest,
                        {
                            "kind": kind,
                            "url": requested,
                            "status": "error",
                            "error": repr(e),
                            "time": time.time(),
                        },
                        lock,
                    )
                    continue

                if kind == "dir":
                    _, final_url, body, ctype, links = result
                    stats["dirs_ok"] += 1
                    stats["links_seen"] += len(links)
                    append_jsonl(
                        manifest,
                        {
                            "kind": "dir",
                            "url": requested,
                            "final_url": final_url,
                            "status": "dir_ok",
                            "links": len(links),
                            "bytes": len(body),
                            "time": time.time(),
                        },
                        lock,
                    )

                    for child in links:
                        if not is_under_base(child, base):
                            stats["links_outside_base"] += 1
                            continue

                        p = urllib.parse.urlsplit(child)
                        # canonical_url already removed query; this is defensive.
                        if p.query or p.fragment:
                            stats["links_query_or_fragment"] += 1
                            continue

                        if child.endswith("/"):
                            if child not in seen_dirs:
                                pending_dirs.append(child)
                        else:
                            if child in seen_files:
                                continue
                            seen_files.add(child)
                            fut2 = ex.submit(process_file, child)
                            active[fut2] = ("file", child)
                            stats["files_submitted"] += 1

                else:
                    _, final_url, body, ctype, target, how = result
                    if how == "exists":
                        stats["files_exists"] += 1
                        status = "file_exists"
                        nbytes = target.stat().st_size
                        sha256 = None
                    else:
                        stats["files_downloaded"] += 1
                        status = "file_ok"
                        nbytes = len(body)
                        sha256 = hashlib.sha256(body).hexdigest()

                    append_jsonl(
                        manifest,
                        {
                            "kind": "file",
                            "url": requested,
                            "final_url": final_url,
                            "status": status,
                            "path": str(target.relative_to(out)),
                            "bytes": nbytes,
                            "content_type": ctype,
                            "sha256": sha256,
                            "time": time.time(),
                        },
                        lock,
                    )

            # concise progress every ~250 completed objects
            completed = stats["dirs_ok"] + stats["files_downloaded"] + stats["files_exists"]
            if completed and completed % 250 < len(done):
                print(
                    f"progress dirs={stats['dirs_ok']} "
                    f"files={stats['files_downloaded'] + stats['files_exists']} "
                    f"queued_dirs={len(pending_dirs)} active={len(active)} "
                    f"errors={stats['dir_errors'] + stats['file_errors']}",
                    file=sys.stderr,
                )

    summary = {
        "base": base,
        "out": str(out),
        "stopped": stop.is_set(),
        **dict(stats),
        "unique_dirs_seen": len(seen_dirs),
        "unique_files_seen": len(seen_files),
        "queue_remaining": len(pending_dirs),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 130 if stop.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
