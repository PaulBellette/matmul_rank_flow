"""Mirror the public JKU 3x3 matrix-multiplication scheme repository.

The JKU site currently presents an interactive repository at
https://www.algebra.uni-linz.ac.at/research/matrix-multiplication/.  Some
systems reject its TLS chain.  This module therefore supports an *explicit*,
host-scoped ``--insecure-jku`` mode: certificate verification is disabled only
for requests to ``algebra.uni-linz.ac.at``.  Redirects to any other host are
rejected.

The crawler is deliberately content-driven rather than assuming a particular
2019-era PHP layout.  It follows same-host links, iframe/script/form targets,
option values, and URL-looking strings embedded in HTML/JS.  Any response that
looks like a 23-summand 3x3 trilinear scheme is saved as a canonical ``.exp``
file, even if the server uses an extensionless/query-string endpoint.

Every fetched URL is recorded in ``jku_manifest.jsonl`` and the crawl is
resumable.  The mirror can therefore be audited independently of the parser.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import ssl
import time
from collections import deque
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)

JKU_BASE = "https://www.algebra.uni-linz.ac.at/research/matrix-multiplication/"
JKU_HOST = "www.algebra.uni-linz.ac.at"
DEFAULT_PREFIX = "/research/matrix-multiplication/"
UA = "rank23-equivalence-research/0.3 (+public academic corpus mirroring)"

SCHEME_TOKEN_RE = re.compile(r"\b([abc])(\d+)(\d+)\b", re.I)
URL_STRING_RE = re.compile(
    r"(?P<q>['\"])(?P<u>(?:https?://[^'\"<>\s]+|(?:\.{0,2}/)?[A-Za-z0-9_./-]+(?:\?[A-Za-z0-9_=&%+.,:/-]+)?))(?P=q)"
)
ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz", ".bz2", ".gz", ".xz")
STATIC_EXTS = (".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".woff2", ".ttf")


class _LinkParser(HTMLParser):
    """Collect URL-bearing attributes, including option values/iframes/forms."""

    URL_ATTRS = {"href", "src", "action", "data-src", "data-url", "value"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key.lower() in self.URL_ATTRS and value:
                self.links.append(value.strip())


def _normalise_url(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    # Fragments never affect server content.  Preserve queries because the JKU
    # repository may use dynamic endpoints keyed by query parameters.
    path = re.sub(r"/{2,}", "/", p.path or "/")
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, p.params, p.query, ""))


def _allowed_url(url: str, *, host: str = JKU_HOST, prefix: str = DEFAULT_PREFIX) -> bool:
    p = urlparse(url)
    if p.scheme not in {"http", "https"} or p.hostname != host:
        return False
    if not p.path.startswith(prefix):
        return False
    low = p.path.lower()
    if low.endswith(STATIC_EXTS):
        return False
    return True


def extract_links(text: str, base_url: str) -> list[str]:
    """Extract ordinary and script-embedded same-site URL candidates."""
    parser = _LinkParser()
    try:
        parser.feed(text)
    except Exception:
        pass
    raw = list(parser.links)
    # Old interactive sites often construct select/iframe URLs in inline JS.
    # Scan quoted strings too, but require them to look URL/path-like.
    for m in URL_STRING_RE.finditer(text):
        u = html.unescape(m.group("u"))
        if "/" in u or "?" in u or u.endswith((".php", ".html", ".htm", ".exp", ".txt", ".js")):
            raw.append(u)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        item = item.strip()
        if not item or item.startswith(("javascript:", "mailto:", "#")):
            continue
        # <option value="123"> is not intrinsically a URL.  Ignore bare tokens.
        if not (":" in item or "/" in item or "?" in item or "." in item):
            continue
        try:
            u = _normalise_url(urljoin(base_url, item))
        except Exception:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def canonical_scheme_lines(data: bytes) -> list[str] | None:
    """Return 23 candidate .exp lines when a response looks like a 3x3 scheme.

    This intentionally does not prove the Brent identities; the existing exact
    ``load_scheme_exp`` parser/search pipeline does that downstream.  Here we
    only recognise the repository's trilinear-product representation robustly.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except Exception:
            return None
    lines = [x.strip().rstrip(";") for x in text.splitlines()
             if x.strip() and not x.lstrip().startswith(("#", "//", "%"))]
    # Handle simple HTML <pre> containers without depending on BeautifulSoup.
    # Prefer the PRE payload even when the surrounding HTML happens to produce
    # exactly 23 physical lines.
    if "<pre" in text.lower():
        pres = re.findall(r"<pre[^>]*>(.*?)</pre>", text, flags=re.I | re.S)
        for pre in pres:
            plain = re.sub(r"<[^>]+>", "", html.unescape(pre))
            cand = [x.strip().rstrip(";") for x in plain.splitlines() if x.strip()]
            if len(cand) == 23:
                lines = cand
                break
    if len(lines) != 23:
        return None
    joined = "\n".join(lines)
    toks = SCHEME_TOKEN_RE.findall(joined)
    if not toks:
        return None
    if any(int(i) > 3 or int(j) > 3 or int(i) < 1 or int(j) < 1 for _, i, j in toks):
        return None
    # Each summand should visibly involve all three tensor legs.
    for line in lines:
        low = line.lower()
        if not all(re.search(rf"\b{p}[1-3][1-3]\b", low) for p in "abc"):
            return None
    return lines


def _content_ext(url: str, content_type: str | None) -> str:
    path = urlparse(url).path.lower()
    for ext in ARCHIVE_EXTS:
        if path.endswith(ext):
            return ext
    if path.endswith(".exp"):
        return ".exp"
    if path.endswith(".json"):
        return ".json"
    if path.endswith(".js") or (content_type and "javascript" in content_type):
        return ".js"
    if path.endswith((".html", ".htm")) or (content_type and "html" in content_type):
        return ".html"
    if path.endswith(".txt") or (content_type and content_type.startswith("text/plain")):
        return ".txt"
    return ".bin"


class _SameHostRedirect(HTTPRedirectHandler):
    def __init__(self, host: str):
        super().__init__()
        self.host = host

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        target = urljoin(req.full_url, newurl)
        if urlparse(target).hostname != self.host:
            raise HTTPError(target, code, f"cross-host redirect rejected: {target}", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, target)


def build_jku_opener(*, insecure_jku: bool):
    if insecure_jku:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx = ssl.create_default_context()
    return build_opener(_SameHostRedirect(JKU_HOST), HTTPSHandler(context=ctx))


@dataclass
class ManifestRow:
    url: str
    status: str
    final_url: str | None = None
    http_status: int | None = None
    content_type: str | None = None
    bytes: int = 0
    sha256: str | None = None
    saved: str | None = None
    discovered_links: int = 0
    error: str | None = None


def _load_manifest(path: Path) -> dict[str, ManifestRow]:
    out: dict[str, ManifestRow] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            row = ManifestRow(**obj)
            out[row.url] = row
        except Exception:
            continue
    return out


def _append_manifest(path: Path, row: ManifestRow) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def mirror_jku(
    out: Path,
    *,
    insecure_jku: bool = False,
    base_url: str = JKU_BASE,
    max_pages: int = 50000,
    max_files: int = 0,
    delay: float = 0.03,
    timeout: float = 60.0,
    refresh: bool = False,
) -> dict:
    """Mirror discoverable JKU repository content and return a summary."""
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    scheme_dir = out / "schemes"
    raw_dir.mkdir(exist_ok=True)
    scheme_dir.mkdir(exist_ok=True)
    manifest_path = out / "jku_manifest.jsonl"
    prior = {} if refresh else _load_manifest(manifest_path)
    opener = build_jku_opener(insecure_jku=insecure_jku)

    q: deque[str] = deque([_normalise_url(base_url)])
    queued = set(q)
    visited = 0
    saved_schemes = 0
    downloaded_files = 0
    errors = 0

    while q and visited < max_pages:
        url = q.popleft()
        if not _allowed_url(url):
            continue
        old = prior.get(url)
        if old and old.status in {"scheme", "saved"}:
            continue
        if old and old.status == "page" and old.saved:
            # Resume without a network refetch: replay links from the saved page
            # so unfinished descendants are re-enqueued.
            saved_page = out / old.saved
            if saved_page.exists():
                try:
                    text = saved_page.read_text(errors="replace")
                    base = old.final_url or url
                    for child in extract_links(text, base):
                        if _allowed_url(child) and child not in queued and child not in prior:
                            queued.add(child)
                            q.append(child)
                    continue
                except Exception:
                    pass
        visited += 1
        try:
            req = Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with opener.open(req, timeout=timeout) as resp:
                data = resp.read()
                final_url = _normalise_url(resp.geturl())
                if urlparse(final_url).hostname != JKU_HOST:
                    raise RuntimeError(f"cross-host final URL rejected: {final_url}")
                ctype = (resp.headers.get_content_type() if resp.headers else None)
                status = getattr(resp, "status", 200)
            digest = hashlib.sha256(data).hexdigest()
            scheme_lines = canonical_scheme_lines(data)
            links: list[str] = []
            saved: Path | None = None
            row_status = "saved"

            if scheme_lines is not None:
                saved = scheme_dir / f"scheme-{digest[:20]}.exp"
                saved.write_text("\n".join(scheme_lines) + "\n")
                row_status = "scheme"
                saved_schemes += 1
                downloaded_files += 1
            else:
                ext = _content_ext(final_url, ctype)
                is_text_page = ext in {".html", ".js", ".txt"}
                if is_text_page:
                    try:
                        text = data.decode("utf-8", "replace")
                        links = [u for u in extract_links(text, final_url) if _allowed_url(u)]
                    except Exception:
                        links = []
                    # Preserve fetched pages/scripts for audit/debugging.
                    saved = raw_dir / f"page-{digest[:20]}{ext}"
                    saved.write_bytes(data)
                    row_status = "page"
                else:
                    saved = raw_dir / f"file-{digest[:20]}{ext}"
                    saved.write_bytes(data)
                    downloaded_files += 1
                    row_status = "saved"

            for child in links:
                if child not in queued and child not in prior:
                    queued.add(child)
                    q.append(child)

            row = ManifestRow(
                url=url, status=row_status, final_url=final_url,
                http_status=int(status), content_type=ctype, bytes=len(data),
                sha256=digest, saved=str(saved.relative_to(out)) if saved else None,
                discovered_links=len(links),
            )
            _append_manifest(manifest_path, row)

            if max_files and downloaded_files >= max_files:
                break
        except Exception as e:
            errors += 1
            _append_manifest(manifest_path, ManifestRow(url=url, status="error", error=str(e)))
        if delay:
            time.sleep(delay)

    # Deduplicate count over the whole manifest, not just this invocation.
    all_rows = _load_manifest(manifest_path)
    unique_schemes = {r.sha256 for r in all_rows.values() if r.status == "scheme" and r.sha256}
    summary = {
        "base_url": base_url,
        "insecure_jku": insecure_jku,
        "urls_in_manifest": len(all_rows),
        "visited_this_run": visited,
        "unique_schemes": len(unique_schemes),
        "schemes_saved_this_run": saved_schemes,
        "errors_this_run": errors,
        "queue_remaining": len(q),
        "manifest": str(manifest_path),
        "schemes_dir": str(scheme_dir),
    }
    (out / "jku_mirror_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("external/rank23_corpora/jku"))
    ap.add_argument("--insecure-jku", action="store_true",
                    help="disable TLS verification ONLY for algebra.uni-linz.ac.at")
    ap.add_argument("--max-pages", type=int, default=50000,
                    help="maximum URLs fetched this invocation; 0 is not supported")
    ap.add_argument("--max-files", type=int, default=0,
                    help="stop after this many non-page downloads (0 = unlimited)")
    ap.add_argument("--delay", type=float, default=0.03,
                    help="polite delay between requests in seconds")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore prior manifest completion states")
    ap.add_argument("--base-url", default=JKU_BASE,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    summary = mirror_jku(
        args.out,
        insecure_jku=args.insecure_jku,
        base_url=args.base_url,
        max_pages=args.max_pages,
        max_files=args.max_files,
        delay=args.delay,
        timeout=args.timeout,
        refresh=args.refresh,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
