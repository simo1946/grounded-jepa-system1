"""Scope every osaudit allowance to the one printed number it justifies.

osaudit (from commit 79e3bb7) rejects an [[numbers.allow]] entry that covers more than one printed number: each must
name the file, the line (for a PDF: the page) and the occurrence (the n-th equal value on that page, in the order
osaudit reads them). The builder marks every number it prints from a {{set}}, {{fact}}, {{ext}}, the p-value floor
or a section heading, and records why (for a cited work, the reason names the reference). Here each printed number
that no [[numbers.derive]] entry covers is found in osaudit's own reading of the PDF and matched to the marked number
with the same value whose surrounding text agrees best; it gets one scoped entry with that reason. A number that
cannot be matched unambiguously stops the build.
"""
from __future__ import annotations

import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MARK = re.compile("(\\d+)")         # placed just before a marked number in the rendered text
CONTEXT = 40                                     # characters of context compared on each side
MIN_SCORE = 6                                    # matching characters needed to accept a site


@dataclass(frozen=True)
class Site:
    num: str
    reason: str
    left: str = ""
    right: str = ""


def norm(s: str) -> str:
    """Text as compared: compatibility-normalised, without whitespace or markers."""
    s = unicodedata.normalize("NFKC", MARK.sub("", s))
    return re.sub(r"\s+", "", s)


def contexts(doc: str, sites: dict[int, Site]) -> dict[int, Site]:
    """Fill each site's left and right context from the rendered document text holding the marks."""
    out = {}
    for m in MARK.finditer(doc):
        sid = int(m.group(1))
        s = sites[sid]
        out[sid] = Site(s.num, s.reason, norm(doc[:m.start()])[-CONTEXT:], norm(doc[m.end():])[:CONTEXT])
    missing = set(sites) - set(out)
    if missing:
        raise RuntimeError(f"marked numbers missing from the rendered text: {sorted(missing)[:5]}")
    return out


def _common_suffix(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def _common_prefix(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _pdf_positions(text: str):
    """osaudit's PDF numbers with their page and character span in the page text (same patterns, blanked
    with spaces of equal length so positions survive)."""
    from osaudit import _pdf
    from osaudit._number import make
    for page, body in enumerate(_pdf.pages(text), 1):
        raw = body
        for pat in (_pdf.QUOTE, _pdf.CITATION, _pdf.PART, _pdf.LONE_NUMBER, _pdf.LIST_LABEL):
            body = pat.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), body)
        for m in _pdf.PDF_NUMBER.finditer(body):
            n = make("", page, m.group(0).strip(), m.group(1), m.group(2), m.group(3) is not None)
            yield n, raw, m.start(), m.end()


def scoped_entries(root: Path, toml_without_allow: str, sites: dict[int, Site], pdf_rel: str) -> list[dict]:
    """One scoped allow entry per printed number that no derive entry covers."""
    from osaudit import _pdf
    from osaudit._config import load, parse
    try:
        from osaudit._numbers import _dispositions, paper_numbers
    except ImportError as e:                         # osaudit 0.7.0 from PyPI predates scoped allowances
        raise RuntimeError("the build needs osaudit at commit 79e3bb7 or later (see paper/reproduce.sh): pip install "
                           "'osaudit @ git+https://github.com/ASSERT-KTH/osaudit@79e3bb737266e3162f66e4f213a4e985975bd5fb'"
                           ) from e

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "osaudit.toml"
        cfg_path.write_text(toml_without_allow)
        cfg = load(root, cfg_path)
        nums = list(paper_numbers(cfg))
        hits, _ = _dispositions(cfg)
    located = list(_pdf_positions(_pdf.text_of(root / pdf_rel)))
    if [(n.line, n.text) for n in nums] != [(n.line, n.text) for n, *_ in located]:
        raise RuntimeError("could not reproduce osaudit's reading of the PDF: number lists differ")

    by_key: dict[tuple, list[Site]] = {}
    for s in sites.values():
        p = parse(s.num)
        by_key.setdefault(p.key, []).append(s)

    entries, failures = [], []
    for i, n in enumerate(nums):
        if hits[i]:
            continue                                 # a derive entry traces it to the results
        _, raw, start, end = located[i]
        left, right = norm(raw[:start])[-CONTEXT:], norm(raw[start:])[:CONTEXT]
        cands = by_key.get(n.key, [])
        scored = sorted(((_common_suffix(left, s.left) + _common_prefix(right, s.right), s) for s in cands),
                        key=lambda t: -t[0])
        reasons = {s.reason for _, s in scored}
        if not scored:
            failures.append(f"page {n.line}: {n.text} has no marked setting, fact or citation")
            continue
        best, site = scored[0]
        tied = {s.reason for sc, s in scored if sc == best}
        if len(reasons) > 1 and (best < MIN_SCORE or len(tied) > 1):
            failures.append(f"page {n.line}: {n.text} ({raw[max(0, start - 40):end].strip()!r}) matches "
                            f"{len(tied)} reasons equally ({best} characters)")
            continue
        occurrence = 1 + sum(1 for m in nums[:i] if m.line == n.line and m.key == n.key)
        entries.append({"value": site.num, "reason": site.reason, "file": pdf_rel,
                        "line": n.line, "occurrence": occurrence})
    if failures:
        raise RuntimeError("allowances that cannot be scoped:\n  " + "\n  ".join(failures))
    return entries
