"""Build the paper from paper.md and the results files, so that every printed number comes from a results file.

    python paper/build.py            # -> paper/paper.docx, paper/paper.pdf, osaudit.toml

paper.md is plain text with a few markers:

    TITLE: / AUTHOR: / AFFIL: / ABSTRACT:   title block (one line each)
    # 1. Heading, ## 5.1 Heading           section headings
    - item                                 bullet
    TABLE: name                            a table from tables.py
    FIGURE: file.png | width in inches | caption
    REFERENCES                             start of the reference list (one reference per line)
    **bold**, *italic*

and placeholders, which are the only way a number enters the text:

    {{pct0: expr}}    a share in percent with 0 decimals and a % sign ("55%"); expr is in percent units
    {{num1: expr}}    a plain number with 1 decimal ("16.7"); {{num0: expr}} requires an integral value
    {{sgn1: expr}}    a signed difference with 1 decimal ("+16.7", "−19.4")
    {{dec1: expr}}    a number that may be negative, minus sign only ("−3.3", "8.9"), e.g. interval ends
    {{int: expr}}     a count (the value must be a JSON integer)
    {{p: expr}}       a p-value ("0.006", "0.39", "1", or "< 0.001")
    {{set: 64 | why}}       a setting of the study (an osaudit allow entry with that reason)
    {{ext: 42 | [22] | why}} a number taken from a cited work (allow entry citing the reference)
    {{fact: 3,000 | why}}   a recorded fact that no results file holds (allow entry with that reason)

expr names a value of a results file in results/paper/ as osaudit does: <file stem>.<key>... with ["key"] for
keys that are not identifiers and [i] for list indices, combined with + - * / and parentheses. Each number
printed from an expression is written to osaudit.toml as a [[numbers.derive]] entry with the same expression,
so osaudit checks every printed number against the results independently of this script.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results" / "paper"
MINUS = "−"
P_FLOOR = 0.001
SCHEMA = "https://github.com/ASSERT-KTH/assert-research-agent/blob/main/schemas/audit-v1.json"


# ---------------------------------------------------------------------------------------------- values
def load_values() -> dict:
    vals = {}
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text())
        d.pop("provenance", None)
        vals[f.stem] = d
    return vals


class Expr:
    """Evaluate an osaudit-style expression over the results tree (names, [key], [i], + - * /, numbers)."""

    def __init__(self, values: dict):
        self.values = values

    def __call__(self, text: str):
        tree = ast.parse(text.strip(), mode="eval")
        return self._eval(tree.body)

    def _eval(self, node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if b == 0:
                raise ZeroDivisionError(ast.unparse(node))
            return a / b
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        return self._lookup(node)

    def _lookup(self, node):
        if isinstance(node, ast.Name):
            if node.id not in self.values:
                raise KeyError(f"no results file {node.id}.json in {RESULTS}")
            return self.values[node.id]
        if isinstance(node, ast.Attribute):
            return self._lookup(node.value)[node.attr]
        if isinstance(node, ast.Subscript):
            key = node.slice
            if not isinstance(key, ast.Constant):
                raise ValueError(f"index must be a constant: {ast.unparse(node)}")
            return self._lookup(node.value)[key.value]
        raise ValueError(f"not allowed in an expression: {ast.unparse(node)}")


# ---------------------------------------------------------------------------------------------- formatting
@dataclass
class Ledger:
    """What osaudit must be told: one derive entry per printed number, one allow entry per setting/citation."""
    derive: dict[tuple[str, str], None] = field(default_factory=dict)
    allow: dict[tuple[str, str, str], None] = field(default_factory=dict)

    def toml(self) -> str:
        out = []
        for value, expr in self.derive:
            out += ["[[numbers.derive]]", f"value = {json.dumps(value)}", f"expr = {json.dumps(expr)}", ""]
        for value, reason, cite in self.allow:
            out += ["[[numbers.allow]]", f"value = {json.dumps(value)}", f"reason = {json.dumps(reason)}"]
            out += [f"cite = {json.dumps(cite)}"] if cite else []
            out += [""]
        return "\n".join(out)


def _dec(x: float, d: int) -> str:
    """x rounded half-up to d decimals, from its shortest repr (so 0.125 -> 0.13, as osaudit's half-unit test
    accepts both neighbours of a tie)."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{q:f}"


class Formatter:
    def __init__(self, values: dict, ledger: Ledger, draft: bool = False):
        self.ev, self.ledger, self.draft, self.pending = Expr(values), ledger, draft, []

    def _record(self, printed: str, expr: str, value: float) -> None:
        """One derive entry per printed number (as osaudit reads it: no sign, % kept). osaudit reads printed
        numbers without their sign, so a negative value is derived as 0 - (expr)."""
        self.ledger.derive[(printed, expr if value >= 0 else f"0 - ({expr})")] = None

    def fmt(self, kind: str, arg: str) -> str:
        if kind == "set":
            num, why = (s.strip() for s in arg.split("|", 1))
            self.ledger.allow[(num, f"setting of the study: {why}", "")] = None
            return num
        if kind == "fact":
            num, why = (s.strip() for s in arg.split("|", 1))
            self.ledger.allow[(num, why, "")] = None
            return num
        if kind == "ext":
            num, key, why = (s.strip() for s in arg.split("|", 2))
            self.ledger.allow[(num, f"reported by the cited work: {why}", key)] = None
            return num
        expr = arg.strip()
        try:
            v = self.ev(expr)
        except KeyError:
            if not self.draft:
                raise
            self.pending.append(expr)                      # a draft shows what is still missing, visibly
            return f"[pending: {expr}]"
        if isinstance(v, (dict, list)) or v is None:
            raise ValueError(f"{expr} is not a number")
        m = re.fullmatch(r"(pct|num|sgn|dec)(\d)", kind)
        if kind == "int":
            if not isinstance(v, int):
                raise ValueError(f"{{{{int: {expr}}}}} needs a JSON integer, got {v!r}")
            s = f"{v:,}" if abs(v) >= 10_000 else str(v)
            self._record(s.replace(",", ""), expr, v)
            return s
        if kind == "p":
            if v < P_FLOOR:
                why = "reporting floor for p-values (smaller values are printed as < 0.001)"
                self.ledger.allow[(str(P_FLOOR), why, "")] = None
                return f"< {P_FLOOR}"
            s = "1" if v == 1 else _dec(v, 2) if v >= 0.0995 else _dec(v, 3)
            if s in ("1", "1.00") and v != 1:
                s = _dec(v, 3)
            self._record(s, expr, v)
            return s
        if m is None:
            raise ValueError(f"unknown format {kind}")
        k, d = m.group(1), int(m.group(2))
        if k in ("num", "pct") and v < 0:
            raise ValueError(f"{expr} = {v} is negative: use sgn (always signed) or dec (minus sign only)")
        if k == "num" and d == 0 and float(v) != int(v):
            raise ValueError(f"{{{{num0: {expr}}}}} = {v}: osaudit matches a plain integer only to an integral value; "
                             "use num1 or pct0")
        s = _dec(abs(v) if k in ("sgn", "dec") else v, d)
        if k == "pct":
            self._record(s + "%", expr, v)
            return s + "%"
        self._record(s, expr, v)
        if k == "sgn":
            return ("+" if v > 0 else MINUS if v < 0 else "") + s
        if k == "dec":
            return (MINUS if v < 0 else "") + s
        return s

    def text(self, s: str) -> str:
        return re.sub(r"\{\{\s*(\w+)\s*:(.*?)\}\}", lambda m: self.fmt(m.group(1), m.group(2)), s)


# ---------------------------------------------------------------------------------------------- inline runs
def runs(text: str) -> list[dict]:
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^)", text):
        if not part:
            continue
        if part.startswith("**"):
            out.append({"t": part[2:-2], "b": True})
        elif part.startswith("*"):
            out.append({"t": part[1:-1], "i": True})
        elif part.startswith("^"):
            out.append({"t": part[1:-1], "sup": True})
        else:
            out.append({"t": part})
    return out


# ---------------------------------------------------------------------------------------------- document
def parse(md: str, f: Formatter, tables) -> dict:
    meta, blocks, para, in_refs = {}, [], [], False

    def flush():
        if para:
            text = f.text(" ".join(para))
            blocks.append({"type": "ref" if in_refs else "p", "runs": runs(text)})
            para.clear()

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("<!--"):
            continue
        if not line.strip():
            flush()
            continue
        head = re.match(r"(TITLE|AUTHOR|AFFIL|ABSTRACT):\s*(.*)", line)
        if head:
            flush()
            k, v = head.groups()
            if k == "ABSTRACT":
                blocks.append({"type": "abstract", "runs": runs(f.text(v))})
            else:
                meta[k.lower()] = f.text(v)
            continue
        if line.startswith("## ") or line.startswith("# "):
            flush()
            in_refs = False
            title = f.text(line.lstrip("# "))
            num = re.match(r"(?:Appendix )?([A-Z]?\d+(?:\.\d+)?|[A-Z]\.\d+)\.? ", title)
            if num and re.fullmatch(r"\d+\.\d+", num.group(1)):   # "5.2 Heading" reads as a number to osaudit
                f.ledger.allow[(num.group(1), "section number", "")] = None
            blocks.append({"type": "h2" if line.startswith("## ") else "h1", "text": title})
            continue
        if line == "REFERENCES":
            flush()
            blocks.append({"type": "h1", "text": "References"})
            in_refs = True
            continue
        if line.startswith("CODE: "):                     # one line of code, shown verbatim (never a placeholder)
            flush()
            blocks.append({"type": "code", "text": line[6:]})
            continue
        if line.startswith("TABLE: "):
            flush()
            blocks.append(tables.build(line[7:].strip(), f))
            continue
        if line.startswith("FIGURE: "):
            flush()
            path, width, caption = (s.strip() for s in line[8:].split("|", 2))
            from PIL import Image
            with Image.open(PAPER / path) as im:
                aspect = im.height / im.width
            blocks.append({"type": "figure", "path": path, "width_px": float(width) * 96, "aspect": aspect,
                           "alt": re.sub(r"\*", "", caption)[:200], "caption": runs(f.text(caption))})
            continue
        if line.startswith("- "):
            flush()
            blocks.append({"type": "bullet", "runs": runs(f.text(line[2:]))})
            continue
        numbered = re.match(r"(\d+)\. (.*)", line)
        if numbered and not in_refs:
            flush()
            blocks.append({"type": "number", "runs": runs(f.text(numbered.group(2)))})
            continue
        if in_refs:
            flush()
            para.append(line)
            flush()
            continue
        para.append(line.strip())
    flush()
    return {"meta": meta, "blocks": blocks}


def osaudit_toml(ledger: Ledger) -> str:
    base = (PAPER / "osaudit.base.toml").read_text().rstrip()
    return (f'schema = "{SCHEMA}"\n\n# Generated by paper/build.py from paper/osaudit.base.toml and the placeholders of '
            f"paper/paper.md; do not edit by hand.\n\n{base}\n\n{ledger.toml()}")


def main(draft: bool = False, source: str = "paper.md", name: str = "paper") -> None:
    """Build paper/<name>.docx and .pdf from paper/<source>; the paper (name "paper") also writes osaudit.toml."""
    sys.path.insert(0, str(PAPER))
    import tables
    ledger = Ledger()
    f = Formatter(load_values(), ledger, draft)
    spec = parse((PAPER / source).read_text(), f, tables)
    build = PAPER / "build"
    build.mkdir(exist_ok=True)
    (build / f"{name}.spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    docx = PAPER / f"{name}.docx"
    subprocess.run(["node", str(PAPER / "docx_builder.js"), str(build / f"{name}.spec.json"), str(docx)], check=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(PAPER), str(docx)], check=True,
                   capture_output=True)
    # a draft never overwrites the audited configuration; its entries go to paper/build/ for a trial audit
    if name == "paper":
        (build / "osaudit.toml" if draft else ROOT / "osaudit.toml").write_text(osaudit_toml(ledger))
    print(json.dumps({"docx": str(docx), "pdf": str(PAPER / f"{name}.pdf"), "derive": len(ledger.derive),
                      "allow": len(ledger.allow), "pending": f.pending}, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true", help="show missing values as [pending: ...]")
    ap.add_argument("--source", default="paper.md")
    ap.add_argument("--name", default="paper", help="output name: paper/<name>.docx and .pdf")
    a = ap.parse_args()
    main(a.draft, a.source, a.name)
