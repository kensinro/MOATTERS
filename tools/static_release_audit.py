#!/usr/bin/env python3
"""Run a deterministic static audit and refresh STATIC_AUDIT_REPORT.tsv."""
from __future__ import annotations

import ast
import csv
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "STATIC_AUDIT_REPORT.tsv"
FORBIDDEN = [
    "Post" + "-D-SR",
    "PostD" + "_SR",
    "AIDO" + "-Post",
    "D:" + "\\AIDO",
    "D:/" + "AIDO",
]


def audited_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in {"archive", ".venv"}:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.as_posix().lower())


def main() -> int:
    rows: list[dict[str, str | int]] = []
    errors: list[str] = []
    for path in audited_files():
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        syntax = "PASS"
        error = ""
        try:
            ast.parse(text)
        except SyntaxError as exc:
            syntax = "FAIL"
            error = str(exc).replace("\t", " ").replace("\n", " ")
            errors.append(f"Syntax error: {rel}: {exc}")
        hits = [pattern for pattern in FORBIDDEN if re.search(re.escape(pattern), text, flags=re.I)]
        if hits:
            errors.append(f"Forbidden internal token: {rel}: {', '.join(hits)}")
        rows.append(
            {
                "file": rel,
                "syntax": syntax,
                "lines": len(text.splitlines()),
                "forbidden_tokens": ";".join(hits),
                "error": error,
            }
        )

    with REPORT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "syntax", "lines", "forbidden_tokens", "error"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Audited {len(rows)} active Python files")
    print(f"Report: {REPORT.relative_to(ROOT)}")
    if errors:
        print("\n".join(errors))
        return 1
    print("Static release audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
