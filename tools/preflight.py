#!/usr/bin/env python3
"""Validate the MOATTERS runtime before an expensive workflow run."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moatters.config import RSCRIPT, data_path, output_path, runtime_summary

EXPECTED_INPUTS = {
    "go_bp_gmt": data_path("GSEA/c5.go.bp.v2026.1.Hs.symbols.gmt"),
    "hallmark_gmt": data_path("GSEA/h.all.v2026.1.Hs.symbols.gmt"),
    "tcga_brca": data_path("UCSC_XENA/Breast Cancer (BRCA)"),
    "metabric": data_path("External/brca_metabric"),
    "gse96058": data_path("External/GSE96058"),
    "tcga_kirc": data_path("UCSC_XENA/Kidney Clear Cell Carcinoma (KIRC)"),
    "tcga_luad": data_path("UCSC_XENA/Lung Adenocarcinoma (LUAD)"),
}
REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "networkx",
    "lifelines",
    "matplotlib",
]
REQUIRED_R_PACKAGES = ["GSVA", "pathifier"]
VALID_SCOPES = ("structure", "core", "benchmark", "full")


def _find_executable(command: str) -> str | None:
    direct = Path(command).expanduser()
    if direct.exists() and direct.is_file():
        return str(direct.resolve())
    return shutil.which(command)


def _check_output_root() -> dict[str, Any]:
    path = output_path()
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".moatters_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return {"path": str(path), "writable": True, "error": None}
    except Exception as exc:
        return {"path": str(path), "writable": False, "error": str(exc)}


def _check_r_packages(executable: str, timeout: int) -> dict[str, Any]:
    expression = (
        "pkgs <- c(" + ",".join(json.dumps(p) for p in REQUIRED_R_PACKAGES) + ");"
        "status <- vapply(pkgs, requireNamespace, quietly=TRUE, FUN.VALUE=logical(1));"
        "cat(paste(names(status), as.integer(status), sep='=', collapse='\\n'));"
        "quit(status=ifelse(all(status),0,3))"
    )
    try:
        result = subprocess.run(
            [executable, "--vanilla", "-e", expression],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "checked": True,
            "packages": {name: False for name in REQUIRED_R_PACKAGES},
            "error": f"R package check timed out after {timeout} seconds",
        }
    except OSError as exc:
        return {
            "checked": True,
            "packages": {name: False for name in REQUIRED_R_PACKAGES},
            "error": str(exc),
        }

    parsed = {name: False for name in REQUIRED_R_PACKAGES}
    for line in result.stdout.splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name in parsed:
            parsed[name] = value == "1"
    error = result.stderr.strip() or None
    return {"checked": True, "packages": parsed, "error": error}


def _status(ok: bool, required: bool) -> str:
    if ok:
        return "PASS"
    return "FAIL" if required else "NOT_REQUIRED"


def build_report(scope: str, *, check_r_packages: bool, timeout: int) -> tuple[dict[str, Any], bool]:
    require_inputs = scope in {"core", "full"}
    require_python = scope in {"structure", "core", "benchmark", "full"}
    require_r = scope in {"benchmark", "full"}

    inputs = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in EXPECTED_INPUTS.items()
    }
    python_modules = {
        name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES
    }
    r_configured = str(RSCRIPT)
    r_executable = _find_executable(r_configured)
    rscript: dict[str, Any] = {
        "configured": r_configured,
        "found": r_executable is not None,
        "resolved": r_executable,
        "packages": {"checked": False, "packages": {}, "error": None},
    }
    if r_executable and check_r_packages and require_r:
        rscript["packages"] = _check_r_packages(r_executable, timeout)

    output = _check_output_root()
    input_ok = all(item["exists"] for item in inputs.values())
    python_ok = all(python_modules.values())
    r_binary_ok = r_executable is not None
    package_result = rscript["packages"]
    r_packages_ok = (
        all(package_result["packages"].values())
        if package_result["checked"]
        else not check_r_packages
    )

    checks = {
        "inputs": {
            "required": require_inputs,
            "ok": input_ok,
            "status": _status(input_ok, require_inputs),
        },
        "python": {
            "required": require_python,
            "ok": python_ok,
            "status": _status(python_ok, require_python),
        },
        "output_root": {
            "required": True,
            "ok": output["writable"],
            "status": _status(output["writable"], True),
        },
        "rscript": {
            "required": require_r,
            "ok": r_binary_ok,
            "status": _status(r_binary_ok, require_r),
        },
        "r_packages": {
            "required": require_r and check_r_packages,
            "ok": r_packages_ok,
            "status": _status(r_packages_ok, require_r and check_r_packages),
        },
    }
    overall_ok = all(item["ok"] for item in checks.values() if item["required"])
    report = {
        "scope": scope,
        "overall_status": "PASS" if overall_ok else "INCOMPLETE",
        "runtime": runtime_summary(),
        "checks": checks,
        "inputs": inputs,
        "python_modules": python_modules,
        "output_root": output,
        "rscript": rscript,
    }
    return report, overall_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check repository structure/runtime, analysis inputs, and the optional "
            "R benchmark environment. Default scope: full."
        )
    )
    parser.add_argument("--scope", choices=VALID_SCOPES, default="full")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument(
        "--skip-r-package-check",
        action="store_true",
        help="Require Rscript for benchmark/full scope but do not query GSVA/Pathifier",
    )
    parser.add_argument("--r-timeout", type=int, default=30)
    args = parser.parse_args()

    report, ok = build_report(
        args.scope,
        check_r_packages=not args.skip_r_package_check,
        timeout=max(1, args.r_timeout),
    )
    print(json.dumps(report, indent=2))
    if not args.json:
        print(f"PREFLIGHT [{args.scope.upper()}]: {report['overall_status']}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
