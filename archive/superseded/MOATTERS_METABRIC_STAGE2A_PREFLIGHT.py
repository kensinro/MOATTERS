# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
"""
MOATTERS
METABRIC Stage 2A — TCGA-BRCA artifact discovery and transfer preflight

Purpose
-------
Locate and audit the existing TCGA-BRCA-locked biological-process/module/state
artifacts needed for external METABRIC reconstruction. This stage does NOT fit,
retrain, or transfer a model. It only creates a reproducible inventory and
selects candidate artifacts.

Inputs
------
D:\MOATTERS-Output
D:\MOATTERS-Data\GSEA
D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE1

Output
------
D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE2A_PREFLIGHT
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


MOATTERS_TEMP_CANDIDATES = [Path(r"D:\MOATTERS-Output"), Path(r"D:\MOATTERS-Output")]
GSEA_DIR = Path(r"D:\MOATTERS-Data\GSEA")
STAGE1_DIR = Path(r"D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE1")
OUT_DIR = Path(r"D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE2A_PREFLIGHT")

KNOWN_RELEVANT_DIR_NAMES = [
    "MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530",
    "MOATTERS_STAGE_BP_Module_Rewiring_V1_20260531",
    "MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531",
    "MOATTERS_BRCA_PATIENT_STATE_RANDOM_TEST_V1_20260531",
    "MOATTERS_BRCA_STATE_DownstreamValidation",
]

TARGET_PATTERNS = [
    r"BRCA.*DStage_BP_results.*\.csv$",
    r"BRCA.*selected_BP_for_network.*\.csv$",
    r"BRCA.*BP_correlation_network_edges.*\.csv$",
    r"BRCA.*BP_correlation_network_node_metrics.*\.csv$",
    r"BRCA.*module_assignment.*\.csv$",
    r"BRCA.*module_summary.*\.csv$",
    r"BRCA.*profile_module_composition.*\.csv$",
    r"BRCA.*module_late_alignment_direction.*\.csv$",
    r"BRCA.*early_late_module_centroids.*\.csv$",
    r"BRCA.*patient_module_scores.*\.csv$",
    r"BRCA.*strategy_master_table.*\.csv$",
    r".*summary.*\.json$",
]

ESSENTIAL_ROLES = {
    "bp_definition": [
        "DStage_BP_results",
        "selected_BP_for_network",
    ],
    "module_assignment": ["module_assignment"],
    "module_composition": ["profile_module_composition"],
    "state_direction": ["module_late_alignment_direction"],
    "centroids": ["early_late_module_centroids"],
}


def log(msg: str, fh=None) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_table_preview(path: Path, nrows: int = 8) -> dict:
    result = {
        "readable": False,
        "delimiter": None,
        "n_preview_rows": 0,
        "n_columns": 0,
        "columns": [],
        "error": None,
    }
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)
            if df.shape[1] > 1 or sep == ",":
                result.update({
                    "readable": True,
                    "delimiter": "\\t" if sep == "\t" else sep,
                    "n_preview_rows": int(df.shape[0]),
                    "n_columns": int(df.shape[1]),
                    "columns": [str(x) for x in df.columns],
                })
                return result
        except Exception as exc:
            result["error"] = repr(exc)
    return result


def canonical_temp_root() -> Path:
    existing = [p for p in MOATTERS_TEMP_CANDIDATES if p.exists()]
    if not existing:
        raise FileNotFoundError("Neither D:\\MOATTERS-Temp nor D:\\MOATTERS-TEMP exists.")
    # On Windows these usually resolve to the same folder; use the first.
    return existing[0]


def collect_candidate_roots(temp_root: Path) -> list[Path]:
    roots = []
    for name in KNOWN_RELEVANT_DIR_NAMES:
        exact = temp_root / name
        if exact.exists():
            roots.append(exact)
        # Include timestamped/variant folders.
        roots.extend([p for p in temp_root.glob(name + "*") if p.is_dir()])
    # Also include folders whose names strongly suggest BRCA BP/state output.
    for p in temp_root.iterdir():
        if not p.is_dir():
            continue
        s = p.name.upper()
        if "BRCA" in s and any(k in s for k in ["BP", "STRATEGY", "ATTRACTOR", "MODULE"]):
            roots.append(p)
    unique = []
    seen = set()
    for p in roots:
        rp = str(p.resolve()).lower()
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    return unique


def matches_target(name: str) -> bool:
    return any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in TARGET_PATTERNS)


def infer_role(filename: str) -> str:
    lower = filename.lower()
    for role, tokens in ESSENTIAL_ROLES.items():
        if any(token.lower() in lower for token in tokens):
            return role
    if "correlation_network_edges" in lower:
        return "network_edges"
    if "correlation_network_node_metrics" in lower:
        return "network_node_metrics"
    if "patient_module_scores" in lower:
        return "tcga_patient_module_scores"
    if lower.endswith(".json"):
        return "summary_json"
    return "supporting"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    (OUT_DIR / "logs").mkdir(exist_ok=True)

    log_path = OUT_DIR / "logs" / "stage2a_preflight.log"
    with log_path.open("w", encoding="utf-8") as fh:
        log("Starting METABRIC Stage 2A transfer preflight", fh)

        temp_root = canonical_temp_root()
        log(f"MOATTERS temp root: {temp_root}", fh)

        if not STAGE1_DIR.exists():
            raise FileNotFoundError(f"Stage 1 output not found: {STAGE1_DIR}")
        summary_path = STAGE1_DIR / "METABRIC_STAGE1_SUMMARY.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Stage 1 summary not found: {summary_path}")
        with summary_path.open("r", encoding="utf-8") as f:
            stage1 = json.load(f)
        if stage1.get("status") != "PASS":
            raise RuntimeError("Stage 1 status is not PASS.")
        log(f"Stage 1 PASS; locked METABRIC samples: {stage1['locked_primary_matrix']['n_samples']}", fh)

        roots = collect_candidate_roots(temp_root)
        log(f"Relevant candidate directories found: {len(roots)}", fh)
        for root in roots:
            log(f"  candidate root: {root}", fh)

        rows = []
        all_files = []
        for root in roots:
            for path in root.rglob("*"):
                if path.is_file():
                    all_files.append(path)

        selected_files = [p for p in all_files if matches_target(p.name)]
        log(f"Target-like artifacts found: {len(selected_files)}", fh)

        for i, path in enumerate(selected_files, start=1):
            rel = str(path.relative_to(temp_root))
            log(f"Inspecting {i}/{len(selected_files)}: {rel}", fh)
            stat = path.stat()
            preview = safe_table_preview(path) if path.suffix.lower() == ".csv" else {}
            rows.append({
                "role": infer_role(path.name),
                "filename": path.name,
                "full_path": str(path),
                "relative_path": rel,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256_file(path),
                "readable_table": preview.get("readable"),
                "delimiter": preview.get("delimiter"),
                "n_preview_rows": preview.get("n_preview_rows"),
                "n_columns": preview.get("n_columns"),
                "columns": json.dumps(preview.get("columns", []), ensure_ascii=False),
                "read_error": preview.get("error"),
            })

        inventory = pd.DataFrame(rows)
        inventory_path = OUT_DIR / "tables" / "TCGA_BRCA_transfer_artifact_inventory.csv"
        inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

        role_rows = []
        for role in ESSENTIAL_ROLES:
            sub = inventory[inventory["role"] == role] if not inventory.empty else pd.DataFrame()
            role_rows.append({
                "role": role,
                "required": True,
                "n_candidates": int(len(sub)),
                "status": "FOUND" if len(sub) > 0 else "MISSING",
                "candidate_paths": " | ".join(sub["full_path"].astype(str).tolist()) if len(sub) else "",
            })
        role_status = pd.DataFrame(role_rows)
        role_status.to_csv(
            OUT_DIR / "tables" / "TCGA_BRCA_transfer_role_status.csv",
            index=False,
            encoding="utf-8-sig",
        )

        gmt_candidates = sorted(GSEA_DIR.glob("c5.go.bp*.gmt")) if GSEA_DIR.exists() else []
        gmt_rows = []
        for path in gmt_candidates:
            gmt_rows.append({
                "filename": path.name,
                "full_path": str(path),
                "size_bytes": path.stat().st_size,
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256_file(path),
            })
        pd.DataFrame(gmt_rows).to_csv(
            OUT_DIR / "tables" / "GO_BP_GMT_inventory.csv",
            index=False,
            encoding="utf-8-sig",
        )

        missing_roles = role_status.loc[role_status["status"] == "MISSING", "role"].tolist()
        status = "PASS" if not missing_roles and len(gmt_candidates) > 0 else "HOLD"

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "METABRIC_STAGE2A_TCGA_TRANSFER_PREFLIGHT",
            "status": status,
            "stage1_status": stage1.get("status"),
            "metabric_locked_samples": stage1["locked_primary_matrix"]["n_samples"],
            "candidate_roots": [str(p) for p in roots],
            "n_target_artifacts": len(selected_files),
            "essential_role_status": role_status.to_dict(orient="records"),
            "missing_roles": missing_roles,
            "n_go_bp_gmt_candidates": len(gmt_candidates),
            "go_bp_gmt_candidates": [str(p) for p in gmt_candidates],
            "next_step": (
                "Stage 2B: lock one artifact per essential role, parse TCGA BP/module definitions, "
                "quantify METABRIC gene-set coverage, and reconstruct patient-level module states "
                "without refitting."
                if status == "PASS"
                else "Resolve missing TCGA-BRCA artifacts or GO-BP GMT before external transfer."
            ),
        }
        with (OUT_DIR / "METABRIC_STAGE2A_SUMMARY.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        readme = f"""MOATTERS — METABRIC Stage 2A

Status: {status}
Stage 1 locked samples: {stage1['locked_primary_matrix']['n_samples']}
Candidate TCGA-BRCA artifacts: {len(selected_files)}
GO-BP GMT candidates: {len(gmt_candidates)}
Missing essential roles: {', '.join(missing_roles) if missing_roles else 'None'}

This stage performed discovery and audit only. It did not retrain or transfer a model.
"""
        (OUT_DIR / "README_STAGE2A.txt").write_text(readme, encoding="utf-8")

        log(f"Stage 2A completed: {status}", fh)
        if missing_roles:
            log(f"Missing roles: {', '.join(missing_roles)}", fh)
        log(f"GO-BP GMT candidates: {len(gmt_candidates)}", fh)


if __name__ == "__main__":
    main()
