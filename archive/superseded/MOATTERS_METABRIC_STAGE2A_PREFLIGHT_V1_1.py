# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
"""
MOATTERS
METABRIC Stage 2A v1.1 — broad TCGA-BRCA artifact locator

This revision broadens discovery beyond a fixed folder-name list.
It scans D:\MOATTERS-Output, D:\MOATTERS-Output, D:\MOATTERS-Data and optionally D:\MOATTERS
for filenames matching the required transfer artifacts.

It does NOT read large matrices, fit models, or alter source files.
"""

from __future__ import annotations
import os, re, json, hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT_CANDIDATES = [
    Path(r"D:\MOATTERS-Output"),
    Path(r"D:\MOATTERS-Output"),
    Path(r"D:\MOATTERS-Data"),
    Path(r"D:\MOATTERS"),
]

OUT_DIR = Path(r"D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE2A_PREFLIGHT_V1_1")

TOKENS = {
    "bp_definition": ["dstage_bp_results", "selected_bp_for_network"],
    "module_assignment": ["module_assignment"],
    "module_composition": ["profile_module_composition", "module_composition"],
    "state_direction": ["module_late_alignment_direction", "late_alignment_direction"],
    "centroids": ["early_late_module_centroids", "module_centroids", "centroid"],
    "network_edges": ["bp_correlation_network_edges"],
    "network_nodes": ["bp_correlation_network_node_metrics"],
    "patient_module_scores": ["patient_module_scores"],
    "strategy_master": ["strategy_master_table"],
}

ALLOWED_EXTS = {".csv", ".tsv", ".txt", ".json", ".parquet", ".pkl", ".pickle", ".xlsx"}

PRUNE_NAMES = {
    "$recycle.bin", "system volume information", "windows", "program files",
    "program files (x86)", "anaconda3", ".git", "__pycache__", "node_modules"
}

def log(msg, fh):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n"); fh.flush()

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def infer_role(name):
    s = name.lower()
    for role, toks in TOKENS.items():
        if any(t in s for t in toks):
            return role
    return "other"

def filename_is_candidate(name):
    s = name.lower()
    return any(t in s for toks in TOKENS.values() for t in toks)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR/"tables").mkdir(exist_ok=True)
    (OUT_DIR/"logs").mkdir(exist_ok=True)

    roots = []
    seen = set()
    for p in ROOT_CANDIDATES:
        if p.exists():
            rp = str(p.resolve()).lower()
            if rp not in seen:
                roots.append(p)
                seen.add(rp)

    with open(OUT_DIR/"logs"/"stage2a_v1_1.log", "w", encoding="utf-8") as fh:
        log(f"Roots to scan: {len(roots)}", fh)
        for r in roots: log(f"  {r}", fh)

        rows = []
        n_dirs = 0
        n_files = 0

        for root in roots:
            log(f"Scanning root: {root}", fh)
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d.lower() not in PRUNE_NAMES]
                n_dirs += 1
                if n_dirs % 1000 == 0:
                    log(f"Progress: {n_dirs} directories, {n_files} files", fh)

                for fn in filenames:
                    n_files += 1
                    if not filename_is_candidate(fn):
                        continue
                    path = Path(dirpath) / fn
                    try:
                        st = path.stat()
                        rows.append({
                            "role": infer_role(fn),
                            "filename": fn,
                            "full_path": str(path),
                            "root": str(root),
                            "extension": path.suffix.lower(),
                            "size_bytes": st.st_size,
                            "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                            "sha256": sha256_file(path) if st.st_size <= 500_000_000 else "SKIPPED_GT_500MB",
                        })
                        log(f"FOUND [{infer_role(fn)}]: {path}", fh)
                    except Exception as exc:
                        rows.append({
                            "role": infer_role(fn),
                            "filename": fn,
                            "full_path": str(path),
                            "root": str(root),
                            "extension": path.suffix.lower(),
                            "size_bytes": None,
                            "modified_time": None,
                            "sha256": None,
                            "error": repr(exc),
                        })

        inv = pd.DataFrame(rows, columns=[
            "role","filename","full_path","root","extension",
            "size_bytes","modified_time","sha256","error"
        ])
        inv.to_csv(OUT_DIR/"tables"/"TCGA_BRCA_transfer_artifact_inventory_broad.csv",
                   index=False, encoding="utf-8-sig")

        essential = ["bp_definition","module_assignment","module_composition","state_direction","centroids"]
        status_rows = []
        for role in essential:
            sub = inv[inv["role"] == role] if not inv.empty else pd.DataFrame()
            status_rows.append({
                "role": role,
                "n_candidates": int(len(sub)),
                "status": "FOUND" if len(sub) else "MISSING",
                "candidate_paths": " | ".join(sub["full_path"].astype(str).tolist()) if len(sub) else "",
            })
        status_df = pd.DataFrame(status_rows)
        status_df.to_csv(OUT_DIR/"tables"/"TCGA_BRCA_transfer_role_status_broad.csv",
                         index=False, encoding="utf-8-sig")

        missing = status_df.loc[status_df["status"]=="MISSING","role"].tolist()
        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "PASS" if not missing else "HOLD",
            "roots_scanned": [str(x) for x in roots],
            "directories_scanned": n_dirs,
            "files_seen": n_files,
            "candidates_found": int(len(inv)),
            "missing_roles": missing,
            "role_status": status_rows,
            "note": "Broad filename discovery only; no model fitting or transfer was performed."
        }
        with open(OUT_DIR/"METABRIC_STAGE2A_V1_1_SUMMARY.json","w",encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        (OUT_DIR/"README_STAGE2A_V1_1.txt").write_text(
            f"Status: {summary['status']}\n"
            f"Directories scanned: {n_dirs}\n"
            f"Files seen: {n_files}\n"
            f"Candidates found: {len(inv)}\n"
            f"Missing roles: {', '.join(missing) if missing else 'None'}\n",
            encoding="utf-8"
        )
        log(f"Completed: {summary['status']}; candidates={len(inv)}; missing={missing}", fh)

if __name__ == "__main__":
    main()
