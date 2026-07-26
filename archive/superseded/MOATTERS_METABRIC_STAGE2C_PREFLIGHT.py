# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
METABRIC Stage 2C-Preflight — recover the locked reconstruction formula

Purpose
-------
Recover the exact historical implementation that generated:
- BRCA_patient_module_scores_z.csv
- BRCA_early_late_module_centroids.csv
- BRCA_patient_strategy_master_table.csv

The script searches existing MOATTERS Python/R scripts for explicit references to the
locked artifact filenames and for nearby scoring/standardization/centroid code.
It also audits the exact 30-BP module structure and K=5/10/15/20 retention.

No model fitting or METABRIC reconstruction is performed.

Output
------
D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE2C_PREFLIGHT
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOTS = [
    Path(r"D:\MOATTERS-Output"),
    Path(r"D:\MOATTERS-Output"),
    Path(r"D:\MOATTERS-Data"),
    Path(r"D:\MOATTERS"),
]

STAGE2B = Path(
    r"D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE2B_LOCK_COVERAGE_V1_1"
)
OUT = Path(r"D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE2C_PREFLIGHT")

SEARCH_TERMS = [
    "BRCA_patient_module_scores_z.csv",
    "BRCA_early_late_module_centroids.csv",
    "BRCA_patient_strategy_master_table.csv",
    "BRCA_profile_module_composition.csv",
    "module_scores_z",
    "early_late_module_centroids",
    "risk_direction_sign",
    "late_vs_early_similarity_delta",
]

SCRIPT_EXTS = {".py", ".r", ".R", ".ipynb", ".txt", ".md"}
PRUNE = {
    "$recycle.bin", "system volume information", "windows", "program files",
    "program files (x86)", "anaconda3", ".git", "__pycache__", "node_modules",
}


def log(msg, fh):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def safe_read_text(path, max_bytes=20_000_000):
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def extract_context(text, hit_start, radius=1800):
    a = max(0, hit_start - radius)
    b = min(len(text), hit_start + radius)
    return text[a:b]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "snippets").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    required = [
        STAGE2B / "locked_artifacts" / "BRCA_module_assignment.csv",
        STAGE2B / "locked_artifacts" / "BRCA_profile_module_composition.csv",
        STAGE2B / "tables" / "TCGA_BRCA_locked_BP_METABRIC_coverage_long.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 2B inputs: {missing}")

    with (OUT / "logs" / "stage2c_preflight.log").open("w", encoding="utf-8") as fh:
        log("Starting Stage 2C preflight", fh)

        assignment = pd.read_csv(required[0])
        composition = pd.read_csv(required[1])
        coverage = pd.read_csv(required[2])

        selected_cov = coverage[coverage["source_role"] == "selected_bp"].copy()
        k_rows = []
        for k in [5, 10, 15, 20]:
            k_rows.append({
                "K_min_matched_genes": k,
                "n_selected_BP": int(len(selected_cov)),
                "n_eligible": int(selected_cov[f"eligible_K{k}"].sum()),
                "eligible_fraction": float(selected_cov[f"eligible_K{k}"].mean()),
                "excluded_terms": " | ".join(
                    selected_cov.loc[
                        ~selected_cov[f"eligible_K{k}"], "source_bp_value"
                    ].astype(str).tolist()
                ),
            })
        pd.DataFrame(k_rows).to_csv(
            OUT / "tables" / "selected_30_BP_K_sensitivity.csv",
            index=False, encoding="utf-8-sig"
        )
        for row in k_rows:
            log(
                f"Selected 30 BP, K>={row['K_min_matched_genes']}: "
                f"{row['n_eligible']}/{row['n_selected_BP']}",
                fh,
            )

        # Confirm which assignment group exactly matches the 7-module composition.
        comp_map = {}
        for _, row in composition.iterrows():
            terms = {
                x.strip() for x in str(row["BP_terms"]).split("|") if x.strip()
            }
            comp_map[str(row["module_id"])] = terms

        group_rows = []
        for group, gdf in assignment.groupby("group"):
            assign_map = {
                str(mid): set(sub["term"].astype(str))
                for mid, sub in gdf.groupby("module_id")
            }
            all_modules = sorted(set(comp_map) | set(assign_map))
            exact_modules = 0
            overlap_n = 0
            union_n = 0
            for mid in all_modules:
                a = comp_map.get(mid, set())
                b = assign_map.get(mid, set())
                exact_modules += int(a == b)
                overlap_n += len(a & b)
                union_n += len(a | b)
            group_rows.append({
                "assignment_group": group,
                "n_modules": len(assign_map),
                "n_terms": int(gdf["term"].nunique()),
                "exact_module_sets": exact_modules,
                "composition_modules": len(comp_map),
                "term_jaccard_over_modules": overlap_n / union_n if union_n else 0.0,
                "exact_full_match": exact_modules == len(comp_map) == len(assign_map),
            })

        group_match = pd.DataFrame(group_rows).sort_values(
            ["exact_full_match", "term_jaccard_over_modules"],
            ascending=False,
        )
        group_match.to_csv(
            OUT / "tables" / "module_assignment_group_match.csv",
            index=False, encoding="utf-8-sig"
        )
        best_group = str(group_match.iloc[0]["assignment_group"])
        log(
            f"Best module-assignment match: {best_group}; "
            f"exact_full_match={bool(group_match.iloc[0]['exact_full_match'])}",
            fh,
        )

        # Search historical scripts.
        roots = []
        seen = set()
        for root in ROOTS:
            if root.exists():
                key = str(root.resolve()).lower()
                if key not in seen:
                    roots.append(root)
                    seen.add(key)

        hit_rows = []
        n_files = 0
        n_scripts = 0
        snippet_counter = 0

        for root in roots:
            log(f"Searching scripts under {root}", fh)
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d.lower() not in PRUNE]
                for filename in filenames:
                    n_files += 1
                    path = Path(dirpath) / filename
                    if path.suffix not in SCRIPT_EXTS and path.suffix.lower() not in {
                        x.lower() for x in SCRIPT_EXTS
                    }:
                        continue
                    n_scripts += 1
                    text = safe_read_text(path)
                    if not text:
                        continue
                    lower = text.lower()
                    matched_terms = [term for term in SEARCH_TERMS if term.lower() in lower]
                    if not matched_terms:
                        continue

                    first_positions = [
                        lower.find(term.lower()) for term in matched_terms
                        if lower.find(term.lower()) >= 0
                    ]
                    pos = min(first_positions)
                    context = extract_context(text, pos)
                    snippet_counter += 1
                    snippet_name = f"snippet_{snippet_counter:03d}_{path.stem[:80]}.txt"
                    (OUT / "snippets" / snippet_name).write_text(
                        f"SOURCE: {path}\n"
                        f"MATCHED TERMS: {matched_terms}\n\n"
                        f"{context}",
                        encoding="utf-8",
                    )
                    hit_rows.append({
                        "source_path": str(path),
                        "filename": path.name,
                        "extension": path.suffix,
                        "matched_terms": " | ".join(matched_terms),
                        "n_matched_terms": len(matched_terms),
                        "snippet_file": str(OUT / "snippets" / snippet_name),
                    })
                    log(f"SCRIPT HIT: {path}", fh)

        hits = pd.DataFrame(
            hit_rows,
            columns=[
                "source_path", "filename", "extension", "matched_terms",
                "n_matched_terms", "snippet_file",
            ],
        )
        if len(hits):
            hits = hits.sort_values("n_matched_terms", ascending=False)
        hits.to_csv(
            OUT / "tables" / "historical_reconstruction_script_hits.csv",
            index=False, encoding="utf-8-sig"
        )

        status = "PASS" if len(hits) > 0 else "HOLD"
        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "selected_30_bp_k_sensitivity": k_rows,
            "best_assignment_group": best_group,
            "assignment_group_match": group_match.to_dict(orient="records"),
            "files_seen": n_files,
            "scripts_examined": n_scripts,
            "historical_script_hits": int(len(hits)),
            "next_step": (
                "Inspect top script snippets, lock the exact scoring and standardization "
                "formula, then run METABRIC Stage 2C reconstruction."
                if status == "PASS"
                else "Search archived ZIP contents or original project directory manually."
            ),
        }
        with (OUT / "METABRIC_STAGE2C_PREFLIGHT_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        (OUT / "README_STAGE2C_PREFLIGHT.txt").write_text(
            f"Status: {status}\n"
            f"Best module assignment group: {best_group}\n"
            f"Historical script hits: {len(hits)}\n"
            f"Scripts examined: {n_scripts}\n",
            encoding="utf-8",
        )

        log(
            f"Stage 2C preflight completed: {status}; "
            f"historical script hits={len(hits)}",
            fh,
        )


if __name__ == "__main__":
    main()
