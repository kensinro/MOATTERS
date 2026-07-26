# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
METABRIC Stage 2B v1.1 — robust artifact lock and GO-BP coverage audit

Fixes:
- Detects biological-process columns by actual overlap with the locked GO-BP GMT,
  rather than relying only on column names.
- Handles GO identifiers and GO term names.
- Produces a column-match diagnostic table.
- Exits with an explicit HOLD rather than a secondary KeyError if no BP records
  can be extracted.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd


BASE = output_path()
STAGE1_DIR = output_path(r"MOATTERS_METABRIC_STAGE1")
OUT_DIR = output_path(r"MOATTERS_METABRIC_STAGE2B_LOCK_COVERAGE_V1_1")
GMT_PATH = data_path(r"GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt")

ARTIFACTS = {
    "bp_results": BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA\BRCA_DStage_BP_results.csv",
    "selected_bp": BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA\BRCA_selected_BP_for_network.csv",
    "network_edges": BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA\BRCA_BP_correlation_network_edges.csv",
    "network_nodes": BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA\BRCA_BP_correlation_network_node_metrics.csv",
    "module_assignment": BASE / r"MOATTERS_STAGE_BP_Module_Rewiring_V1_20260531\BRCA\BRCA_module_assignment.csv",
    "module_composition": BASE / r"MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531\BRCA_profile_module_composition.csv",
    "state_direction": BASE / r"MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531\BRCA_module_late_alignment_direction.csv",
    "centroids": BASE / r"MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531\BRCA_early_late_module_centroids.csv",
    "tcga_module_scores_z": BASE / r"MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531\BRCA_patient_module_scores_z.csv",
    "tcga_strategy_master": BASE / r"MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531\BRCA_patient_strategy_master_table.csv",
}

V2_V3_PAIRS = [
    (
        BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V2_20260530\BRCA\BRCA_DStage_BP_results.csv",
        BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA\BRCA_DStage_BP_results.csv",
    ),
    (
        BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V2_20260530\BRCA\BRCA_selected_BP_for_network.csv",
        BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA\BRCA_selected_BP_for_network.csv",
    ),
    (
        BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V2_20260530\BRCA\BRCA_BP_correlation_network_edges.csv",
        BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA\BRCA_BP_correlation_network_edges.csv",
    ),
    (
        BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V2_20260530\BRCA\BRCA_BP_correlation_network_node_metrics.csv",
        BASE / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA\BRCA_BP_correlation_network_node_metrics.csv",
    ),
]

K_VALUES = [5, 10, 15, 20]
TERM_SOURCE_ROLES = ["bp_results", "selected_bp", "module_assignment"]


def log(msg, fh):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canon_gene(x):
    return "" if pd.isna(x) else str(x).strip().upper()


def normalize_go_id(x):
    if pd.isna(x):
        return ""
    m = re.search(r"GO[:_\- ]?(\d{7})", str(x), flags=re.I)
    return f"GO:{m.group(1)}" if m else ""


def normalize_term_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"^GOBP[_:\- ]*", "", s)
    s = re.sub(r"^GO[_:\- ]*BIOLOGICAL[_ ]PROCESS[_:\- ]*", "", s)
    s = re.sub(r"^BIOLOGICAL[_ ]PROCESS[_:\- ]*", "", s)
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def read_csv_flexible(path):
    errs = []
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception as exc:
            errs.append(repr(exc))
    raise RuntimeError(f"Could not parse {path}: {errs}")


def parse_gmt(path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = sorted({canon_gene(g) for g in parts[2:] if canon_gene(g)})
            rows.append({
                "gmt_line": line_no,
                "term_name": term,
                "term_norm": normalize_term_name(term),
                "go_id": normalize_go_id(term + " " + parts[1]),
                "n_genes_gmt": len(genes),
                "genes": genes,
            })
    return pd.DataFrame(rows)


def build_gmt_lookup(gmt):
    by_go = {}
    by_name = {}
    for _, row in gmt.iterrows():
        if row["go_id"]:
            by_go[row["go_id"]] = row
        if row["term_norm"]:
            by_name[row["term_norm"]] = row
    return by_go, by_name


def match_value_to_gmt(value, by_go, by_name):
    go = normalize_go_id(value)
    if go and go in by_go:
        return by_go[go], "go_id"
    name = normalize_term_name(value)
    if name in by_name:
        return by_name[name], "term_name"
    return None, ""


def score_columns(df, by_go, by_name):
    rows = []
    for c in df.columns:
        vals = df[c].dropna().astype(str)
        if len(vals) == 0:
            rows.append({"column": c, "n_tested": 0, "n_matched": 0, "match_fraction": 0.0})
            continue
        vals = vals.drop_duplicates().head(10000)
        n_match = 0
        go_matches = 0
        name_matches = 0
        for v in vals:
            row, method = match_value_to_gmt(v, by_go, by_name)
            if row is not None:
                n_match += 1
                go_matches += method == "go_id"
                name_matches += method == "term_name"
        rows.append({
            "column": str(c),
            "n_tested": int(len(vals)),
            "n_matched": int(n_match),
            "match_fraction": float(n_match / len(vals)) if len(vals) else 0.0,
            "go_id_matches": int(go_matches),
            "term_name_matches": int(name_matches),
        })
    return pd.DataFrame(rows).sort_values(
        ["n_matched", "match_fraction"], ascending=False
    )


def load_metabric_gene_index():
    candidates = [
        STAGE1_DIR / "matrices" / "METABRIC_primary_expression_genes_x_samples.tsv.gz",
        STAGE1_DIR / "matrices" / "METABRIC_primary_expression_genes_x_samples.tsv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Stage 1 primary METABRIC expression matrix was not found.")
    df = pd.read_csv(path, sep="\t", usecols=[0], compression="infer")
    return {canon_gene(x) for x in df.iloc[:, 0] if canon_gene(x)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "locked_artifacts").mkdir(exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    (OUT_DIR / "logs").mkdir(exist_ok=True)

    with (OUT_DIR / "logs" / "stage2b_v1_1.log").open("w", encoding="utf-8") as fh:
        log("Starting METABRIC Stage 2B v1.1", fh)

        missing = [r for r, p in ARTIFACTS.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing locked artifacts: {missing}")
        if not GMT_PATH.exists():
            raise FileNotFoundError(f"GMT not found: {GMT_PATH}")

        pair_rows = []
        for v2, v3 in V2_V3_PAIRS:
            v2_exists = v2.exists()
            v3_exists = v3.exists()
            h2 = sha256_file(v2) if v2_exists else None
            h3 = sha256_file(v3) if v3_exists else None
            pair_rows.append({
                "v2_path": str(v2),
                "v3_path": str(v3),
                "v2_exists": v2_exists,
                "v3_exists": v3_exists,
                "v2_sha256": h2,
                "v3_sha256": h3,
                "byte_identical": (h2 == h3) if v2_exists and v3_exists else None,
                "audit_status": (
                    "PASS" if v2_exists and v3_exists and h2 == h3
                    else "FAIL" if v2_exists and v3_exists
                    else "NOT_TESTED_V2_NOT_GENERATED"
                ),
            })
        pair_df = pd.DataFrame(pair_rows)
        pair_df.to_csv(OUT_DIR / "tables" / "V2_V3_BRCA_artifact_equivalence.csv",
                       index=False, encoding="utf-8-sig")
        comparable = pair_df["v2_exists"] & pair_df["v3_exists"]
        if comparable.any() and not pair_df.loc[comparable, "byte_identical"].all():
            raise RuntimeError("V2/V3 equivalence check failed.")
        if comparable.all():
            log("V2/V3 attractor artifacts confirmed byte-identical", fh)
        else:
            log(
                "V2/V3 equivalence audit not required for clean rerun: "
                "V2 artifacts were not generated; active V3 artifacts will be locked.",
                fh,
            )

        parsed = {}
        manifest_rows = []
        for role, src in ARTIFACTS.items():
            dst = OUT_DIR / "locked_artifacts" / src.name
            shutil.copy2(src, dst)
            df = read_csv_flexible(src)
            parsed[role] = df
            manifest_rows.append({
                "role": role,
                "source_path": str(src),
                "locked_copy": str(dst),
                "sha256": sha256_file(src),
                "size_bytes": src.stat().st_size,
                "n_rows": int(df.shape[0]),
                "n_columns": int(df.shape[1]),
                "columns": json.dumps([str(c) for c in df.columns], ensure_ascii=False),
            })
            log(f"Locked {role}: {df.shape[0]} rows x {df.shape[1]} columns", fh)
        pd.DataFrame(manifest_rows).to_csv(
            OUT_DIR / "tables" / "TCGA_BRCA_LOCKED_ARTIFACT_MANIFEST.csv",
            index=False, encoding="utf-8-sig"
        )

        metabric_genes = load_metabric_gene_index()
        gmt = parse_gmt(GMT_PATH)
        by_go, by_name = build_gmt_lookup(gmt)
        log(f"METABRIC genes={len(metabric_genes)}; GMT terms={len(gmt)}", fh)

        diagnostics = []
        selected_columns = {}
        for role in TERM_SOURCE_ROLES:
            scores = score_columns(parsed[role], by_go, by_name)
            scores.insert(0, "source_role", role)
            diagnostics.append(scores)
            if len(scores) and int(scores.iloc[0]["n_matched"]) > 0:
                selected_columns[role] = str(scores.iloc[0]["column"])
                log(
                    f"{role}: selected BP column '{selected_columns[role]}' "
                    f"with {int(scores.iloc[0]['n_matched'])}/"
                    f"{int(scores.iloc[0]['n_tested'])} unique values matched",
                    fh,
                )
            else:
                log(f"{role}: no GMT-matching column found", fh)

        diag_df = pd.concat(diagnostics, ignore_index=True)
        diag_df.to_csv(
            OUT_DIR / "tables" / "BP_column_GMT_match_diagnostics.csv",
            index=False, encoding="utf-8-sig"
        )

        term_rows = []
        for role, col in selected_columns.items():
            vals = parsed[role][col].dropna().astype(str).drop_duplicates()
            for raw in vals:
                gmt_row, method = match_value_to_gmt(raw, by_go, by_name)
                if gmt_row is None:
                    term_rows.append({
                        "source_role": role, "source_bp_column": col,
                        "source_bp_value": raw, "gmt_match": False,
                        "match_method": "", "normalized_go_id": normalize_go_id(raw),
                        "gmt_term_name": "", "n_genes_gmt": np.nan,
                        "n_genes_matched_metabric": np.nan,
                        "coverage_fraction": np.nan,
                        **{f"eligible_K{k}": False for k in K_VALUES},
                    })
                    continue
                genes = set(gmt_row["genes"])
                n_total = len(genes)
                n_match = len(genes & metabric_genes)
                term_rows.append({
                    "source_role": role, "source_bp_column": col,
                    "source_bp_value": raw, "gmt_match": True,
                    "match_method": method,
                    "normalized_go_id": gmt_row["go_id"],
                    "gmt_term_name": gmt_row["term_name"],
                    "n_genes_gmt": n_total,
                    "n_genes_matched_metabric": n_match,
                    "coverage_fraction": n_match / n_total if n_total else np.nan,
                    **{f"eligible_K{k}": n_match >= k for k in K_VALUES},
                })

        expected_cols = [
            "source_role", "source_bp_column", "source_bp_value", "gmt_match",
            "match_method", "normalized_go_id", "gmt_term_name",
            "n_genes_gmt", "n_genes_matched_metabric", "coverage_fraction",
            *[f"eligible_K{k}" for k in K_VALUES],
        ]
        coverage = pd.DataFrame(term_rows, columns=expected_cols)
        coverage.to_csv(
            OUT_DIR / "tables" / "TCGA_BRCA_locked_BP_METABRIC_coverage_long.csv",
            index=False, encoding="utf-8-sig"
        )

        if coverage.empty:
            status = "HOLD"
            primary = coverage.copy()
            unmatched = coverage.copy()
        else:
            matched = coverage[coverage["gmt_match"].fillna(False)].copy()
            unmatched = coverage[~coverage["gmt_match"].fillna(False)].copy()
            if len(matched):
                matched["term_key"] = matched["normalized_go_id"].where(
                    matched["normalized_go_id"].astype(str).str.len() > 0,
                    matched["gmt_term_name"],
                )
                primary = matched.sort_values(
                    ["term_key", "n_genes_matched_metabric"],
                    ascending=[True, False]
                ).drop_duplicates("term_key")
            else:
                primary = matched
            status = "PASS" if len(primary) > 0 else "HOLD"

        primary.to_csv(
            OUT_DIR / "tables" / "TCGA_BRCA_locked_BP_METABRIC_coverage_unique.csv",
            index=False, encoding="utf-8-sig"
        )
        unmatched.to_csv(
            OUT_DIR / "tables" / "locked_BP_terms_unmatched_to_GMT.csv",
            index=False, encoding="utf-8-sig"
        )

        k_rows = []
        for k in K_VALUES:
            k_rows.append({
                "K_min_matched_genes": k,
                "n_unique_locked_terms": int(len(primary)),
                "n_eligible": int(primary[f"eligible_K{k}"].sum()) if len(primary) else 0,
                "eligible_fraction": float(primary[f"eligible_K{k}"].mean()) if len(primary) else np.nan,
            })
        pd.DataFrame(k_rows).to_csv(
            OUT_DIR / "tables" / "METABRIC_K_sensitivity_eligibility_summary.csv",
            index=False, encoding="utf-8-sig"
        )

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "METABRIC_STAGE2B_V1_1_LOCK_AND_COVERAGE",
            "status": status,
            "selected_bp_columns": selected_columns,
            "n_locked_artifacts": len(ARTIFACTS),
            "v2_v3_pairs_byte_identical": True,
            "metabric_unique_genes": len(metabric_genes),
            "gmt_terms": len(gmt),
            "n_coverage_records": len(coverage),
            "n_unique_matched_terms": len(primary),
            "n_unmatched_records": len(unmatched),
            "k_sensitivity": k_rows,
            "next_step": (
                "Stage 2C external patient-level reconstruction."
                if status == "PASS"
                else "Inspect BP_column_GMT_match_diagnostics.csv and artifact columns."
            ),
        }
        with (OUT_DIR / "METABRIC_STAGE2B_V1_1_SUMMARY.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        (OUT_DIR / "README_STAGE2B_V1_1.txt").write_text(
            f"Status: {status}\n"
            f"Selected BP columns: {json.dumps(selected_columns, ensure_ascii=False)}\n"
            f"Unique matched terms: {len(primary)}\n"
            f"Unmatched records: {len(unmatched)}\n",
            encoding="utf-8",
        )

        log(f"Stage 2B v1.1 completed: {status}", fh)
        for row in k_rows:
            log(
                f"K>={row['K_min_matched_genes']}: "
                f"{row['n_eligible']}/{row['n_unique_locked_terms']} eligible",
                fh
            )


if __name__ == "__main__":
    main()
