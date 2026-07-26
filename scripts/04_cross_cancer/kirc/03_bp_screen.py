# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
TCGA-KIRC Cross-Cancer Stage 2A — GO-BP observation-readiness and
stage-discriminability screening

Input
-----
D:\MOATTERS-Output\MOATTERS_KIRC_STAGE1
D:\MOATTERS-Data\GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt

Output
------
D:\MOATTERS-Output\MOATTERS_KIRC_STAGE2A_BP_SCREEN

Primary contract
----------------
- Primary cohort: TCGA-KIRC primary tumours.
- Primary contrast: Stage I/II (Early) vs Stage III/IV (Late).
- Gene expression is standardized gene-wise across locked KIRC patients.
- BP score = mean standardized expression of matched genes.
- Observation-ready threshold: matched genes >= 10.
- Primary stage comparison: Welch two-sample t-test.
- Discriminability: D = -log10(p).
- Nominal screening threshold: p < 0.05, equivalently D > 1.30103.
- The number of selected BP terms is allowed to vary by cancer.
- No BRCA BP term list or BRCA module architecture is transferred.

Sensitivity outputs
-------------------
Observation-readiness is additionally reported for K = 5, 10, 15 and 20.
Nominal-D sensitivity counts are reported for D >= 1.0, 1.30103, 1.5 and 2.0.

This stage does not construct a BP-correlation network or patient-level modules.
"""

from __future__ import annotations

import gzip
import json
import math
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


STAGE1 = output_path(r"MOATTERS_KIRC_STAGE1")
EXPRESSION = (
    STAGE1 / "matrices" /
    "KIRC_primary_expression_genes_x_samples.tsv.gz"
)
CLINICAL = (
    STAGE1 / "tables" /
    "KIRC_primary_clinical_harmonized.tsv"
)
GMT = data_path(r"GSEA/c5.go.bp.v2026.1.Hs.symbols.gmt")
OUT = output_path(r"MOATTERS_KIRC_STAGE2A_BP_SCREEN")

PRIMARY_K = 10
K_VALUES = [5, 10, 15, 20]
D_THRESHOLDS = [1.0, 1.30103, 1.5, 2.0]
NOMINAL_P = 0.05


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def clean_gene(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def load_gmt(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            description = parts[1].strip()
            genes = sorted({
                clean_gene(g) for g in parts[2:] if clean_gene(g)
            })
            rows.append({
                "gmt_line": line_no,
                "term": term,
                "description": description,
                "genes": genes,
                "n_genes_gmt": len(genes),
            })
    return rows


def safe_D(p: float) -> float:
    if not np.isfinite(p):
        return np.nan
    return float(-math.log10(max(float(p), np.finfo(float).tiny)))


def main():
    for path in [EXPRESSION, CLINICAL, GMT]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "matrices", "logs", "audit"]:
        (OUT / sub).mkdir(exist_ok=True)

    with (OUT / "logs" / "kirc_stage2a.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting TCGA-KIRC Stage 2A BP screening", fh)

        expr = pd.read_csv(
            EXPRESSION,
            sep="\t",
            compression="infer",
            low_memory=False,
        )
        if expr.columns[0] != "gene_symbol":
            expr = expr.rename(columns={expr.columns[0]: "gene_symbol"})

        expr["gene_symbol"] = expr["gene_symbol"].map(clean_gene)
        expr = expr[
            expr["gene_symbol"].notna()
            & expr["gene_symbol"].ne("")
        ].copy()
        expr = expr.drop_duplicates("gene_symbol", keep="first")
        expr = expr.set_index("gene_symbol")

        clinical = pd.read_csv(
            CLINICAL, sep="\t", low_memory=False
        )
        clinical["sample_id"] = clinical["sample_id"].astype(str)
        clinical = clinical.set_index("sample_id")

        common_samples = [
            c for c in expr.columns if c in clinical.index
        ]
        if len(common_samples) < 400:
            raise RuntimeError(
                f"Expression-clinical overlap unexpectedly low: {len(common_samples)}"
            )

        expr = expr[common_samples].apply(
            pd.to_numeric, errors="coerce"
        )
        clinical = clinical.loc[common_samples]

        stage = pd.to_numeric(
            clinical["stage_late_binary"], errors="coerce"
        )
        usable = stage.isin([0, 1])
        stage_samples = stage.index[usable].tolist()
        early_samples = stage.index[stage.eq(0)].tolist()
        late_samples = stage.index[stage.eq(1)].tolist()

        log(
            f"Locked overlap: {len(common_samples)}; "
            f"stage usable={len(stage_samples)}; "
            f"Early={len(early_samples)}; Late={len(late_samples)}",
            fh,
        )

        # Gene-wise standardization across all locked KIRC patients.
        gene_mean = expr.mean(axis=1, skipna=True)
        gene_sd = expr.std(axis=1, ddof=1, skipna=True)
        usable_gene = gene_sd.gt(0) & gene_sd.notna()
        expr = expr.loc[usable_gene]
        gene_mean = gene_mean.loc[usable_gene]
        gene_sd = gene_sd.loc[usable_gene]
        z = expr.sub(gene_mean, axis=0).div(gene_sd, axis=0)

        log(
            f"Expression genes before zero-variance removal: "
            f"{len(usable_gene)}; retained={z.shape[0]}",
            fh,
        )

        gmt_rows = load_gmt(GMT)
        expression_genes = set(z.index)

        result_rows = []
        score_columns = {}
        matched_gene_rows = []

        for i, item in enumerate(gmt_rows, start=1):
            matched = sorted(
                expression_genes.intersection(item["genes"])
            )
            n_match = len(matched)

            row = {
                "term": item["term"],
                "description": item["description"],
                "n_genes_gmt": item["n_genes_gmt"],
                "n_genes_matched": n_match,
                "coverage_fraction": (
                    n_match / item["n_genes_gmt"]
                    if item["n_genes_gmt"] else np.nan
                ),
            }
            for k in K_VALUES:
                row[f"observation_ready_K{k}"] = n_match >= k

            if n_match >= PRIMARY_K:
                score = z.loc[matched].mean(axis=0, skipna=True)
                early = score.loc[early_samples].dropna().to_numpy(float)
                late = score.loc[late_samples].dropna().to_numpy(float)

                stat, p = ttest_ind(
                    late,
                    early,
                    equal_var=False,
                    nan_policy="omit",
                )
                delta = float(np.mean(late) - np.mean(early))
                pooled_sd = float(
                    np.sqrt(
                        (
                            (len(late) - 1) * np.var(late, ddof=1)
                            + (len(early) - 1) * np.var(early, ddof=1)
                        )
                        / (len(late) + len(early) - 2)
                    )
                ) if len(late) > 1 and len(early) > 1 else np.nan

                row.update({
                    "n_early": len(early),
                    "n_late": len(late),
                    "mean_early": float(np.mean(early)),
                    "mean_late": float(np.mean(late)),
                    "late_minus_early": delta,
                    "welch_t": float(stat),
                    "p_value": float(p),
                    "D": safe_D(float(p)),
                    "cohens_d_late_minus_early": (
                        delta / pooled_sd
                        if np.isfinite(pooled_sd) and pooled_sd > 0
                        else np.nan
                    ),
                    "nominal_selected_p_lt_0_05": bool(p < NOMINAL_P),
                })

                score_columns[item["term"]] = score
                matched_gene_rows.append({
                    "term": item["term"],
                    "n_genes_matched": n_match,
                    "matched_genes": " | ".join(matched),
                })
            else:
                row.update({
                    "n_early": np.nan,
                    "n_late": np.nan,
                    "mean_early": np.nan,
                    "mean_late": np.nan,
                    "late_minus_early": np.nan,
                    "welch_t": np.nan,
                    "p_value": np.nan,
                    "D": np.nan,
                    "cohens_d_late_minus_early": np.nan,
                    "nominal_selected_p_lt_0_05": False,
                })

            result_rows.append(row)

            if i % 500 == 0:
                log(f"Processed GO-BP terms: {i}/{len(gmt_rows)}", fh)

        results = pd.DataFrame(result_rows)

        eligible_mask = results[f"observation_ready_K{PRIMARY_K}"]
        results["q_value_BH_within_K10"] = np.nan
        if eligible_mask.any():
            results.loc[
                eligible_mask, "q_value_BH_within_K10"
            ] = multipletests(
                results.loc[eligible_mask, "p_value"].astype(float),
                method="fdr_bh",
            )[1]

        results = results.sort_values(
            ["nominal_selected_p_lt_0_05", "D", "n_genes_matched"],
            ascending=[False, False, False],
        )
        results.to_csv(
            OUT / "tables" / "KIRC_GO_BP_stage_discriminability_all.csv",
            index=False, encoding="utf-8-sig"
        )

        selected = results[
            results["nominal_selected_p_lt_0_05"]
        ].copy()
        selected.to_csv(
            OUT / "tables" / "KIRC_selected_BP_nominal_p_lt_0_05.csv",
            index=False, encoding="utf-8-sig"
        )

        pd.DataFrame(matched_gene_rows).to_csv(
            OUT / "tables" / "KIRC_K10_BP_matched_gene_manifest.csv",
            index=False, encoding="utf-8-sig"
        )

        # Patient-level K10 BP score matrix for the next network stage.
        if score_columns:
            bp_scores = pd.DataFrame(score_columns).T
            bp_scores.index.name = "term"
            bp_scores.to_csv(
                OUT / "matrices" /
                "KIRC_K10_observation_ready_BP_scores_terms_x_patients.csv.gz",
                compression="gzip",
            )

            selected_terms = selected["term"].tolist()
            bp_scores.loc[
                [t for t in selected_terms if t in bp_scores.index]
            ].to_csv(
                OUT / "matrices" /
                "KIRC_selected_BP_scores_terms_x_patients.csv.gz",
                compression="gzip",
            )

        readiness_rows = []
        for k in K_VALUES:
            readiness_rows.append({
                "K_min_matched_genes": k,
                "n_GMT_terms": len(results),
                "n_observation_ready": int(
                    results[f"observation_ready_K{k}"].sum()
                ),
                "observation_ready_fraction": float(
                    results[f"observation_ready_K{k}"].mean()
                ),
            })
        readiness = pd.DataFrame(readiness_rows)
        readiness.to_csv(
            OUT / "tables" / "KIRC_observation_readiness_by_K.csv",
            index=False, encoding="utf-8-sig"
        )

        d_rows = []
        eligible = results.loc[eligible_mask.reindex(results.index, fill_value=False)].copy()
        for d_cut in D_THRESHOLDS:
            d_rows.append({
                "D_threshold": d_cut,
                "equivalent_p_threshold": 10 ** (-d_cut),
                "n_K10_BP_selected": int(
                    pd.to_numeric(eligible["D"], errors="coerce").ge(d_cut).sum()
                ),
            })
        d_sensitivity = pd.DataFrame(d_rows)
        d_sensitivity.to_csv(
            OUT / "tables" / "KIRC_D_threshold_sensitivity_counts.csv",
            index=False, encoding="utf-8-sig"
        )

        status = "PASS"
        if len(selected) < 5:
            status = "HOLD_TOO_FEW_SELECTED_BP"

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "TCGA_KIRC_STAGE2A_BP_SCREEN",
            "status": status,
            "n_locked_samples": len(common_samples),
            "n_stage_usable": len(stage_samples),
            "n_early": len(early_samples),
            "n_late": len(late_samples),
            "n_nonzero_variance_expression_genes": int(z.shape[0]),
            "n_GMT_terms": len(results),
            "primary_K": PRIMARY_K,
            "n_observation_ready_K10": int(eligible_mask.sum()),
            "nominal_p_threshold": NOMINAL_P,
            "D_threshold": -math.log10(NOMINAL_P),
            "n_nominal_selected_BP": int(len(selected)),
            "n_BH_FDR_lt_0_05": int(
                pd.to_numeric(
                    results["q_value_BH_within_K10"], errors="coerce"
                ).lt(0.05).sum()
            ),
            "observation_readiness_by_K": readiness_rows,
            "D_threshold_sensitivity": d_rows,
            "methodological_boundaries": [
                "KIRC received its own GO-BP screening.",
                "No BRCA BP terms or modules were transferred.",
                "The nominal p<0.05 threshold is a reconstruction-oriented screen, not a claim that each BP is independently validated.",
                "Network construction and module reconstruction will be performed only in the next stage.",
            ],
            "next_step": (
                "Stage 2B: construct the KIRC Late-stage BP-correlation network, "
                "audit r/p thresholds, and derive KIRC-specific modules."
            ),
        }
        with (OUT / "KIRC_STAGE2A_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        (OUT / "README_KIRC_STAGE2A.txt").write_text(
            "MOATTERS — TCGA-KIRC Stage 2A\n\n"
            f"Status: {status}\n"
            f"Stage-usable patients: {len(stage_samples)} "
            f"(Early={len(early_samples)}, Late={len(late_samples)})\n"
            f"K10 observation-ready BP terms: {int(eligible_mask.sum())}\n"
            f"Nominally selected BP terms: {len(selected)}\n"
            "No BRCA BP/module definitions were transferred.\n",
            encoding="utf-8",
        )

        log(f"KIRC Stage 2A completed: {status}", fh)
        log(
            f"K10 observation-ready BP={int(eligible_mask.sum())}; "
            f"nominal selected={len(selected)}; "
            f"BH-FDR<0.05={summary['n_BH_FDR_lt_0_05']}",
            fh,
        )


if __name__ == "__main__":
    main()
