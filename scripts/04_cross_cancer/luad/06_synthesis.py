# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
TCGA-LUAD Cross-Cancer Stage 2D — reviewer-facing synthesis and
stage-adjusted survival audit

Inputs
------
D:\MOATTERS-Output\MOATTERS_LUAD_STAGE2A_BP_SCREEN
D:\MOATTERS-Output\MOATTERS_LUAD_STAGE2B_NETWORK_MODULES
D:\MOATTERS-Output\MOATTERS_LUAD_STAGE2C_RECONSTRUCTION

Output
------
D:\MOATTERS-Output\MOATTERS_LUAD_STAGE2D_SYNTHESIS

Purpose
-------
1. Integrate K, D and |r| sensitivity results.
2. Summarize LUAD-specific reconstruction performance.
3. Separate apparent, centroid-only held-out and null evidence roles.
4. Re-audit survival after adjustment for observed pathologic stage.
5. Produce manuscript- and rebuttal-ready compact tables and bounded text.

LUAD is a cross-cancer pipeline applicability demonstration, not an
independent external validation cohort.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
except ImportError as exc:
    raise ImportError(
        "lifelines is required for Stage 2D. Install with: pip install lifelines"
    ) from exc


S2A = output_path(r"MOATTERS_LUAD_STAGE2A_BP_SCREEN")
S2B = output_path(r"MOATTERS_LUAD_STAGE2B_NETWORK_MODULES")
S2C = output_path(r"MOATTERS_LUAD_STAGE2C_RECONSTRUCTION")
OUT = output_path(r"MOATTERS_LUAD_STAGE2D_SYNTHESIS")

READINESS = S2A / "tables" / "LUAD_observation_readiness_by_K.csv"
D_SENS = S2A / "tables" / "LUAD_D_threshold_sensitivity_counts.csv"
NETWORK = S2B / "tables" / "LUAD_network_threshold_sensitivity_summary.csv"
ALIGN = S2C / "tables" / "LUAD_stage_alignment_results.csv"
SURVIVAL = S2C / "tables" / "LUAD_survival_context_results.csv"
PATIENT = S2C / "tables" / "LUAD_patient_reconstruction_master.tsv"
SUMMARY_2A = S2A / "LUAD_STAGE2A_SUMMARY.json"
SUMMARY_2B = S2B / "LUAD_STAGE2B_SUMMARY.json"
SUMMARY_2C = S2C / "LUAD_STAGE2C_SUMMARY.json"


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def fit_cox_models(df: pd.DataFrame, score_col: str) -> list[dict]:
    rows = []
    base_cols = [
        "OS_time_days", "OS_event", score_col,
        "stage_late_binary", "age_at_diagnosis", "pack_years_smoked",
    ]
    dat0 = df[[c for c in base_cols if c in df.columns]].copy()
    for c in dat0.columns:
        dat0[c] = pd.to_numeric(dat0[c], errors="coerce")

    model_specs = [
        ("unadjusted", [score_col]),
        ("stage_adjusted", [score_col, "stage_late_binary"]),
        ("stage_age_adjusted", [
            score_col, "stage_late_binary", "age_at_diagnosis"
        ]),
        ("stage_age_smoking_adjusted", [
            score_col, "stage_late_binary",
            "age_at_diagnosis", "pack_years_smoked"
        ]),
    ]

    for model_name, covariates in model_specs:
        required = ["OS_time_days", "OS_event"] + covariates
        if any(c not in dat0.columns for c in required):
            continue

        dat = dat0[required].dropna().copy()
        dat = dat[
            dat["OS_time_days"].gt(0)
            & dat["OS_event"].isin([0, 1])
        ]
        if len(dat) < 100 or dat["OS_event"].sum() < 20:
            continue

        standardize = [score_col]
        for c in ["age_at_diagnosis", "pack_years_smoked"]:
            if c in covariates:
                standardize.append(c)

        for c in standardize:
            sd = dat[c].std(ddof=1)
            if np.isfinite(sd) and sd > 0:
                dat[c] = (dat[c] - dat[c].mean()) / sd

        cph = CoxPHFitter()
        try:
            cph.fit(
                dat,
                duration_col="OS_time_days",
                event_col="OS_event",
            )
        except Exception as exc:
            rows.append({
                "score": score_col,
                "model": model_name,
                "status": f"FIT_FAILED: {repr(exc)}",
            })
            continue

        coef = float(cph.params_[score_col])
        ci = cph.confidence_intervals_.loc[score_col]

        rows.append({
            "score": score_col,
            "model": model_name,
            "status": "PASS",
            "n": int(len(dat)),
            "events": int(dat["OS_event"].sum()),
            "HR_per_SD_score": float(np.exp(coef)),
            "CI95_low": float(np.exp(ci.iloc[0])),
            "CI95_high": float(np.exp(ci.iloc[1])),
            "p_value": float(cph.summary.loc[score_col, "p"]),
            "model_concordance_index": float(cph.concordance_index_),
            "covariates": " + ".join(covariates),
        })
    return rows


def main():
    inputs = [
        READINESS, D_SENS, NETWORK, ALIGN, SURVIVAL, PATIENT,
        SUMMARY_2A, SUMMARY_2B, SUMMARY_2C,
    ]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "logs", "text", "audit"]:
        (OUT / sub).mkdir(exist_ok=True)

    with (OUT / "logs" / "luad_stage2d.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting TCGA-LUAD Stage 2D synthesis", fh)

        readiness = pd.read_csv(READINESS)
        d_sens = pd.read_csv(D_SENS)
        network = pd.read_csv(NETWORK)
        align = pd.read_csv(ALIGN)
        patient = pd.read_csv(PATIENT, sep="\t", low_memory=False)

        s2a = json.loads(SUMMARY_2A.read_text(encoding="utf-8"))
        s2b = json.loads(SUMMARY_2B.read_text(encoding="utf-8"))
        s2c = json.loads(SUMMARY_2C.read_text(encoding="utf-8"))

        sensitivity_rows = []

        for r in readiness.itertuples(index=False):
            sensitivity_rows.append({
                "parameter_family": "minimum_matched_genes",
                "parameter": f"K>={int(r.K_min_matched_genes)}",
                "primary_setting": int(r.K_min_matched_genes) == 10,
                "result_1_name": "observation_ready_BP",
                "result_1_value": int(r.n_observation_ready),
                "result_2_name": "fraction",
                "result_2_value": float(r.observation_ready_fraction),
            })

        for r in d_sens.itertuples(index=False):
            sensitivity_rows.append({
                "parameter_family": "discriminability_threshold",
                "parameter": f"D>={r.D_threshold:g}",
                "primary_setting": np.isclose(r.D_threshold, 1.30103),
                "result_1_name": "selected_K10_BP",
                "result_1_value": int(r.n_K10_BP_selected),
                "result_2_name": "equivalent_p",
                "result_2_value": float(r.equivalent_p_threshold),
            })

        for r in network.itertuples(index=False):
            sensitivity_rows.append({
                "parameter_family": "network_threshold",
                "parameter": f"|r|>={r.abs_r_threshold:.2f}",
                "primary_setting": np.isclose(r.abs_r_threshold, 0.35),
                "result_1_name": "edges",
                "result_1_value": int(r.n_edges),
                "result_2_name": "modules",
                "result_2_value": int(r.n_modules),
                "network_density": float(r.network_density),
                "modularity": float(r.modularity),
                "module_size_range": (
                    f"{int(r.smallest_module_size)}-"
                    f"{int(r.largest_module_size)}"
                ),
            })

        sensitivity = pd.DataFrame(sensitivity_rows)
        sensitivity.to_csv(
            OUT / "tables" / "Table_LUAD_parameter_sensitivity.csv",
            index=False,
            encoding="utf-8-sig",
        )

        recon = align.copy()
        recon["cohort"] = "TCGA-LUAD"
        recon["analysis_role"] = np.where(
            recon["evidence_role"].eq("centroid_only_heldout_alignment"),
            "internal_centroid_only_heldout_audit",
            "internal_apparent_reference_association",
        )
        recon["cross_cancer_interpretation"] = (
            "LUAD-specific pipeline reconstruction; not external validation"
        )
        recon.to_csv(
            OUT / "tables" / "Table_LUAD_reconstruction_performance.csv",
            index=False,
            encoding="utf-8-sig",
        )

        cox_rows = []
        for score in [
            "late_minus_early_margin",
            "adverse_direction_burden",
            "cv_late_minus_early_margin",
        ]:
            if score in patient.columns:
                cox_rows.extend(fit_cox_models(patient, score))

        adjusted_survival = pd.DataFrame(cox_rows)
        adjusted_survival.to_csv(
            OUT / "tables" / "Table_LUAD_survival_adjusted_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )

        primary_network = network[
            np.isclose(network["abs_r_threshold"], 0.35)
        ].iloc[0]
        cv_row = align[
            align["score"].eq("cv_late_minus_early_margin")
        ].iloc[0]
        apparent_row = align[
            align["score"].eq("late_minus_early_margin")
        ].iloc[0]

        reviewer_row = pd.DataFrame([{
            "cancer": "TCGA-LUAD",
            "role": "cross-cancer_pipeline_applicability",
            "n_locked_primary_tumours": s2c["n_locked_patients"],
            "n_stage_usable": s2c["n_stage_usable"],
            "early_n": s2c["n_early"],
            "late_n": s2c["n_late"],
            "nonzero_variance_genes": s2a[
                "n_nonzero_variance_expression_genes"
            ],
            "GO_BP_terms_screened": s2a["n_GMT_terms"],
            "observation_ready_K10": s2a["n_observation_ready_K10"],
            "nominal_selected_D_gt_1_301": s2a["n_nominal_selected_BP"],
            "locked_task_discriminative_BP": s2b["top_N_BP"],
            "primary_network_abs_r": s2b["primary_abs_r_threshold"],
            "primary_network_edges": int(primary_network["n_edges"]),
            "primary_network_modules": int(primary_network["n_modules"]),
            "primary_module_sizes": json.dumps(
                s2b["primary_module_sizes"], ensure_ascii=False
            ),
            "apparent_stage_AUC": apparent_row[
                "auc_orientation_invariant"
            ],
            "centroid_only_heldout_AUC": cv_row[
                "auc_orientation_invariant"
            ],
            "centroid_permutation_p": s2c[
                "permutation_null"
            ]["empirical_p_value"],
            "bounded_claim": (
                "The same reconstruction workflow was executable in LUAD and "
                "produced a non-null LUAD-specific patient representation. "
                "This is an internal cross-cancer applicability demonstration, "
                "not independent external validation or a clinical model."
            ),
        }])

        reviewer_row.to_csv(
            OUT / "tables" / "Table_LUAD_reviewer_facing_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        stage_adjusted = adjusted_survival[
            adjusted_survival["model"].eq("stage_adjusted")
            & adjusted_survival["status"].eq("PASS")
        ].copy()

        survival_text = (
            "Stage-adjusted survival associations were re-audited because the "
            "representation was explicitly derived from pathologic stage."
        )
        if len(stage_adjusted):
            vals = []
            for r in stage_adjusted.itertuples(index=False):
                vals.append(
                    f"{r.score}: HR/SD={r.HR_per_SD_score:.3f}, "
                    f"95% CI {r.CI95_low:.3f}-{r.CI95_high:.3f}, "
                    f"p={r.p_value:.3g}"
                )
            survival_text += " " + "; ".join(vals) + "."

        rebuttal = f"""Reviewer-facing LUAD cross-cancer response

To further evaluate cross-cancer applicability, we repeated the complete
observation-readiness, task-discriminability, network reconstruction and
patient-state workflow de novo in TCGA-LUAD rather than transferring the
BRCA- or KIRC-specific BP terms or module architectures. Among
{s2c['n_stage_usable']} stage-eligible primary tumours
({s2c['n_early']} Stage I/II and {s2c['n_late']} Stage III/IV),
{s2a['n_observation_ready_K10']} GO biological processes were
observation-ready at K>=10. The 30 most task-discriminative LUAD observables
were locked and yielded {int(primary_network['n_modules'])} LUAD-specific
modules at the prespecified |r|>=0.35 network threshold. The reconstructed
Late-minus-Early centroid margin achieved an apparent AUC of
{apparent_row['auc_orientation_invariant']:.3f}; centroid-only repeated
held-out evaluation yielded AUC
{cv_row['auc_orientation_invariant']:.3f}. The observed mean held-out AUC
exceeded a 500-label-permutation null
(empirical p={s2c['permutation_null']['empirical_p_value']:.4g}).

Sensitivity analyses showed gradual changes in observation-readiness across
K=5/10/15/20 and progressive network partitioning from two modules at
|r|>=0.25 to three at |r|>=0.35 and four at |r|>=0.45. These results show
that the workflow remains executable in a more heterogeneous thoracic cancer
context while producing cancer-specific architecture. Because BP selection
and network construction used the full LUAD cohort, the repeated
cross-validation audits centroid fitting only and is not presented as fully
nested external validation. {survival_text}
"""
        (OUT / "text" / "LUAD_REVIEWER_RESPONSE_DRAFT.txt").write_text(
            rebuttal,
            encoding="utf-8",
        )

        methods = """Cross-cancer applicability in TCGA-LUAD

The analysis workflow was repeated de novo in TCGA lung adenocarcinoma
(LUAD). Primary-tumour samples were restricted to one specimen per patient,
and pathologic Stage I/II and Stage III/IV were used as the early- and
late-stage task contrast, respectively. Gene expression was standardized
gene-wise across patients. GO biological-process scores were calculated as
the mean standardized expression of matched genes, with a prespecified
minimum of 10 matched genes. The 30 processes with the largest
D=-log10(p) values from Welch early-versus-late comparisons were locked.
A Pearson BP-correlation network was estimated among late-stage patients,
with |r|>=0.35 as the primary edge threshold and 0.25 and 0.45 as
sensitivity settings. Cancer-specific communities were obtained by greedy
modularity optimization. Module scores were standardized across patients,
and early- and late-stage centroids were derived in module space. Patient
states were summarized by cosine similarity to each centroid, the
late-minus-early similarity margin, and an adverse-direction burden.
Repeated 20x5 stratified cross-validation refitted only the centroids in
training folds; BP selection and module construction were fixed from the
full LUAD cohort and therefore this analysis was not interpreted as fully
nested external validation. A 500-label-permutation analysis was used as a
null audit. Smoking exposure was retained as contextual information and was
not used to define the reconstruction task.
"""
        (OUT / "text" / "LUAD_METHODS_DRAFT.txt").write_text(
            methods,
            encoding="utf-8",
        )

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "TCGA_LUAD_STAGE2D_SYNTHESIS",
            "status": "PASS",
            "reviewer_question_addressed": (
                "Cross-cancer applicability of the reconstruction pipeline"
            ),
            "cancer": "TCGA-LUAD",
            "role": "internal cross-cancer applicability demonstration",
            "primary_results": reviewer_row.iloc[0].to_dict(),
            "outputs": [
                "Table_LUAD_parameter_sensitivity.csv",
                "Table_LUAD_reconstruction_performance.csv",
                "Table_LUAD_survival_adjusted_audit.csv",
                "Table_LUAD_reviewer_facing_summary.csv",
                "LUAD_REVIEWER_RESPONSE_DRAFT.txt",
                "LUAD_METHODS_DRAFT.txt",
            ],
            "next_step": (
                "Create a joint BRCA/KIRC/LUAD cross-cancer synthesis and "
                "integrate it with the external-validation results."
            ),
        }

        with (OUT / "LUAD_STAGE2D_SUMMARY.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        log("LUAD Stage 2D completed: PASS", fh)
        log(
            f"Apparent AUC={apparent_row['auc_orientation_invariant']:.4f}; "
            f"centroid-only held-out AUC="
            f"{cv_row['auc_orientation_invariant']:.4f}",
            fh,
        )


if __name__ == "__main__":
    main()
