# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
METABRIC Stage 2D — endpoint-blinded external validation

Design
------
The METABRIC reconstruction was completed before endpoint testing. This stage
does not refit BP terms, modules, centroids, directions, or state thresholds.

Primary locked representation
-----------------------------
K >= 10, 30 BP terms, 7 modules, 1,980 patients.

Validation targets
------------------
Binary:
- late stage
- ER
- PR
- HER2
- PAM50 luminal
- PAM50 basal

Time-to-event:
- overall survival (OS)
- relapse-free survival (RFS)

Primary continuous reconstruction metrics
-----------------------------------------
- adverse_burden
- late_vs_early_similarity_delta
- late_vs_early_distance_delta
- Sim_late_centroid
- Dis_late_centroid

Additional auditable observables
--------------------------------
- seven module z-scores
- closer_to_late_centroid
- patient_state_class

Statistical outputs
-------------------
Binary endpoints:
- group counts and medians
- Mann-Whitney U
- rank-biserial effect size
- ROC AUC and orientation-invariant AUC
- bootstrap 95% CI
- Benjamini-Hochberg FDR
- endpoint-label permutation null for the prespecified primary metric

Survival endpoints:
- univariable Cox model per standardized continuous metric
- hazard ratio and 95% CI
- concordance index
- median-split Kaplan-Meier/log-rank audit
- Benjamini-Hochberg FDR
- endpoint-time/event permutation is not used because it would destroy censoring
  structure; a patient-label permutation null is applied to the primary score
  while preserving paired time/event records.

Categorical outputs:
- state-class x endpoint contingency tables
- chi-square and Cramer's V
- late-centroid proximity x endpoint odds ratios

Output
------
D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE2D_VALIDATION
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False


STAGE1 = output_path(r"MOATTERS_METABRIC_STAGE1")
STAGE2C = output_path(r"MOATTERS_METABRIC_STAGE2C_RECONSTRUCTION")
OUT = output_path(r"MOATTERS_METABRIC_STAGE2D_VALIDATION")

PRIMARY_RECON = STAGE2C / "METABRIC_patient_reconstruction_PRIMARY_K10.csv"
MODULE_Z = STAGE2C / "matrices" / "METABRIC_module_scores_z_K10.csv"
CLINICAL = STAGE1 / "tables" / "METABRIC_primary_clinical_harmonized.tsv"

SEED = 260724
N_BOOT = 2000
N_PERM = 1000

PRIMARY_METRIC = "adverse_burden"

CORE_METRICS = [
    "adverse_burden",
    "late_vs_early_similarity_delta",
    "late_vs_early_distance_delta",
    "Sim_late_centroid",
    "Dis_late_centroid",
]

BINARY_ENDPOINTS = {
    "stage_late": "stage_late_binary",
    "ER": "ER_binary",
    "PR": "PR_binary",
    "HER2": "HER2_binary",
    "PAM50_luminal": "PAM50_luminal_binary",
    "PAM50_basal": "PAM50_basal_binary",
}

SURVIVAL_ENDPOINTS = {
    "OS": ("Overall Survival (Months)", "OS_event"),
    "RFS": ("Relapse Free Status (Months)", "RFS_event"),
}


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def bh_adjust(df: pd.DataFrame, p_col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()
    ok = pd.to_numeric(df[p_col], errors="coerce").notna()
    df[out_col] = np.nan
    if ok.any():
        df.loc[ok, out_col] = multipletests(
            df.loc[ok, p_col].astype(float), method="fdr_bh"
        )[1]
    return df


def rank_biserial_from_u(u: float, n1: int, n0: int) -> float:
    # Positive means values tend to be higher in endpoint-positive patients.
    return 2.0 * u / (n1 * n0) - 1.0


def bootstrap_auc_ci(y, x, rng, n_boot=N_BOOT):
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(roc_auc_score(yy, x[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def binary_metric_test(df, endpoint_name, endpoint_col, metric, rng):
    d = df[[endpoint_col, metric]].copy()
    d[endpoint_col] = pd.to_numeric(d[endpoint_col], errors="coerce")
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna()
    d = d[d[endpoint_col].isin([0, 1])]

    pos = d.loc[d[endpoint_col] == 1, metric].to_numpy(float)
    neg = d.loc[d[endpoint_col] == 0, metric].to_numpy(float)

    if len(pos) < 3 or len(neg) < 3:
        return None

    u, p = mannwhitneyu(pos, neg, alternative="two-sided")
    auc = roc_auc_score(d[endpoint_col].astype(int), d[metric])
    ci_lo, ci_hi = bootstrap_auc_ci(
        d[endpoint_col].astype(int).to_numpy(),
        d[metric].to_numpy(),
        rng,
    )

    return {
        "endpoint": endpoint_name,
        "endpoint_column": endpoint_col,
        "metric": metric,
        "n_total": len(d),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "median_positive": float(np.median(pos)),
        "median_negative": float(np.median(neg)),
        "median_difference_positive_minus_negative": float(np.median(pos) - np.median(neg)),
        "mann_whitney_U": float(u),
        "p_value": float(p),
        "rank_biserial": float(rank_biserial_from_u(u, len(pos), len(neg))),
        "auc_directional": float(auc),
        "auc_orientation_invariant": float(max(auc, 1.0 - auc)),
        "auc_ci95_low": ci_lo,
        "auc_ci95_high": ci_hi,
    }


def permutation_auc_null(df, endpoint_name, endpoint_col, metric, rng):
    d = df[[endpoint_col, metric]].copy()
    d[endpoint_col] = pd.to_numeric(d[endpoint_col], errors="coerce")
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna()
    d = d[d[endpoint_col].isin([0, 1])]

    y = d[endpoint_col].astype(int).to_numpy()
    x = d[metric].to_numpy(float)
    obs = roc_auc_score(y, x)
    obs_abs = abs(obs - 0.5)

    null = np.empty(N_PERM, dtype=float)
    for i in range(N_PERM):
        yp = rng.permutation(y)
        null[i] = abs(roc_auc_score(yp, x) - 0.5)

    p_perm = (1.0 + np.sum(null >= obs_abs)) / (N_PERM + 1.0)
    return {
        "endpoint": endpoint_name,
        "endpoint_column": endpoint_col,
        "metric": metric,
        "n": len(d),
        "observed_auc_directional": float(obs),
        "observed_abs_auc_minus_0_5": float(obs_abs),
        "null_mean_abs_auc_minus_0_5": float(null.mean()),
        "null_q95_abs_auc_minus_0_5": float(np.quantile(null, 0.95)),
        "permutation_p": float(p_perm),
        "n_permutations": N_PERM,
    }


def cramers_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    denom = min(k - 1, r - 1)
    return float(math.sqrt((chi2 / n) / denom)) if n > 0 and denom > 0 else np.nan


def categorical_endpoint_test(df, endpoint_name, endpoint_col, cat_col):
    d = df[[endpoint_col, cat_col]].dropna().copy()
    d[endpoint_col] = pd.to_numeric(d[endpoint_col], errors="coerce")
    d = d.dropna()
    d = d[d[endpoint_col].isin([0, 1])]
    tab = pd.crosstab(d[cat_col], d[endpoint_col])

    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None, tab

    chi2, p, dof, _ = chi2_contingency(tab)
    result = {
        "endpoint": endpoint_name,
        "endpoint_column": endpoint_col,
        "categorical_variable": cat_col,
        "n": int(tab.to_numpy().sum()),
        "n_categories": int(tab.shape[0]),
        "chi_square": float(chi2),
        "degrees_of_freedom": int(dof),
        "p_value": float(p),
        "cramers_v": cramers_v(tab),
    }
    return result, tab


def binary_or_test(df, endpoint_name, endpoint_col, predictor_col):
    d = df[[endpoint_col, predictor_col]].dropna().copy()
    d[endpoint_col] = pd.to_numeric(d[endpoint_col], errors="coerce").astype(int)
    d[predictor_col] = d[predictor_col].astype(int)
    tab = pd.crosstab(d[predictor_col], d[endpoint_col]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )

    a = float(tab.loc[1, 1])
    b = float(tab.loc[1, 0])
    c = float(tab.loc[0, 1])
    dd = float(tab.loc[0, 0])

    # Haldane-Anscombe correction if needed.
    if min(a, b, c, dd) == 0:
        a, b, c, dd = a + 0.5, b + 0.5, c + 0.5, dd + 0.5

    log_or = math.log((a * dd) / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
    z = log_or / se
    from scipy.stats import norm
    p = 2 * norm.sf(abs(z))

    return {
        "endpoint": endpoint_name,
        "endpoint_column": endpoint_col,
        "binary_predictor": predictor_col,
        "n": int(tab.to_numpy().sum()),
        "odds_ratio": float(math.exp(log_or)),
        "or_ci95_low": float(math.exp(log_or - 1.96 * se)),
        "or_ci95_high": float(math.exp(log_or + 1.96 * se)),
        "p_value": float(p),
        "table_00": int(tab.loc[0, 0]),
        "table_01": int(tab.loc[0, 1]),
        "table_10": int(tab.loc[1, 0]),
        "table_11": int(tab.loc[1, 1]),
    }


def cox_metric_test(df, endpoint_name, time_col, event_col, metric):
    if not LIFELINES_AVAILABLE:
        return None, None

    d = df[[time_col, event_col, metric]].copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[event_col] = pd.to_numeric(d[event_col], errors="coerce")
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna()
    d = d[(d[time_col] > 0) & d[event_col].isin([0, 1])]

    if len(d) < 30 or d[event_col].sum() < 10 or d[metric].std(ddof=1) == 0:
        return None, None

    z = (d[metric] - d[metric].mean()) / d[metric].std(ddof=1)
    fit_df = pd.DataFrame({
        "time": d[time_col],
        "event": d[event_col].astype(int),
        "metric_z": z,
    })

    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(fit_df, duration_col="time", event_col="event")

    coef = float(cph.params_["metric_z"])
    se = float(cph.standard_errors_["metric_z"])
    p = float(cph.summary.loc["metric_z", "p"])

    median = float(d[metric].median())
    high = d[metric] > median
    lr = logrank_test(
        d.loc[high, time_col],
        d.loc[~high, time_col],
        event_observed_A=d.loc[high, event_col],
        event_observed_B=d.loc[~high, event_col],
    )

    result = {
        "endpoint": endpoint_name,
        "time_column": time_col,
        "event_column": event_col,
        "metric": metric,
        "n": len(d),
        "n_events": int(d[event_col].sum()),
        "cox_beta_per_SD": coef,
        "hazard_ratio_per_SD": float(math.exp(coef)),
        "hr_ci95_low": float(math.exp(coef - 1.96 * se)),
        "hr_ci95_high": float(math.exp(coef + 1.96 * se)),
        "p_value": p,
        "concordance_index": float(cph.concordance_index_),
        "median_split_threshold": median,
        "median_split_logrank_p": float(lr.p_value),
        "n_high": int(high.sum()),
        "n_low": int((~high).sum()),
    }

    # Return patient-complete data for primary-metric permutation.
    perm_data = pd.DataFrame({
        "time": d[time_col].to_numpy(),
        "event": d[event_col].astype(int).to_numpy(),
        "metric": d[metric].to_numpy(),
    })
    return result, perm_data


def survival_primary_permutation(perm_data, endpoint_name, rng):
    if not LIFELINES_AVAILABLE or perm_data is None:
        return None

    d = perm_data.copy()
    z = (d["metric"] - d["metric"].mean()) / d["metric"].std(ddof=1)
    obs_df = pd.DataFrame({"time": d["time"], "event": d["event"], "metric_z": z})
    cph = CoxPHFitter()
    cph.fit(obs_df, duration_col="time", event_col="event")
    obs_beta = float(cph.params_["metric_z"])
    obs_abs = abs(obs_beta)

    null = []
    for _ in range(N_PERM):
        # Preserve each patient's paired time/event record and permute only the score.
        zp = rng.permutation(z.to_numpy())
        p_df = pd.DataFrame({
            "time": d["time"].to_numpy(),
            "event": d["event"].to_numpy(),
            "metric_z": zp,
        })
        try:
            cp = CoxPHFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cp.fit(p_df, duration_col="time", event_col="event")
            null.append(abs(float(cp.params_["metric_z"])))
        except Exception:
            continue

    null = np.asarray(null, dtype=float)
    p_perm = (1 + np.sum(null >= obs_abs)) / (1 + len(null))
    return {
        "endpoint": endpoint_name,
        "metric": PRIMARY_METRIC,
        "observed_abs_cox_beta": obs_abs,
        "null_mean_abs_cox_beta": float(null.mean()) if len(null) else np.nan,
        "null_q95_abs_cox_beta": float(np.quantile(null, 0.95)) if len(null) else np.nan,
        "permutation_p": float(p_perm),
        "successful_permutations": int(len(null)),
        "requested_permutations": N_PERM,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "contingency", "audit", "logs"]:
        (OUT / sub).mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)

    with (OUT / "logs" / "stage2d_validation.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting METABRIC endpoint-blinded external validation", fh)

        for path in [PRIMARY_RECON, MODULE_Z, CLINICAL]:
            if not path.exists():
                raise FileNotFoundError(f"Required input not found: {path}")

        recon = pd.read_csv(PRIMARY_RECON, low_memory=False)
        clinical = pd.read_csv(CLINICAL, sep="\t", low_memory=False)
        module_z = pd.read_csv(MODULE_Z, index_col=0, low_memory=False)

        patient_col = "sample_id" if "sample_id" in clinical.columns else "patient_id"
        clinical = clinical.rename(columns={patient_col: "patient"})

        module_patient = module_z.T.copy()
        module_patient.index.name = "patient"
        module_patient = module_patient.add_prefix("module_z_").reset_index()

        merged = recon.merge(clinical, on="patient", how="left")
        merged = merged.merge(module_patient, on="patient", how="left")

        log(
            f"Merged validation table: {merged.shape[0]} patients, "
            f"{merged.shape[1]} columns",
            fh,
        )

        module_metrics = [c for c in merged.columns if c.startswith("module_z_")]
        continuous_metrics = CORE_METRICS + module_metrics

        # Inventory and endpoint completeness.
        inventory_rows = []
        for name, col in BINARY_ENDPOINTS.items():
            x = pd.to_numeric(merged[col], errors="coerce")
            inventory_rows.append({
                "endpoint": name,
                "type": "binary",
                "column_1": col,
                "column_2": "",
                "n_usable": int(x.isin([0, 1]).sum()),
                "n_events_or_positive": int((x == 1).sum()),
                "n_non_events_or_negative": int((x == 0).sum()),
            })
        for name, (tcol, ecol) in SURVIVAL_ENDPOINTS.items():
            t = pd.to_numeric(merged[tcol], errors="coerce")
            e = pd.to_numeric(merged[ecol], errors="coerce")
            ok = t.gt(0) & e.isin([0, 1])
            inventory_rows.append({
                "endpoint": name,
                "type": "survival",
                "column_1": tcol,
                "column_2": ecol,
                "n_usable": int(ok.sum()),
                "n_events_or_positive": int((ok & e.eq(1)).sum()),
                "n_non_events_or_negative": int((ok & e.eq(0)).sum()),
            })
        inventory = pd.DataFrame(inventory_rows)
        inventory.to_csv(
            OUT / "tables" / "METABRIC_validation_endpoint_inventory.csv",
            index=False, encoding="utf-8-sig"
        )

        # Continuous metric x binary endpoint.
        binary_rows = []
        for endpoint_name, endpoint_col in BINARY_ENDPOINTS.items():
            log(f"Binary validation: {endpoint_name}", fh)
            for metric in continuous_metrics:
                row = binary_metric_test(
                    merged, endpoint_name, endpoint_col, metric, rng
                )
                if row is not None:
                    binary_rows.append(row)

        binary_results = pd.DataFrame(binary_rows)
        binary_results = bh_adjust(binary_results, "p_value", "q_value_global")
        binary_results["q_value_within_endpoint"] = np.nan
        for endpoint in binary_results["endpoint"].unique():
            mask = binary_results["endpoint"] == endpoint
            binary_results.loc[mask, "q_value_within_endpoint"] = multipletests(
                binary_results.loc[mask, "p_value"], method="fdr_bh"
            )[1]
        binary_results.to_csv(
            OUT / "tables" / "METABRIC_binary_endpoint_continuous_metric_tests.csv",
            index=False, encoding="utf-8-sig"
        )

        # Primary-score permutation nulls.
        perm_rows = []
        for endpoint_name, endpoint_col in BINARY_ENDPOINTS.items():
            perm_rows.append(
                permutation_auc_null(
                    merged, endpoint_name, endpoint_col, PRIMARY_METRIC, rng
                )
            )
        pd.DataFrame(perm_rows).to_csv(
            OUT / "audit" / "METABRIC_primary_score_binary_permutation_null.csv",
            index=False, encoding="utf-8-sig"
        )

        # Categorical state class and binary late-centroid proximity.
        categorical_rows = []
        odds_rows = []
        for endpoint_name, endpoint_col in BINARY_ENDPOINTS.items():
            row, tab = categorical_endpoint_test(
                merged, endpoint_name, endpoint_col, "patient_state_class"
            )
            if row is not None:
                categorical_rows.append(row)
                tab.to_csv(
                    OUT / "contingency" /
                    f"state_class_by_{endpoint_name}.csv",
                    encoding="utf-8-sig"
                )
            odds_rows.append(
                binary_or_test(
                    merged, endpoint_name, endpoint_col,
                    "closer_to_late_centroid"
                )
            )

        categorical = pd.DataFrame(categorical_rows)
        categorical = bh_adjust(categorical, "p_value", "q_value")
        categorical.to_csv(
            OUT / "tables" / "METABRIC_state_class_binary_endpoint_tests.csv",
            index=False, encoding="utf-8-sig"
        )

        odds = pd.DataFrame(odds_rows)
        odds = bh_adjust(odds, "p_value", "q_value")
        odds.to_csv(
            OUT / "tables" / "METABRIC_late_centroid_proximity_odds_ratios.csv",
            index=False, encoding="utf-8-sig"
        )

        # Survival.
        survival_rows = []
        survival_perm_rows = []
        if LIFELINES_AVAILABLE:
            for endpoint_name, (time_col, event_col) in SURVIVAL_ENDPOINTS.items():
                log(f"Survival validation: {endpoint_name}", fh)
                primary_perm_data = None
                for metric in continuous_metrics:
                    row, pdata = cox_metric_test(
                        merged, endpoint_name, time_col, event_col, metric
                    )
                    if row is not None:
                        survival_rows.append(row)
                    if metric == PRIMARY_METRIC:
                        primary_perm_data = pdata
                perm_row = survival_primary_permutation(
                    primary_perm_data, endpoint_name, rng
                )
                if perm_row is not None:
                    survival_perm_rows.append(perm_row)
        else:
            log("lifelines is unavailable; survival analysis skipped", fh)

        survival = pd.DataFrame(survival_rows)
        if len(survival):
            survival = bh_adjust(survival, "p_value", "q_value_global")
            survival["q_value_within_endpoint"] = np.nan
            for endpoint in survival["endpoint"].unique():
                mask = survival["endpoint"] == endpoint
                survival.loc[mask, "q_value_within_endpoint"] = multipletests(
                    survival.loc[mask, "p_value"], method="fdr_bh"
                )[1]
        survival.to_csv(
            OUT / "tables" / "METABRIC_survival_continuous_metric_tests.csv",
            index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(survival_perm_rows).to_csv(
            OUT / "audit" / "METABRIC_primary_score_survival_permutation_null.csv",
            index=False, encoding="utf-8-sig"
        )

        # Compact primary-results table.
        primary_binary = binary_results[
            binary_results["metric"] == PRIMARY_METRIC
        ].copy()
        primary_binary["analysis_family"] = "binary_endpoint"
        primary_binary.to_csv(
            OUT / "tables" / "METABRIC_PRIMARY_adverse_burden_binary_results.csv",
            index=False, encoding="utf-8-sig"
        )

        primary_survival = (
            survival[survival["metric"] == PRIMARY_METRIC].copy()
            if len(survival) else pd.DataFrame()
        )
        primary_survival.to_csv(
            OUT / "tables" / "METABRIC_PRIMARY_adverse_burden_survival_results.csv",
            index=False, encoding="utf-8-sig"
        )

        # Ranked module associations for interpretation, not module selection.
        ranked_binary = binary_results.sort_values(
            ["endpoint", "q_value_within_endpoint", "auc_orientation_invariant"],
            ascending=[True, True, False],
        )
        ranked_binary.to_csv(
            OUT / "tables" / "METABRIC_ranked_binary_associations.csv",
            index=False, encoding="utf-8-sig"
        )

        if len(survival):
            survival.sort_values(
                ["endpoint", "q_value_within_endpoint", "p_value"]
            ).to_csv(
                OUT / "tables" / "METABRIC_ranked_survival_associations.csv",
                index=False, encoding="utf-8-sig"
            )

        # Machine-readable summary.
        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "METABRIC_STAGE2D_ENDPOINT_BLINDED_EXTERNAL_VALIDATION",
            "status": "PASS" if LIFELINES_AVAILABLE else "PASS_WITHOUT_SURVIVAL",
            "n_patients": int(merged.shape[0]),
            "primary_reconstruction": "K>=10; locked 30-BP, 7-module space",
            "primary_metric": PRIMARY_METRIC,
            "binary_endpoints": list(BINARY_ENDPOINTS),
            "survival_endpoints": list(SURVIVAL_ENDPOINTS),
            "continuous_metrics_tested": continuous_metrics,
            "bootstrap_iterations": N_BOOT,
            "permutation_iterations": N_PERM,
            "lifelines_available": LIFELINES_AVAILABLE,
            "primary_binary_results": primary_binary.to_dict(orient="records"),
            "primary_survival_results": primary_survival.to_dict(orient="records"),
            "interpretive_boundaries": [
                "METABRIC endpoints were not used during reconstruction.",
                "Associations quantify external alignment, not clinical utility.",
                "Module-level tests are descriptive external audits and do not redefine modules.",
                "Direction-free AUC is reported alongside directional AUC to avoid hiding inversion.",
                "Multiple testing is controlled globally and within endpoint families.",
            ],
            "next_step": (
                "Review effect sizes, confidence intervals, null results and endpoint-specific "
                "patterns; then prepare manuscript-ready tables/figures and a bounded external-"
                "validation claim."
            ),
        }
        with (OUT / "METABRIC_STAGE2D_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        merged.to_csv(
            OUT / "audit" / "METABRIC_STAGE2D_merged_analysis_table.csv.gz",
            index=False, compression="gzip"
        )

        (OUT / "README_STAGE2D.txt").write_text(
            "MOATTERS — METABRIC Stage 2D\n\n"
            f"Status: {'PASS' if LIFELINES_AVAILABLE else 'PASS_WITHOUT_SURVIVAL'}\n"
            f"Patients: {merged.shape[0]}\n"
            "Primary reconstruction: K>=10, 30 BP, 7 modules\n"
            f"Bootstrap: {N_BOOT}\n"
            f"Permutation: {N_PERM}\n\n"
            "The reconstruction was completed without METABRIC endpoint labels.\n"
            "This stage evaluates external alignment and survival relevance; it does not "
            "claim clinical utility or refit the representation.\n",
            encoding="utf-8",
        )

        log("Stage 2D completed", fh)
        log(
            f"Binary tests={len(binary_results)}; "
            f"categorical tests={len(categorical)}; "
            f"survival tests={len(survival)}",
            fh,
        )


if __name__ == "__main__":
    main()
