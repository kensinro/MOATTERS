# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
TCGA-BRCA standardized reference-cohort audit

Purpose
-------
Re-evaluate the existing TCGA-BRCA patient-level BP-state reconstruction using
the same statistical output contract used for METABRIC and GSE96058.

Important interpretation
------------------------
TCGA-BRCA is the derivation/reference cohort. These are apparent in-sample
associations and must NOT be presented as independent external validation.

Inputs are discovered under D:\MOATTERS-Output:
- BRCA_patient_strategy_master_table.csv
- 99_final_merged_BPstate_clinical_endpoints.csv

No BP selection, module construction, centroid fitting, direction fitting, or
state-threshold fitting is repeated.

Output
------
D:\MOATTERS-Output\MOATTERS_TCGA_REFERENCE_AUDIT
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
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False


ROOT = output_path()
OUT = ROOT / "MOATTERS_TCGA_REFERENCE_AUDIT"

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
    "ER_pathology": "ER_binary",
    "PR_pathology": "PR_binary",
    "HER2_pathology": "HER2_binary",
    "PAM50_luminal": "PAM50_luminal_binary",
    "PAM50_basal": "PAM50_basal_binary",
    "node_positive": "node_positive_binary",
}

SURVIVAL_ENDPOINTS = {
    "OS": ("OS_time_days", "OS_event"),
    "DSS": ("DSS_time_days", "DSS_event"),
    "PFI": ("PFI_time_days", "PFI_event"),
    "DFI": ("DFI_time_days", "DFI_event"),
}


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def find_latest(filename: str) -> Path:
    hits = sorted(
        ROOT.rglob(filename),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {ROOT}")
    return hits[0]


def normalize_patient(x) -> str:
    s = str(x).strip().upper()
    # Convert sample-level TCGA barcodes to patient-level barcode.
    if s.startswith("TCGA-"):
        return "-".join(s.split("-")[:3])
    return s


def binary_from_text(series: pd.Series, positive, negative) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    out = pd.Series(pd.NA, index=series.index, dtype="Float64")
    out[s.isin({str(x).upper() for x in positive})] = 1.0
    out[s.isin({str(x).upper() for x in negative})] = 0.0
    return out


def choose_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def bh_adjust(df, p_col, out_col):
    df = df.copy()
    ok = pd.to_numeric(df[p_col], errors="coerce").notna()
    df[out_col] = np.nan
    if ok.any():
        df.loc[ok, out_col] = multipletests(
            df.loc[ok, p_col].astype(float), method="fdr_bh"
        )[1]
    return df


def rank_biserial_from_u(u, n1, n0):
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
    lo, hi = bootstrap_auc_ci(
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
        "median_difference_positive_minus_negative": float(
            np.median(pos) - np.median(neg)
        ),
        "mann_whitney_U": float(u),
        "p_value": float(p),
        "rank_biserial": float(rank_biserial_from_u(u, len(pos), len(neg))),
        "auc_directional": float(auc),
        "auc_orientation_invariant": float(max(auc, 1.0 - auc)),
        "auc_ci95_low": lo,
        "auc_ci95_high": hi,
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
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = abs(roc_auc_score(rng.permutation(y), x) - 0.5)
    p_perm = (1 + np.sum(null >= obs_abs)) / (N_PERM + 1)
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


def cramers_v(tab):
    chi2, _, _, _ = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    r, k = tab.shape
    denom = min(r - 1, k - 1)
    return float(math.sqrt((chi2 / n) / denom)) if n > 0 and denom > 0 else np.nan


def categorical_test(df, endpoint_name, endpoint_col):
    d = df[[endpoint_col, "patient_state_class"]].dropna().copy()
    d[endpoint_col] = pd.to_numeric(d[endpoint_col], errors="coerce")
    d = d.dropna()
    d = d[d[endpoint_col].isin([0, 1])]
    tab = pd.crosstab(d["patient_state_class"], d[endpoint_col])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None, tab
    chi2, p, dof, _ = chi2_contingency(tab)
    return {
        "endpoint": endpoint_name,
        "endpoint_column": endpoint_col,
        "categorical_variable": "patient_state_class",
        "n": int(tab.to_numpy().sum()),
        "n_categories": int(tab.shape[0]),
        "chi_square": float(chi2),
        "degrees_of_freedom": int(dof),
        "p_value": float(p),
        "cramers_v": cramers_v(tab),
    }, tab


def binary_or_test(df, endpoint_name, endpoint_col):
    d = df[[endpoint_col, "closer_to_late_centroid"]].dropna().copy()
    d[endpoint_col] = pd.to_numeric(d[endpoint_col], errors="coerce").astype(int)
    d["closer_to_late_centroid"] = d["closer_to_late_centroid"].astype(int)
    tab = pd.crosstab(
        d["closer_to_late_centroid"], d[endpoint_col]
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    a, b = float(tab.loc[1, 1]), float(tab.loc[1, 0])
    c, dd = float(tab.loc[0, 1]), float(tab.loc[0, 0])
    if min(a, b, c, dd) == 0:
        a, b, c, dd = a + .5, b + .5, c + .5, dd + .5
    log_or = math.log((a * dd) / (b * c))
    se = math.sqrt(1/a + 1/b + 1/c + 1/dd)
    from scipy.stats import norm
    p = 2 * norm.sf(abs(log_or / se))
    return {
        "endpoint": endpoint_name,
        "endpoint_column": endpoint_col,
        "binary_predictor": "closer_to_late_centroid",
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
    for c in [time_col, event_col, metric]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    d = d[(d[time_col] > 0) & d[event_col].isin([0, 1])]
    if len(d) < 30 or d[event_col].sum() < 10 or d[metric].std(ddof=1) == 0:
        return None, None
    z = (d[metric] - d[metric].mean()) / d[metric].std(ddof=1)
    fit = pd.DataFrame({
        "time": d[time_col],
        "event": d[event_col].astype(int),
        "metric_z": z,
    })
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(fit, duration_col="time", event_col="event")
    coef = float(cph.params_["metric_z"])
    se = float(cph.standard_errors_["metric_z"])
    p = float(cph.summary.loc["metric_z", "p"])
    median = float(d[metric].median())
    high = d[metric] > median
    lr = logrank_test(
        d.loc[high, time_col], d.loc[~high, time_col],
        event_observed_A=d.loc[high, event_col],
        event_observed_B=d.loc[~high, event_col],
    )
    return {
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
    }, pd.DataFrame({
        "time": d[time_col].to_numpy(),
        "event": d[event_col].astype(int).to_numpy(),
        "metric": d[metric].to_numpy(),
    })


def survival_permutation(perm_data, endpoint_name, rng):
    if not LIFELINES_AVAILABLE or perm_data is None:
        return None
    d = perm_data.copy()
    z = (d["metric"] - d["metric"].mean()) / d["metric"].std(ddof=1)
    obs = pd.DataFrame({"time": d["time"], "event": d["event"], "metric_z": z})
    cph = CoxPHFitter()
    cph.fit(obs, duration_col="time", event_col="event")
    obs_abs = abs(float(cph.params_["metric_z"]))
    null = []
    for _ in range(N_PERM):
        pdat = pd.DataFrame({
            "time": d["time"].to_numpy(),
            "event": d["event"].to_numpy(),
            "metric_z": rng.permutation(z.to_numpy()),
        })
        try:
            cp = CoxPHFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cp.fit(pdat, duration_col="time", event_col="event")
            null.append(abs(float(cp.params_["metric_z"])))
        except Exception:
            pass
    null = np.asarray(null, dtype=float)
    return {
        "endpoint": endpoint_name,
        "metric": PRIMARY_METRIC,
        "observed_abs_cox_beta": obs_abs,
        "null_mean_abs_cox_beta": float(null.mean()) if len(null) else np.nan,
        "null_q95_abs_cox_beta": float(np.quantile(null, .95)) if len(null) else np.nan,
        "permutation_p": float((1 + np.sum(null >= obs_abs)) / (1 + len(null))),
        "successful_permutations": int(len(null)),
        "requested_permutations": N_PERM,
    }


def harmonize_endpoints(df):
    out = df.copy()

    out["stage_late_binary"] = binary_from_text(
        out["stage_group"], ["LATE"], ["EARLY"]
    )

    er_col = choose_column(out, [
        "ER_Status_nature2012",
        "breast_carcinoma_estrogen_receptor_status",
    ])
    pr_col = choose_column(out, [
        "PR_Status_nature2012",
        "breast_carcinoma_progesterone_receptor_status",
    ])
    her2_col = choose_column(out, [
        "HER2_Final_Status_nature2012",
        "lab_proc_her2_neu_immunohistochemistry_receptor_status",
    ])
    pam_col = choose_column(out, [
        "PAM50Call_RNAseq",
        "PAM50_mRNA_nature2012",
    ])
    node_col = choose_column(out, [
        "Node_Coded_nature2012",
        "pathologic_N",
    ])

    out["ER_binary"] = binary_from_text(
        out[er_col], ["POSITIVE"], ["NEGATIVE"]
    ) if er_col else pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["PR_binary"] = binary_from_text(
        out[pr_col], ["POSITIVE"], ["NEGATIVE"]
    ) if pr_col else pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["HER2_binary"] = binary_from_text(
        out[her2_col],
        ["POSITIVE"],
        ["NEGATIVE"],
    ) if her2_col else pd.Series(pd.NA, index=out.index, dtype="Float64")

    if pam_col:
        p = out[pam_col].astype("string").str.strip().str.upper()
        out["PAM50_luminal_binary"] = pd.Series(
            np.where(
                p.isin(["LUMA", "LUMB", "LUMINAL A", "LUMINAL B"]),
                1.0,
                np.where(p.notna(), 0.0, np.nan),
            ),
            index=out.index, dtype="Float64"
        )
        out["PAM50_basal_binary"] = pd.Series(
            np.where(
                p.str.contains("BASAL", na=False),
                1.0,
                np.where(p.notna(), 0.0, np.nan),
            ),
            index=out.index, dtype="Float64"
        )
        out["PAM50_clean"] = out[pam_col]

    if node_col:
        n = out[node_col].astype("string").str.strip().str.upper()
        out["node_positive_binary"] = pd.Series(
            np.where(
                n.str.contains("POS", na=False)
                | n.str.match(r"^N[1-3]", na=False)
                | n.isin(["1", "YES"]),
                1.0,
                np.where(
                    n.str.contains("NEG", na=False)
                    | n.str.match(r"^N0", na=False)
                    | n.isin(["0", "NO"]),
                    0.0,
                    np.nan,
                ),
            ),
            index=out.index, dtype="Float64"
        )

    surv_map = {
        "OS": ("OS.time", "OS"),
        "DSS": ("DSS.time", "DSS"),
        "PFI": ("PFI.time", "PFI"),
        "DFI": ("DFI.time", "DFI"),
    }
    for name, (tcol, ecol) in surv_map.items():
        if tcol in out:
            out[f"{name}_time_days"] = pd.to_numeric(out[tcol], errors="coerce")
        if ecol in out:
            out[f"{name}_event"] = pd.to_numeric(out[ecol], errors="coerce")

    out["closer_to_late_centroid"] = (
        pd.to_numeric(out["Dis_late_centroid"], errors="coerce")
        < pd.to_numeric(out["Dis_early_centroid"], errors="coerce")
    ).astype("Int64")

    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "contingency", "audit", "logs"]:
        (OUT / sub).mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)

    with (OUT / "logs" / "tcga_reference_audit.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting TCGA-BRCA standardized reference-cohort audit", fh)

        master_path = find_latest("BRCA_patient_strategy_master_table.csv")
        merged_path = find_latest("99_final_merged_BPstate_clinical_endpoints.csv")
        log(f"Master table: {master_path}", fh)
        log(f"Clinical endpoint table: {merged_path}", fh)

        master = pd.read_csv(master_path, low_memory=False)
        clinical = pd.read_csv(merged_path, low_memory=False)

        master["patient_key"] = master["patient"].map(normalize_patient)
        clinical_patient_col = choose_column(
            clinical, ["patient", "patient_id", "_PATIENT", "_PATIENT_clin"]
        )
        if clinical_patient_col is None:
            raise KeyError("No patient identifier found in final clinical table.")
        clinical["patient_key"] = clinical[clinical_patient_col].map(normalize_patient)

        # Keep endpoint/clinical data from the most complete patient-level row.
        clinical = clinical.drop_duplicates("patient_key", keep="first")
        merged = master.merge(
            clinical.drop(columns=["patient"], errors="ignore"),
            on="patient_key",
            how="left",
            suffixes=("", "_clinical"),
            validate="one_to_one",
        )
        merged = harmonize_endpoints(merged)

        log(f"Merged reference table: {len(merged)} patients, {merged.shape[1]} columns", fh)

        module_metrics = [
            c for c in merged.columns if c.startswith("module_M") and c.endswith("_z")
        ]
        continuous_metrics = CORE_METRICS + module_metrics

        # Hard preflight.
        endpoint_rows = []
        for endpoint, col in BINARY_ENDPOINTS.items():
            x = pd.to_numeric(merged[col], errors="coerce")
            pos, neg = int((x == 1).sum()), int((x == 0).sum())
            endpoint_rows.append({
                "endpoint": endpoint,
                "type": "binary",
                "column_1": col,
                "column_2": "",
                "n_usable": pos + neg,
                "n_events_or_positive": pos,
                "n_non_events_or_negative": neg,
            })
            if pos < 3 or neg < 3:
                raise RuntimeError(
                    f"Binary endpoint preflight failed: {endpoint}, pos={pos}, neg={neg}"
                )

        for endpoint, (tcol, ecol) in SURVIVAL_ENDPOINTS.items():
            t = pd.to_numeric(merged.get(tcol), errors="coerce")
            e = pd.to_numeric(merged.get(ecol), errors="coerce")
            ok = t.gt(0) & e.isin([0, 1])
            endpoint_rows.append({
                "endpoint": endpoint,
                "type": "survival",
                "column_1": tcol,
                "column_2": ecol,
                "n_usable": int(ok.sum()),
                "n_events_or_positive": int((ok & e.eq(1)).sum()),
                "n_non_events_or_negative": int((ok & e.eq(0)).sum()),
            })

        inventory = pd.DataFrame(endpoint_rows)
        inventory.to_csv(
            OUT / "tables" / "TCGA_reference_endpoint_inventory.csv",
            index=False, encoding="utf-8-sig"
        )

        metric_rows = []
        for metric in continuous_metrics:
            x = pd.to_numeric(merged[metric], errors="coerce")
            metric_rows.append({
                "metric": metric,
                "n_nonmissing": int(x.notna().sum()),
                "n_unique": int(x.nunique(dropna=True)),
            })
        pd.DataFrame(metric_rows).to_csv(
            OUT / "audit" / "TCGA_reference_metric_preflight.csv",
            index=False, encoding="utf-8-sig"
        )

        endpoint_family = {
            "stage_late": "derivation_endpoint",
            "ER_pathology": "secondary_pathology",
            "PR_pathology": "secondary_pathology",
            "HER2_pathology": "secondary_pathology",
            "PAM50_luminal": "secondary_intrinsic_subtype",
            "PAM50_basal": "secondary_intrinsic_subtype",
            "node_positive": "secondary_clinicopathologic",
        }

        binary_rows = []
        for endpoint, col in BINARY_ENDPOINTS.items():
            log(f"Binary reference audit: {endpoint}", fh)
            for metric in continuous_metrics:
                row = binary_metric_test(merged, endpoint, col, metric, rng)
                if row:
                    binary_rows.append(row)

        binary = pd.DataFrame(binary_rows)
        binary["endpoint_family"] = binary["endpoint"].map(endpoint_family)
        binary["evidence_role"] = "derivation_reference_apparent_association"
        binary = bh_adjust(binary, "p_value", "q_value_global")
        binary["q_value_within_endpoint"] = np.nan
        for endpoint in binary["endpoint"].unique():
            mask = binary["endpoint"] == endpoint
            binary.loc[mask, "q_value_within_endpoint"] = multipletests(
                binary.loc[mask, "p_value"], method="fdr_bh"
            )[1]
        binary.to_csv(
            OUT / "tables" / "TCGA_binary_endpoint_continuous_metric_tests.csv",
            index=False, encoding="utf-8-sig"
        )

        perm = pd.DataFrame([
            permutation_auc_null(merged, e, c, PRIMARY_METRIC, rng)
            for e, c in BINARY_ENDPOINTS.items()
        ])
        perm["evidence_role"] = "derivation_reference_null_audit"
        perm.to_csv(
            OUT / "audit" / "TCGA_primary_score_binary_permutation_null.csv",
            index=False, encoding="utf-8-sig"
        )

        cat_rows, or_rows = [], []
        for endpoint, col in BINARY_ENDPOINTS.items():
            row, tab = categorical_test(merged, endpoint, col)
            if row:
                cat_rows.append(row)
                tab.to_csv(
                    OUT / "contingency" / f"state_class_by_{endpoint}.csv",
                    encoding="utf-8-sig"
                )
            or_rows.append(binary_or_test(merged, endpoint, col))

        categorical = bh_adjust(pd.DataFrame(cat_rows), "p_value", "q_value")
        categorical["endpoint_family"] = categorical["endpoint"].map(endpoint_family)
        categorical["evidence_role"] = "derivation_reference_apparent_association"
        categorical.to_csv(
            OUT / "tables" / "TCGA_state_class_binary_endpoint_tests.csv",
            index=False, encoding="utf-8-sig"
        )

        odds = bh_adjust(pd.DataFrame(or_rows), "p_value", "q_value")
        odds["endpoint_family"] = odds["endpoint"].map(endpoint_family)
        odds["evidence_role"] = "derivation_reference_apparent_association"
        odds.to_csv(
            OUT / "tables" / "TCGA_late_centroid_proximity_odds_ratios.csv",
            index=False, encoding="utf-8-sig"
        )

        survival_rows, survival_perm = [], []
        if LIFELINES_AVAILABLE:
            for endpoint, (tcol, ecol) in SURVIVAL_ENDPOINTS.items():
                log(f"Survival reference audit: {endpoint}", fh)
                pdata = None
                for metric in continuous_metrics:
                    row, current = cox_metric_test(
                        merged, endpoint, tcol, ecol, metric
                    )
                    if row:
                        survival_rows.append(row)
                    if metric == PRIMARY_METRIC:
                        pdata = current
                prow = survival_permutation(pdata, endpoint, rng)
                if prow:
                    survival_perm.append(prow)

        survival = pd.DataFrame(survival_rows)
        if len(survival):
            survival["evidence_role"] = "derivation_reference_secondary_survival"
            survival = bh_adjust(survival, "p_value", "q_value_global")
            survival["q_value_within_endpoint"] = np.nan
            for endpoint in survival["endpoint"].unique():
                mask = survival["endpoint"] == endpoint
                survival.loc[mask, "q_value_within_endpoint"] = multipletests(
                    survival.loc[mask, "p_value"], method="fdr_bh"
                )[1]
        survival.to_csv(
            OUT / "tables" / "TCGA_survival_continuous_metric_tests.csv",
            index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(survival_perm).to_csv(
            OUT / "audit" / "TCGA_primary_score_survival_permutation_null.csv",
            index=False, encoding="utf-8-sig"
        )

        primary_binary = binary[binary["metric"] == PRIMARY_METRIC].copy()
        primary_binary.to_csv(
            OUT / "tables" / "TCGA_PRIMARY_adverse_burden_binary_results.csv",
            index=False, encoding="utf-8-sig"
        )
        primary_survival = (
            survival[survival["metric"] == PRIMARY_METRIC].copy()
            if len(survival) else pd.DataFrame()
        )
        primary_survival.to_csv(
            OUT / "tables" / "TCGA_PRIMARY_adverse_burden_survival_results.csv",
            index=False, encoding="utf-8-sig"
        )

        merged.to_csv(
            OUT / "audit" / "TCGA_reference_merged_analysis_table.csv.gz",
            index=False, compression="gzip"
        )

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "TCGA_BRCA_STANDARDIZED_REFERENCE_AUDIT",
            "status": "PASS" if LIFELINES_AVAILABLE else "PASS_WITHOUT_SURVIVAL",
            "role": "derivation_reference",
            "n_patients": int(len(merged)),
            "master_table": str(master_path),
            "clinical_endpoint_table": str(merged_path),
            "primary_metric": PRIMARY_METRIC,
            "continuous_metrics": continuous_metrics,
            "binary_endpoints": list(BINARY_ENDPOINTS),
            "survival_endpoints": list(SURVIVAL_ENDPOINTS),
            "bootstrap_iterations": N_BOOT,
            "permutation_iterations": N_PERM,
            "primary_binary_results": primary_binary.to_dict(orient="records"),
            "primary_survival_results": primary_survival.to_dict(orient="records"),
            "interpretive_boundaries": [
                "TCGA-BRCA defined the BP terms, modules, centroids and directions.",
                "All TCGA associations are apparent derivation/reference-cohort associations.",
                "TCGA results must not be described as independent external validation.",
                "External replication is assessed only in METABRIC and GSE96058.",
                "Survival analyses are secondary and do not establish clinical utility.",
            ],
        }
        with (OUT / "TCGA_REFERENCE_AUDIT_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        (OUT / "README_TCGA_REFERENCE_AUDIT.txt").write_text(
            "MOATTERS — TCGA-BRCA standardized reference audit\n\n"
            f"Status: {'PASS' if LIFELINES_AVAILABLE else 'PASS_WITHOUT_SURVIVAL'}\n"
            f"Patients: {len(merged)}\n"
            "Role: derivation/reference cohort\n"
            "This is not independent external validation.\n",
            encoding="utf-8",
        )

        log("TCGA standardized reference audit completed", fh)
        log(
            f"Binary tests={len(binary)}; categorical tests={len(categorical)}; "
            f"survival tests={len(survival)}",
            fh,
        )


if __name__ == "__main__":
    main()
