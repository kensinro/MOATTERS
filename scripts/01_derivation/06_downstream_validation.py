# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
"""
MOATTERS BRCA BP-state downstream validation
========================================

Date context: 2026-05-31

This script continues from:

D:/MOATTERS-Output/MOATTERS_BRCA_PATIENT_STATE_RANDOM_TEST_V1_20260531/

It performs:
1. Bootstrap stability analysis for stage-associated BP-state metrics.
2. PAM50 / molecular subtype secondary endpoint test.
3. Survival endpoint tests: OS, DSS, PFI, DFI.
4. Supplementary clinical endpoint scan:
   ER, PR, HER2, Node, T/N/M, cancer status, new tumor event, etc.
5. Integration of existing random-control results:
   random BP modules, stage-label shuffle, patient scramble.

Outputs:
D:/MOATTERS-Output/MOATTERS_BRCA_STATE_DownstreamValidation_YYYYMMDD_HHMMSS/

Important:
- Robust file reading for UTF-8, UTF-8-SIG, UTF-16, CP1252, Latin-1.
- Outputs CSV as UTF-8-SIG for Excel compatibility.
- Does not overwrite previous result folders.
"""

import os
import re
import json
import math
import warnings
from pathlib import Path
from moatters.config import data_path, output_path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# =============================================================================
# 0. PATH SETTINGS
# =============================================================================

BRCA_DIR = data_path(r"UCSC_XENA/Breast Cancer (BRCA)")
PHENO_PATH = BRCA_DIR / "Phenotype.tsv"
CLINICAL_PATH = BRCA_DIR / "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"

RANDOM_TEST_DIR = output_path(r"MOATTERS_BRCA_PATIENT_STATE_RANDOM_TEST_V1_20260531")

PATIENT_METRICS_PATH = RANDOM_TEST_DIR / "BRCA_real_patient_centroid_metrics.csv"
REAL_MODULE_STATS_PATH = RANDOM_TEST_DIR / "BRCA_real_module_stage_stats.csv"
REAL_RANDOM_METRICS_PATH = RANDOM_TEST_DIR / "BRCA_real_random_test_metrics.json"

RANDOM_ALL_COMPARE_PATH = RANDOM_TEST_DIR / "BRCA_RANDOM_TEST_all_compare.csv"
RANDOM_SUMMARY_PATH = RANDOM_TEST_DIR / "BRCA_RANDOM_TEST_summary.json"

RANDOM_BP_MODULE_COMPARE_PATH = RANDOM_TEST_DIR / "BRCA_random_BP_module_compare.csv"
STAGE_LABEL_SHUFFLE_COMPARE_PATH = RANDOM_TEST_DIR / "BRCA_stage_label_shuffle_compare.csv"
PATIENT_SCRAMBLE_COMPARE_PATH = RANDOM_TEST_DIR / "BRCA_patient_scramble_compare.csv"

OUT_ROOT = output_path()
RUN_NAME = "MOATTERS_BRCA_STATE_DownstreamValidation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUT_ROOT / RUN_NAME
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"

for d in [OUT_DIR, TABLE_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. BASIC UTILITIES
# =============================================================================

def log(msg):
    print(msg)
    with open(LOG_DIR / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def safe_filename(x, max_len=160):
    s = str(x)
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:max_len]


def write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[WRITE CSV] {path} | shape={df.shape}")


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    log(f"[WRITE JSON] {path}")


def read_json_smart(path):
    path = Path(path)
    if not path.exists():
        log(f"[WARN] JSON not found: {path}")
        return None
    encodings = ["utf-8-sig", "utf-8", "utf-16", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                obj = json.load(f)
            log(f"[READ JSON OK] {path} | encoding={enc}")
            return obj
        except Exception as e:
            last_err = e
    log(f"[WARN] Could not read JSON: {path} | last error={last_err}")
    return None


def read_table_smart(path, sep=None, low_memory=False):
    """
    Robust table reader for UCSC/Xena/local TSV/CSV files.
    Tries UTF-8, UTF-8-SIG, UTF-16, UTF-16LE/BE, CP1252, Latin-1.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    encodings = [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp1252",
        "latin1",
    ]

    seps = [sep] if sep is not None else ["\t", ","]

    last_err = None
    for enc in encodings:
        for s in seps:
            try:
                df = pd.read_csv(
                    path,
                    sep=s,
                    encoding=enc,
                    low_memory=low_memory,
                )
                if df.shape[1] >= 2:
                    log(f"[READ OK] {path} | encoding={enc} | sep={repr(s)} | shape={df.shape}")
                    return df
            except Exception as e:
                last_err = e

    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=enc,
                low_memory=low_memory,
            )
            log(f"[READ OK fallback] {path} | encoding={enc} | sep=infer | shape={df.shape}")
            return df
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Cannot read table: {path}; last error={last_err}")


def normalize_tcga_patient_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(".", "-")
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s, flags=re.I)
    if m:
        return m.group(1).upper()
    return s.upper()


def normalize_text_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    bad = {
        "nan", "na", "n/a", "null", "none",
        "not available", "not reported", "unknown",
        "[not available]", "[not reported]", "[unknown]",
        "--", "-", "?"
    }
    if s.lower() in bad:
        return np.nan
    return s


def d_from_p(p):
    if p is None or pd.isna(p):
        return np.nan
    p = float(p)
    if p <= 0:
        return np.inf
    return -math.log10(p)


def infer_patient_column(df):
    candidates = [
        "patient", "Patient", "PATIENT", "_PATIENT",
        "sample", "Sample", "SAMPLE",
        "sample_id", "sampleID", "SampleID",
        "Hybridization REF",
        "submitter_id",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    best_col = None
    best_frac = 0.0
    for c in df.columns[:80]:
        vals = df[c].dropna().astype(str).head(300)
        if len(vals) == 0:
            continue
        frac = vals.str.contains(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", case=False, regex=True).mean()
        if frac > best_frac:
            best_frac = frac
            best_col = c

    if best_col is not None and best_frac > 0.20:
        return best_col

    raise ValueError("Could not infer patient/sample column.")


def find_cols(df, patterns):
    out = []
    for c in df.columns:
        cs = str(c)
        for pat in patterns:
            if re.search(pat, cs, flags=re.I):
                out.append(c)
                break
    return out


def find_first_col(df, patterns):
    cols = find_cols(df, patterns)
    return cols[0] if cols else None


def simplify_stage_group(x):
    if pd.isna(x):
        return np.nan
    s0 = str(x).strip()
    s = s0.lower()
    s = s.replace("stage", "").replace(" ", "").replace("_", "").replace("-", "")
    s = s.replace("pathologic", "").replace("clinical", "")

    if "early" in s:
        return "Early"
    if "late" in s:
        return "Late"

    if s in {"i", "ia", "ib", "ii", "iia", "iib", "i/ii", "12"}:
        return "Early"
    if s in {"iii", "iiia", "iiib", "iiic", "iv", "iva", "ivb", "iii/iv", "34"}:
        return "Late"

    if re.match(r"^i[ab]?$", s):
        return "Early"
    if re.match(r"^ii[ab]?$", s):
        return "Early"
    if re.match(r"^iii[abc]?$", s):
        return "Late"
    if re.match(r"^iv[ab]?$", s):
        return "Late"

    return np.nan


def simplify_pam50(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    sl = s.lower()

    if sl in {"", "nan", "na", "n/a", "not available", "not reported", "unknown"}:
        return np.nan
    if "basal" in sl:
        return "Basal"
    if "her2" in sl or "her-2" in sl:
        return "Her2"
    if "luma" in sl or "luminal a" in sl:
        return "LumA"
    if "lumb" in sl or "luminal b" in sl:
        return "LumB"
    if "normal" in sl:
        return "Normal"
    return s


def clean_binary_status(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)

    if s in {"", "nan", "na", "n/a", "not available", "not reported", "unknown", "--"}:
        return np.nan

    if "positive" in s and "negative" not in s:
        return "Positive"
    if "negative" in s:
        return "Negative"
    if s in {"pos", "yes", "present", "1", "true"}:
        return "Positive"
    if s in {"neg", "no", "absent", "0", "false"}:
        return "Negative"

    return str(x).strip()


def simplify_node_status(x):
    if pd.isna(x):
        return np.nan
    s0 = str(x).strip()
    s = s0.lower()
    if s in {"", "nan", "na", "n/a", "not available", "not reported", "unknown"}:
        return np.nan
    if "negative" in s:
        return "Node negative"
    if "positive" in s:
        return "Node positive"
    if re.search(r"\bn0\b", s):
        return "Node negative"
    if re.search(r"\bn[123]\b", s):
        return "Node positive"
    if s in {"0", "no", "false"}:
        return "Node negative"
    if s in {"1", "yes", "true"}:
        return "Node positive"
    return s0


def simplify_tnm_stage(x, prefix):
    """
    Clean T/N/M-like values into coarse groups if possible.
    prefix: "T", "N", "M"
    """
    if pd.isna(x):
        return np.nan
    s0 = str(x).strip()
    s = s0.lower()
    if s in {"", "nan", "na", "n/a", "not available", "not reported", "unknown"}:
        return np.nan

    p = prefix.lower()
    m = re.search(rf"\b{p}[0-9x]+[a-z]?\b", s)
    if m:
        val = m.group(0).upper()
        return val

    if prefix == "N":
        if "negative" in s:
            return "N0"
        if "positive" in s:
            return "N+"

    return s0


def welch_two_group(values, groups, pos="Late", neg="Early"):
    tmp = pd.DataFrame({"x": values, "g": groups}).dropna()
    tmp["x"] = pd.to_numeric(tmp["x"], errors="coerce")
    tmp = tmp.dropna()
    x1 = tmp.loc[tmp["g"] == pos, "x"].values
    x0 = tmp.loc[tmp["g"] == neg, "x"].values

    if len(x1) < 5 or len(x0) < 5:
        return {
            "n_pos": len(x1),
            "n_neg": len(x0),
            "mean_pos": np.nan,
            "mean_neg": np.nan,
            "delta_pos_minus_neg": np.nan,
            "p_welch": np.nan,
            "D_welch": np.nan,
        }

    res = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit")
    p = float(res.pvalue)
    return {
        "n_pos": int(len(x1)),
        "n_neg": int(len(x0)),
        "mean_pos": float(np.mean(x1)),
        "mean_neg": float(np.mean(x0)),
        "delta_pos_minus_neg": float(np.mean(x1) - np.mean(x0)),
        "p_welch": p,
        "D_welch": d_from_p(p),
    }


def mannwhitney_two_group(values, groups, pos="Late", neg="Early"):
    tmp = pd.DataFrame({"x": values, "g": groups}).dropna()
    tmp["x"] = pd.to_numeric(tmp["x"], errors="coerce")
    tmp = tmp.dropna()
    x1 = tmp.loc[tmp["g"] == pos, "x"].values
    x0 = tmp.loc[tmp["g"] == neg, "x"].values

    if len(x1) < 5 or len(x0) < 5:
        return {"p_mannwhitney": np.nan, "D_mannwhitney": np.nan}

    try:
        res = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        p = float(res.pvalue)
    except Exception:
        p = np.nan

    return {
        "p_mannwhitney": p,
        "D_mannwhitney": d_from_p(p) if not pd.isna(p) else np.nan,
    }


def kruskal_endpoint(df, metric_col, group_col, min_group_n=10):
    tmp = df[[metric_col, group_col]].copy()
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp[group_col] = tmp[group_col].map(normalize_text_value)
    tmp = tmp.dropna()

    counts = tmp[group_col].value_counts()
    keep = counts[counts >= min_group_n].index.tolist()
    tmp = tmp[tmp[group_col].isin(keep)]

    if tmp[group_col].nunique() < 2:
        return None, pd.DataFrame()

    arrays = [sub[metric_col].values for _, sub in tmp.groupby(group_col)]
    try:
        stat, p = stats.kruskal(*arrays)
    except Exception:
        return None, pd.DataFrame()

    summary = (
        tmp.groupby(group_col)[metric_col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(columns={group_col: "group"})
        .sort_values("mean")
    )

    result = {
        "endpoint": group_col,
        "metric": metric_col,
        "n_total": int(tmp.shape[0]),
        "n_groups": int(tmp[group_col].nunique()),
        "kruskal_stat": float(stat),
        "p_kruskal": float(p),
        "D_kruskal": d_from_p(float(p)),
        "min_group_n": int(min_group_n),
        "groups_kept": "|".join(map(str, keep)),
    }
    return result, summary


def simple_logrank(time, event, group):
    """
    Manual two-group log-rank test.
    """
    df = pd.DataFrame({
        "time": pd.to_numeric(time, errors="coerce"),
        "event": pd.to_numeric(event, errors="coerce"),
        "group": group,
    }).dropna()

    df = df[df["time"] > 0].copy()
    df["event"] = (df["event"] > 0).astype(int)

    groups = list(df["group"].dropna().unique())
    if len(groups) != 2:
        return np.nan, np.nan, df

    g1 = groups[0]
    event_times = np.sort(df.loc[df["event"] == 1, "time"].unique())

    O1 = 0.0
    E1 = 0.0
    V1 = 0.0

    for t in event_times:
        at_risk = df["time"] >= t
        n = int(at_risk.sum())
        if n <= 1:
            continue
        n1 = int((at_risk & (df["group"] == g1)).sum())

        events_at_t = (df["time"] == t) & (df["event"] == 1)
        d = int(events_at_t.sum())
        d1 = int((events_at_t & (df["group"] == g1)).sum())

        if d == 0:
            continue

        O1 += d1
        E1 += d * n1 / n

        if n > 1:
            V1 += (n1 * (n - n1) * d * (n - d)) / (n * n * (n - 1))

    if V1 <= 0:
        return np.nan, np.nan, df

    chi2 = (O1 - E1) ** 2 / V1
    p = 1.0 - stats.chi2.cdf(chi2, df=1)
    return float(p), float(chi2), df


def km_curve(time, event):
    df = pd.DataFrame({
        "time": pd.to_numeric(time, errors="coerce"),
        "event": pd.to_numeric(event, errors="coerce"),
    }).dropna()
    df = df[df["time"] > 0].copy()
    if df.empty:
        return np.array([]), np.array([])

    df["event"] = (df["event"] > 0).astype(int)
    event_times = np.sort(df.loc[df["event"] == 1, "time"].unique())

    x = [0.0]
    y = [1.0]
    s = 1.0

    for t in event_times:
        at_risk = (df["time"] >= t).sum()
        d = ((df["time"] == t) & (df["event"] == 1)).sum()
        if at_risk > 0:
            s *= (1.0 - d / at_risk)
            x.extend([t, t])
            y.extend([y[-1], s])

    return np.array(x), np.array(y)


def plot_box_by_group(df, metric_col, group_col, out_path, title=None):
    tmp = df[[metric_col, group_col]].copy()
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp[group_col] = tmp[group_col].map(normalize_text_value)
    tmp = tmp.dropna()

    if tmp.empty or tmp[group_col].nunique() < 2:
        return

    group_order = (
        tmp.groupby(group_col)[metric_col]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    values = [tmp.loc[tmp[group_col] == g, metric_col].values for g in group_order]

    plt.figure(figsize=(max(7, 1.25 * len(group_order)), 5))
    plt.boxplot(values, tick_labels=group_order, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric_col)
    plt.xlabel(group_col)
    plt.title(title or f"{metric_col} by {group_col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log(f"[FIG] {out_path}")


def plot_km(df, time_col, event_col, group_col, out_path, title=None):
    tmp = df[[time_col, event_col, group_col]].copy()
    tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce")
    tmp[event_col] = pd.to_numeric(tmp[event_col], errors="coerce")
    tmp[group_col] = tmp[group_col].map(normalize_text_value)
    tmp = tmp.dropna()
    tmp = tmp[tmp[time_col] > 0]

    if tmp[group_col].nunique() != 2:
        return

    p, chi2, used = simple_logrank(tmp[time_col], tmp[event_col], tmp[group_col])

    plt.figure(figsize=(7, 5))
    for g, sub in tmp.groupby(group_col):
        x, y = km_curve(sub[time_col], sub[event_col])
        if len(x) > 0:
            plt.step(x, y, where="post", label=f"{g} (n={len(sub)}, events={int(sub[event_col].sum())})")

    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.title(title or f"{time_col} by {group_col}\nlog-rank p={p:.3g}, D={d_from_p(p):.3g}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log(f"[FIG] {out_path}")


def plot_boot_distribution(df, col, observed, out_path, title=None):
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    if len(vals) == 0:
        return
    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=30, alpha=0.85)
    plt.axvline(observed, linestyle="--", linewidth=2)
    plt.xlabel(col)
    plt.ylabel("Bootstrap count")
    plt.title(title or f"Bootstrap distribution: {col}\nobserved={observed:.3g}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log(f"[FIG] {out_path}")


# =============================================================================
# 2. LOAD EXISTING BP-STATE AND RANDOM VALIDATION RESULTS
# =============================================================================

log("=" * 80)
log("Loading BRCA BP-state random-test folder")
log("=" * 80)

required_files = [
    PATIENT_METRICS_PATH,
    REAL_MODULE_STATS_PATH,
    RANDOM_ALL_COMPARE_PATH,
]
for p in required_files:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

patient_df = read_table_smart(PATIENT_METRICS_PATH)
module_df = read_table_smart(REAL_MODULE_STATS_PATH)

random_all_compare = read_table_smart(RANDOM_ALL_COMPARE_PATH)
random_bp_compare = read_table_smart(RANDOM_BP_MODULE_COMPARE_PATH) if RANDOM_BP_MODULE_COMPARE_PATH.exists() else pd.DataFrame()
stage_shuffle_compare = read_table_smart(STAGE_LABEL_SHUFFLE_COMPARE_PATH) if STAGE_LABEL_SHUFFLE_COMPARE_PATH.exists() else pd.DataFrame()
patient_scramble_compare = read_table_smart(PATIENT_SCRAMBLE_COMPARE_PATH) if PATIENT_SCRAMBLE_COMPARE_PATH.exists() else pd.DataFrame()

real_random_metrics = read_json_smart(REAL_RANDOM_METRICS_PATH)
random_summary = read_json_smart(RANDOM_SUMMARY_PATH)

write_csv(patient_df, TABLE_DIR / "00_input_BRCA_real_patient_centroid_metrics.csv")
write_csv(module_df, TABLE_DIR / "00_input_BRCA_real_module_stage_stats.csv")
write_csv(random_all_compare, TABLE_DIR / "00_input_BRCA_RANDOM_TEST_all_compare.csv")

if not random_bp_compare.empty:
    write_csv(random_bp_compare, TABLE_DIR / "00_input_random_BP_module_compare.csv")
if not stage_shuffle_compare.empty:
    write_csv(stage_shuffle_compare, TABLE_DIR / "00_input_stage_label_shuffle_compare.csv")
if not patient_scramble_compare.empty:
    write_csv(patient_scramble_compare, TABLE_DIR / "00_input_patient_scramble_compare.csv")

if real_random_metrics is not None:
    write_json(real_random_metrics, TABLE_DIR / "00_input_BRCA_real_random_test_metrics.json")
if random_summary is not None:
    write_json(random_summary, TABLE_DIR / "00_input_BRCA_RANDOM_TEST_summary.json")


# =============================================================================
# 3. NORMALIZE PATIENT METRICS
# =============================================================================

patient_col = infer_patient_column(patient_df)
patient_df["patient_id"] = patient_df[patient_col].map(normalize_tcga_patient_id)

stage_col = None
for c in ["stage_group", "StageGroup", "stage_binary", "early_late", "Stage_group"]:
    if c in patient_df.columns:
        stage_col = c
        break

if stage_col is None:
    stage_candidates = find_cols(patient_df, [r"stage", r"early", r"late"])
    if stage_candidates:
        stage_col = stage_candidates[0]

if stage_col is None:
    raise ValueError("Cannot find stage group column in BRCA_real_patient_centroid_metrics.csv")

patient_df["StageGroup"] = patient_df[stage_col].map(simplify_stage_group)

core_metric_cols = [
    c for c in [
        "Sim_late",
        "Dis_late",
        "Sim_early",
        "Dis_early",
        "late_minus_early_similarity",
        "late_minus_early_distance",
        "BP_abnormality_burden",
        "abnormality_burden",
        "module_abnormality_burden",
    ]
    if c in patient_df.columns
]

if "Sim_late" not in core_metric_cols:
    raise ValueError("Sim_late not found in patient centroid metrics.")

for c in core_metric_cols:
    patient_df[c] = pd.to_numeric(patient_df[c], errors="coerce")

write_csv(patient_df, TABLE_DIR / "01_patient_metrics_normalized.csv")


# =============================================================================
# 4. LOAD PHENOTYPE AND CLINICAL MATRIX
# =============================================================================

log("=" * 80)
log("Loading BRCA phenotype and clinical matrix")
log("=" * 80)

pheno_df = read_table_smart(PHENO_PATH)
clinical_df = read_table_smart(CLINICAL_PATH)

pheno_patient_col = infer_patient_column(pheno_df)
clinical_patient_col = infer_patient_column(clinical_df)

pheno_df["patient_id"] = pheno_df[pheno_patient_col].map(normalize_tcga_patient_id)
clinical_df["patient_id"] = clinical_df[clinical_patient_col].map(normalize_tcga_patient_id)

pheno_df = pheno_df.drop_duplicates("patient_id")
clinical_df = clinical_df.drop_duplicates("patient_id")

write_csv(pheno_df, TABLE_DIR / "02_Phenotype_normalized.csv")
write_csv(clinical_df, TABLE_DIR / "03_clinicalMatrix_normalized.csv")

merged = patient_df.merge(pheno_df, on="patient_id", how="left", suffixes=("", "_pheno"))
merged = merged.merge(clinical_df, on="patient_id", how="left", suffixes=("", "_clin"))

write_csv(merged, TABLE_DIR / "04_merged_BPstate_clinical.csv")


# =============================================================================
# 5. PRIMARY STAGE CHECK
# =============================================================================

log("=" * 80)
log("Primary stage check")
log("=" * 80)

stage_check_rows = []
for metric in core_metric_cols:
    w = welch_two_group(merged[metric], merged["StageGroup"], pos="Late", neg="Early")
    mw = mannwhitney_two_group(merged[metric], merged["StageGroup"], pos="Late", neg="Early")
    row = {"metric": metric}
    row.update(w)
    row.update(mw)
    stage_check_rows.append(row)

    plot_box_by_group(
        merged,
        metric,
        "StageGroup",
        FIG_DIR / f"primary_stage_{safe_filename(metric)}.png",
        title=f"Primary endpoint: {metric} by Early vs Late stage",
    )

stage_check_df = pd.DataFrame(stage_check_rows).sort_values("D_welch", ascending=False)
write_csv(stage_check_df, TABLE_DIR / "10_primary_stage_metric_check.csv")


# =============================================================================
# 6. BOOTSTRAP STABILITY
# =============================================================================

def run_stage_bootstrap(df, metric_col, B=200, frac=0.8, seed=20260531):
    rng = np.random.default_rng(seed)
    base = df[["patient_id", metric_col, "StageGroup"]].dropna().copy()
    base[metric_col] = pd.to_numeric(base[metric_col], errors="coerce")
    base = base.dropna()
    base = base[base["StageGroup"].isin(["Early", "Late"])]

    early = base[base["StageGroup"] == "Early"]
    late = base[base["StageGroup"] == "Late"]

    n_early = max(5, int(len(early) * frac))
    n_late = max(5, int(len(late) * frac))

    rows = []
    for b in range(B):
        e_idx = rng.choice(early.index.values, size=n_early, replace=True)
        l_idx = rng.choice(late.index.values, size=n_late, replace=True)
        boot = pd.concat([early.loc[e_idx], late.loc[l_idx]], axis=0)

        w = welch_two_group(boot[metric_col], boot["StageGroup"], pos="Late", neg="Early")
        mw = mannwhitney_two_group(boot[metric_col], boot["StageGroup"], pos="Late", neg="Early")

        row = {
            "bootstrap_iter": b + 1,
            "metric": metric_col,
            "n_early_draw": int(n_early),
            "n_late_draw": int(n_late),
        }
        row.update(w)
        row.update(mw)
        rows.append(row)

    return pd.DataFrame(rows)


log("=" * 80)
log("Bootstrap stability")
log("=" * 80)

boot_summary_rows = []
for metric in core_metric_cols:
    boot = run_stage_bootstrap(merged, metric_col=metric, B=200, frac=0.8, seed=20260531)
    write_csv(boot, TABLE_DIR / f"20_bootstrap_stage_{safe_filename(metric)}.csv")

    obs = welch_two_group(merged[metric], merged["StageGroup"], pos="Late", neg="Early")

    row = {
        "metric": metric,
        "B": int(len(boot)),
        "observed_delta_late_minus_early": obs["delta_pos_minus_neg"],
        "observed_p_welch": obs["p_welch"],
        "observed_D_welch": obs["D_welch"],
        "bootstrap_median_delta": float(np.nanmedian(boot["delta_pos_minus_neg"])),
        "bootstrap_q05_delta": float(np.nanquantile(boot["delta_pos_minus_neg"], 0.05)),
        "bootstrap_q95_delta": float(np.nanquantile(boot["delta_pos_minus_neg"], 0.95)),
        "bootstrap_median_D_welch": float(np.nanmedian(boot["D_welch"])),
        "bootstrap_q05_D_welch": float(np.nanquantile(boot["D_welch"], 0.05)),
        "bootstrap_q95_D_welch": float(np.nanquantile(boot["D_welch"], 0.95)),
        "fraction_delta_positive": float(np.nanmean(boot["delta_pos_minus_neg"] > 0)),
        "fraction_delta_negative": float(np.nanmean(boot["delta_pos_minus_neg"] < 0)),
        "fraction_D_gt_1_301": float(np.nanmean(boot["D_welch"] > 1.301)),
        "fraction_p_lt_0_05": float(np.nanmean(boot["p_welch"] < 0.05)),
    }

    boot_summary_rows.append(row)

    plot_boot_distribution(
        boot,
        "D_welch",
        obs["D_welch"],
        FIG_DIR / f"bootstrap_D_stage_{safe_filename(metric)}.png",
        title=f"Bootstrap D: {metric}",
    )
    plot_boot_distribution(
        boot,
        "delta_pos_minus_neg",
        obs["delta_pos_minus_neg"],
        FIG_DIR / f"bootstrap_delta_stage_{safe_filename(metric)}.png",
        title=f"Bootstrap delta Late-Early: {metric}",
    )

boot_summary_df = pd.DataFrame(boot_summary_rows).sort_values("observed_D_welch", ascending=False)
write_csv(boot_summary_df, TABLE_DIR / "21_bootstrap_stage_summary.csv")


# =============================================================================
# 7. PAM50 / MOLECULAR SUBTYPE ENDPOINT
# =============================================================================

log("=" * 80)
log("PAM50 / molecular subtype endpoint")
log("=" * 80)

pam50_candidates = find_cols(merged, [r"PAM50", r"pam50", r"subtype", r"Subtype"])
pam50_col = None

for c in pam50_candidates:
    cs = str(c)
    if re.search(r"PAM50Call_RNAseq|PAM50.*RNAseq|RNAseq.*PAM50", cs, flags=re.I):
        pam50_col = c
        break

if pam50_col is None and pam50_candidates:
    pam50_col = pam50_candidates[0]

categorical_results = []
categorical_group_summaries = []

if pam50_col is not None:
    log(f"[PAM50] selected column: {pam50_col}")
    merged["PAM50_simplified"] = merged[pam50_col].map(simplify_pam50)

    for metric in core_metric_cols:
        result, group_summary = kruskal_endpoint(
            merged,
            metric_col=metric,
            group_col="PAM50_simplified",
            min_group_n=10,
        )
        if result is not None:
            result["endpoint_label"] = "PAM50"
            result["raw_column"] = str(pam50_col)
            categorical_results.append(result)

            group_summary.insert(0, "endpoint_label", "PAM50")
            group_summary.insert(1, "raw_column", str(pam50_col))
            group_summary.insert(2, "metric", metric)
            categorical_group_summaries.append(group_summary)

            plot_box_by_group(
                merged,
                metric,
                "PAM50_simplified",
                FIG_DIR / f"PAM50_{safe_filename(metric)}.png",
                title=f"{metric} by PAM50 subtype",
            )
else:
    log("[WARN] No PAM50/subtype column found.")


# =============================================================================
# 8. SURVIVAL ENDPOINT TESTS: OS / DSS / PFI / DFI
# =============================================================================

def find_survival_cols(df, endpoint):
    ep = endpoint.upper()
    event_col = None
    time_col = None

    for c in df.columns:
        cu = str(c).upper()
        if cu == ep:
            event_col = c
        if cu in {f"{ep}.TIME", f"{ep}_TIME", f"{ep}TIME"}:
            time_col = c

    if event_col is None:
        event_col = find_first_col(df, [rf"^{ep}$", rf"\b{ep}\b"])
    if time_col is None:
        time_col = find_first_col(df, [rf"{ep}.*time", rf"{ep}.*days", rf"time.*{ep}"])

    return event_col, time_col


log("=" * 80)
log("Survival endpoint tests")
log("=" * 80)

survival_rows = []
for endpoint in ["OS", "DSS", "PFI", "DFI"]:
    event_col, time_col = find_survival_cols(merged, endpoint)
    if event_col is None or time_col is None:
        log(f"[WARN] Missing survival cols for {endpoint}: event={event_col}, time={time_col}")
        continue

    log(f"[SURVIVAL] {endpoint}: event={event_col}, time={time_col}")

    for metric in core_metric_cols:
        tmp = merged[["patient_id", metric, event_col, time_col]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp[event_col] = pd.to_numeric(tmp[event_col], errors="coerce")
        tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce")
        tmp = tmp.dropna()
        tmp = tmp[tmp[time_col] > 0]

        if tmp.shape[0] < 50 or tmp[event_col].sum() < 10:
            continue

        cutoff = tmp[metric].median()
        tmp[f"{metric}_median_group"] = np.where(tmp[metric] >= cutoff, "High", "Low")

        p, chi2, used = simple_logrank(
            tmp[time_col],
            tmp[event_col],
            tmp[f"{metric}_median_group"],
        )

        high = tmp[tmp[f"{metric}_median_group"] == "High"]
        low = tmp[tmp[f"{metric}_median_group"] == "Low"]

        row = {
            "endpoint": endpoint,
            "metric": metric,
            "event_col": event_col,
            "time_col": time_col,
            "n_total": int(tmp.shape[0]),
            "n_events": int(tmp[event_col].sum()),
            "median_cutoff": float(cutoff),
            "n_high": int(high.shape[0]),
            "events_high": int(high[event_col].sum()),
            "event_fraction_high": float(high[event_col].mean()),
            "n_low": int(low.shape[0]),
            "events_low": int(low[event_col].sum()),
            "event_fraction_low": float(low[event_col].mean()),
            "high_minus_low_event_fraction": float(high[event_col].mean() - low[event_col].mean()),
            "logrank_chi2": chi2,
            "p_logrank": p,
            "D_logrank": d_from_p(p),
        }
        survival_rows.append(row)

        write_csv(
            tmp,
            TABLE_DIR / f"30_survival_input_{endpoint}_{safe_filename(metric)}.csv",
        )

        plot_km(
            tmp,
            time_col,
            event_col,
            f"{metric}_median_group",
            FIG_DIR / f"KM_{endpoint}_{safe_filename(metric)}.png",
            title=f"{endpoint}: {metric} high vs low\nlog-rank p={p:.3g}, D={d_from_p(p):.3g}",
        )

survival_df = pd.DataFrame(survival_rows)
if not survival_df.empty:
    survival_df = survival_df.sort_values(["D_logrank", "endpoint"], ascending=[False, True])
write_csv(survival_df, TABLE_DIR / "31_survival_endpoint_tests.csv")


# =============================================================================
# 9. SUPPLEMENTARY CLINICAL ENDPOINT SCAN
# =============================================================================

log("=" * 80)
log("Supplementary clinical endpoint scan")
log("=" * 80)

candidate_patterns = {
    "ER_status": [r"\bER\b", r"estrogen"],
    "PR_status": [r"\bPR\b", r"progesterone"],
    "HER2_status": [r"HER2", r"HER-2", r"ERBB2"],
    "Node_status": [r"node", r"lymph.*node"],
    "T_stage": [r"pathologic.*T", r"ajcc.*T", r"\bT.stage\b", r"\btumor.*stage"],
    "N_stage": [r"pathologic.*N", r"ajcc.*N", r"\bN.stage\b"],
    "M_stage": [r"pathologic.*M", r"ajcc.*M", r"\bM.stage\b"],
    "new_tumor_event": [r"new.*tumor", r"tumor.*event", r"recurrence"],
    "person_neoplasm_status": [r"person.*neoplasm.*status", r"cancer.*status", r"tumor.*status"],
    "histological_type": [r"histolog"],
    "grade": [r"grade"],
}

detected_cols = {}
for label, pats in candidate_patterns.items():
    cols = find_cols(merged, pats)

    filtered = []
    for c in cols:
        cs = str(c).lower()
        if cs in {"os", "dss", "pfi", "dfi"}:
            continue
        if "time" in cs or "days" in cs:
            continue
        if c in core_metric_cols:
            continue
        filtered.append(c)

    if filtered:
        detected_cols[label] = filtered[:8]

write_json(
    {k: [str(x) for x in v] for k, v in detected_cols.items()},
    TABLE_DIR / "40_detected_supplementary_endpoint_columns.json",
)

def clean_endpoint_series(label, series):
    if label in {"ER_status", "PR_status", "HER2_status"}:
        return series.map(clean_binary_status)
    if label == "Node_status":
        return series.map(simplify_node_status)
    if label == "T_stage":
        return series.map(lambda x: simplify_tnm_stage(x, "T"))
    if label == "N_stage":
        return series.map(lambda x: simplify_tnm_stage(x, "N"))
    if label == "M_stage":
        return series.map(lambda x: simplify_tnm_stage(x, "M"))
    return series.map(normalize_text_value)


supp_rows = []
supp_group_summaries = []

for label, cols in detected_cols.items():
    for raw_col in cols:
        clean_col = f"{label}__{safe_filename(raw_col)}__clean"
        merged[clean_col] = clean_endpoint_series(label, merged[raw_col])

        n_unique = merged[clean_col].dropna().nunique()
        n_nonnull = merged[clean_col].notna().sum()

        if n_nonnull < 50:
            continue
        if n_unique < 2 or n_unique > 15:
            continue

        for metric in core_metric_cols:
            result, group_summary = kruskal_endpoint(
                merged,
                metric_col=metric,
                group_col=clean_col,
                min_group_n=10,
            )

            if result is None:
                continue

            result["endpoint_label"] = label
            result["raw_column"] = str(raw_col)
            result["clean_column"] = clean_col
            supp_rows.append(result)

            group_summary.insert(0, "endpoint_label", label)
            group_summary.insert(1, "raw_column", str(raw_col))
            group_summary.insert(2, "clean_column", clean_col)
            group_summary.insert(3, "metric", metric)
            supp_group_summaries.append(group_summary)

            if result["D_kruskal"] >= 1.0:
                plot_box_by_group(
                    merged,
                    metric,
                    clean_col,
                    FIG_DIR / f"SUPP_{safe_filename(label)}_{safe_filename(raw_col)}_{safe_filename(metric)}.png",
                    title=f"{metric} by {label}: {raw_col}",
                )

supp_df = pd.DataFrame(supp_rows)
if not supp_df.empty:
    supp_df = supp_df.sort_values("D_kruskal", ascending=False)

write_csv(supp_df, TABLE_DIR / "41_supplementary_clinical_endpoint_scan.csv")

if supp_group_summaries:
    supp_group_summary_df = pd.concat(supp_group_summaries, axis=0, ignore_index=True)
else:
    supp_group_summary_df = pd.DataFrame()
write_csv(supp_group_summary_df, TABLE_DIR / "42_supplementary_clinical_group_summaries.csv")


# =============================================================================
# 10. CATEGORICAL ENDPOINT OUTPUTS
# =============================================================================

categorical_df = pd.DataFrame(categorical_results)
if not categorical_df.empty:
    categorical_df = categorical_df.sort_values("D_kruskal", ascending=False)

write_csv(categorical_df, TABLE_DIR / "50_secondary_PAM50_endpoint_tests.csv")

if categorical_group_summaries:
    categorical_group_summary_df = pd.concat(categorical_group_summaries, axis=0, ignore_index=True)
else:
    categorical_group_summary_df = pd.DataFrame()

write_csv(categorical_group_summary_df, TABLE_DIR / "51_secondary_PAM50_group_summaries.csv")


# =============================================================================
# 11. INTEGRATED EVIDENCE TABLE
# =============================================================================

integrated_rows = []

for _, r in stage_check_df.iterrows():
    integrated_rows.append({
        "analysis_layer": "Primary endpoint",
        "test_type": "Stage Early vs Late",
        "endpoint": "StageGroup",
        "metric": r["metric"],
        "n": int(r["n_pos"] + r["n_neg"]) if not pd.isna(r["n_pos"]) else np.nan,
        "p_value": r["p_welch"],
        "D": r["D_welch"],
        "effect_summary": f"Late-Early delta={r['delta_pos_minus_neg']:.4g}",
        "interpretation": "Primary BP-state stage separation",
    })

for _, r in boot_summary_df.iterrows():
    integrated_rows.append({
        "analysis_layer": "Stability",
        "test_type": "Patient bootstrap",
        "endpoint": "StageGroup",
        "metric": r["metric"],
        "n": r["B"],
        "p_value": np.nan,
        "D": r["bootstrap_median_D_welch"],
        "effect_summary": (
            f"median D={r['bootstrap_median_D_welch']:.4g}; "
            f"q05-q95={r['bootstrap_q05_D_welch']:.4g}-{r['bootstrap_q95_D_welch']:.4g}; "
            f"frac D>1.301={r['fraction_D_gt_1_301']:.3f}"
        ),
        "interpretation": "Bootstrap stability of stage signal",
    })

if not categorical_df.empty:
    for _, r in categorical_df.iterrows():
        integrated_rows.append({
            "analysis_layer": "Secondary endpoint",
            "test_type": "Kruskal-Wallis",
            "endpoint": "PAM50",
            "metric": r["metric"],
            "n": r["n_total"],
            "p_value": r["p_kruskal"],
            "D": r["D_kruskal"],
            "effect_summary": f"groups={r['n_groups']}; kept={r['groups_kept']}",
            "interpretation": "Molecular subtype association",
        })

if not survival_df.empty:
    for _, r in survival_df.iterrows():
        integrated_rows.append({
            "analysis_layer": "Secondary endpoint",
            "test_type": "Log-rank median split",
            "endpoint": r["endpoint"],
            "metric": r["metric"],
            "n": r["n_total"],
            "p_value": r["p_logrank"],
            "D": r["D_logrank"],
            "effect_summary": (
                f"events={r['n_events']}; "
                f"high event frac={r['event_fraction_high']:.3f}; "
                f"low event frac={r['event_fraction_low']:.3f}"
            ),
            "interpretation": "Prognosis-oriented secondary test",
        })

if not supp_df.empty:
    for _, r in supp_df.head(60).iterrows():
        integrated_rows.append({
            "analysis_layer": "Supplementary endpoint",
            "test_type": "Kruskal-Wallis",
            "endpoint": r["endpoint_label"],
            "metric": r["metric"],
            "n": r["n_total"],
            "p_value": r["p_kruskal"],
            "D": r["D_kruskal"],
            "effect_summary": f"raw_column={r['raw_column']}; groups={r['n_groups']}",
            "interpretation": "Supplementary clinical consistency scan",
        })

integrated_df = pd.DataFrame(integrated_rows)
if not integrated_df.empty:
    integrated_df = integrated_df.sort_values(["analysis_layer", "D"], ascending=[True, False])

write_csv(integrated_df, TABLE_DIR / "60_integrated_evidence_table.csv")


# =============================================================================
# 12. FINAL MERGED OUTPUT
# =============================================================================

write_csv(merged, TABLE_DIR / "99_final_merged_BPstate_clinical_endpoints.csv")


# =============================================================================
# 13. SUMMARY README
# =============================================================================

summary_lines = []
summary_lines.append("MOATTERS BRCA BP-state downstream validation")
summary_lines.append("=" * 90)
summary_lines.append("")
summary_lines.append(f"Output directory: {OUT_DIR}")
summary_lines.append(f"Random-test source folder: {RANDOM_TEST_DIR}")
summary_lines.append(f"Patient metrics: {PATIENT_METRICS_PATH}")
summary_lines.append(f"Phenotype: {PHENO_PATH}")
summary_lines.append(f"Clinical matrix: {CLINICAL_PATH}")
summary_lines.append("")
summary_lines.append(f"Patients in BP-state table: {patient_df['patient_id'].nunique()}")
summary_lines.append(f"Patients after clinical merge: {merged['patient_id'].nunique()}")
summary_lines.append(f"Core BP-state metrics tested: {', '.join(core_metric_cols)}")
summary_lines.append("")

summary_lines.append("1. Existing random-control evidence")
summary_lines.append("-" * 90)
if not random_all_compare.empty:
    keep_cols = [c for c in random_all_compare.columns if c.lower() in {
        "control_type", "metric", "real_value", "random_mean", "null_mean",
        "random_p95", "random_p99", "empirical_p", "z_score"
    }]
    if not keep_cols:
        keep_cols = random_all_compare.columns.tolist()[:8]

    for _, r in random_all_compare.head(30).iterrows():
        metric = r.get("metric", r.get("Metric", "NA"))
        control = r.get("control_type", r.get("control", "NA"))
        emp = r.get(
            "empirical_p",
            r.get(
                "emp_p",
                r.get(
                    "empirical_p_greater_equal",
                    r.get("empirical_p_less_equal", np.nan),
                ),
            ),
        )
        rv = r.get("real_value", r.get("real", np.nan))
        summary_lines.append(
            f"{control} | {metric}: real={rv}, empirical_p={emp}"
        )
else:
    summary_lines.append("No BRCA_RANDOM_TEST_all_compare.csv loaded.")
summary_lines.append("")

summary_lines.append("2. Primary Stage endpoint")
summary_lines.append("-" * 90)
for _, r in stage_check_df.head(10).iterrows():
    summary_lines.append(
        f"{r['metric']}: D_welch={r['D_welch']:.4g}, "
        f"p={r['p_welch']:.4g}, "
        f"Late-Early delta={r['delta_pos_minus_neg']:.4g}, "
        f"nLate={int(r['n_pos'])}, nEarly={int(r['n_neg'])}"
    )
summary_lines.append("")

summary_lines.append("3. Bootstrap stability")
summary_lines.append("-" * 90)
for _, r in boot_summary_df.head(10).iterrows():
    summary_lines.append(
        f"{r['metric']}: observed D={r['observed_D_welch']:.4g}; "
        f"bootstrap median D={r['bootstrap_median_D_welch']:.4g}; "
        f"5-95% D={r['bootstrap_q05_D_welch']:.4g}-{r['bootstrap_q95_D_welch']:.4g}; "
        f"frac D>1.301={r['fraction_D_gt_1_301']:.3f}; "
        f"frac p<0.05={r['fraction_p_lt_0_05']:.3f}"
    )
summary_lines.append("")

summary_lines.append("4. PAM50 molecular subtype endpoint")
summary_lines.append("-" * 90)
if pam50_col is not None:
    summary_lines.append(f"PAM50 column used: {pam50_col}")
if not categorical_df.empty:
    for _, r in categorical_df.head(10).iterrows():
        summary_lines.append(
            f"{r['metric']}: n={int(r['n_total'])}, groups={int(r['n_groups'])}, "
            f"p={r['p_kruskal']:.4g}, D={r['D_kruskal']:.4g}, "
            f"groups={r['groups_kept']}"
        )
else:
    summary_lines.append("No PAM50 result.")
summary_lines.append("")

summary_lines.append("5. Survival endpoints")
summary_lines.append("-" * 90)
if not survival_df.empty:
    for _, r in survival_df.head(20).iterrows():
        summary_lines.append(
            f"{r['endpoint']} | {r['metric']}: "
            f"n={int(r['n_total'])}, events={int(r['n_events'])}, "
            f"p={r['p_logrank']:.4g}, D={r['D_logrank']:.4g}, "
            f"event_high={r['event_fraction_high']:.3f}, "
            f"event_low={r['event_fraction_low']:.3f}"
        )
else:
    summary_lines.append("No survival result.")
summary_lines.append("")

summary_lines.append("6. Supplementary endpoint scan")
summary_lines.append("-" * 90)
if not supp_df.empty:
    for _, r in supp_df.head(25).iterrows():
        summary_lines.append(
            f"{r['endpoint_label']} | {r['raw_column']} | {r['metric']}: "
            f"n={int(r['n_total'])}, groups={int(r['n_groups'])}, "
            f"p={r['p_kruskal']:.4g}, D={r['D_kruskal']:.4g}"
        )
else:
    summary_lines.append("No supplementary endpoint result.")
summary_lines.append("")

summary_lines.append("Suggested manuscript interpretation")
summary_lines.append("-" * 90)
summary_lines.append(
    "The downstream BP-state reconstruction should be framed as an application of "
    "observation-ready and task-discriminative BP/pathway observables, not merely as "
    "a pathway-ranking exercise."
)
summary_lines.append(
    "Primary evidence: BP-state metrics separate early vs late BRCA; random-control qualification should be reported only where an empirical null probability is available."
)
summary_lines.append(
    "Bootstrap evidence: tests whether the stage-associated BP-state signal is stable under patient resampling."
)
summary_lines.append(
    "PAM50 evidence: tests whether the BP-state space captures intrinsic molecular subtype heterogeneity."
)
summary_lines.append(
    "Survival evidence: use OS/DSS/PFI/DFI as secondary prognosis-oriented support, not as the primary claim unless strong."
)
summary_lines.append(
    "Distance metrics such as Dis_late should remain supplementary if similarity/alignment metrics dominate."
)

summary_text = "\n".join(summary_lines)

with open(OUT_DIR / "SUMMARY_README.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

log("")
log(summary_text)
log("")
log("[DONE]")
log(f"All outputs written to: {OUT_DIR}")
