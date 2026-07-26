# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# ============================================================
# MOATTERS BRCA Patient-level BP Strategy Profile V1
# "��������뷨��"
#
# Purpose:
#   For BRCA only, build patient-level BP/module strategy profiles.
#
# Input:
#   D:\MOATTERS-Data\UCSC_XENA\Breast Cancer (BRCA)\GE.tsv
#   D:\MOATTERS-Data\GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt
#   D:\MOATTERS-Data\GSEA\h.all.v2026.1.Hs.symbols.gmt
#   D:\MOATTERS-Output\MOATTERS_STAGE_BP_Module_Rewiring_V1_20260531\BRCA
#   D:\MOATTERS-Output\MOATTERS_TME_Conditioned_BP_Modules_V1_20260531\BRCA
#
# Output:
#   D:\MOATTERS-Output\MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531
#
# What it produces:
#   1. BRCA_patient_strategy_master_table.csv
#   2. BRCA_patient_module_scores.csv
#   3. BRCA_patient_strategy_scores.csv
#   4. BRCA_patient_centroid_similarity_distance.csv
#   5. BRCA_patient_strategy_class_summary.csv
#   6. Figures:
#        - patient module heatmap
#        - PCA map
#        - strategy-score heatmap
#        - example patient dashboards
# ============================================================

import re
import math
import json
from pathlib import Path
from moatters.config import data_path, output_path
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================================
# 0. PATH CONFIG
# ============================================================

DATA_ROOT = data_path()
TEMP_ROOT = output_path()

BRCA_GE_PATH = DATA_ROOT / r"UCSC_XENA\Breast Cancer (BRCA)\GE.tsv"
GO_BP_GMT = DATA_ROOT / r"GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt"
HALLMARK_GMT = DATA_ROOT / r"GSEA\h.all.v2026.1.Hs.symbols.gmt"

MODULE_DIR = TEMP_ROOT / r"MOATTERS_STAGE_BP_Module_Rewiring_V1_20260531\BRCA"
TME_DIR = TEMP_ROOT / r"MOATTERS_TME_Conditioned_BP_Modules_V1_20260531\BRCA"
V3_DIR = TEMP_ROOT / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA"

OUT_DIR = TEMP_ROOT / "MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Output:", OUT_DIR)


# ============================================================
# 1. USER CONFIG
# ============================================================

CANCER_CODE = "BRCA"

MIN_MATCHED_GENES = 10

# Which module group to use as patient state basis.
# BRCA late stage had more fragmented modules, so use stage_III_IV modules
# as a fun "late strategy map".
MODULE_GROUP_FOR_PROFILE = "stage_III_IV"

# Also compute early module profile for comparison.
OTHER_GROUPS_TO_SCORE = ["stage_I_II", "stage_III_IV"]

# Patient class thresholds
HIGH_Z = 0.75
VERY_HIGH_Z = 1.25
LATE_SIM_HIGH = 0.35
BURDEN_HIGH = 0.50

# Hallmark proxies for strategy dimensions
HALLMARK_STRATEGY_TERMS = {
    "immune_IFN_gamma": "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "immune_IFN_alpha": "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "inflammatory": "HALLMARK_INFLAMMATORY_RESPONSE",
    "TNFA_NFKB": "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "EMT_stromal": "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "angiogenesis": "HALLMARK_ANGIOGENESIS",
    "hypoxia": "HALLMARK_HYPOXIA",
    "glycolysis": "HALLMARK_GLYCOLYSIS",
    "oxidative_phosphorylation": "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "fatty_acid_metabolism": "HALLMARK_FATTY_ACID_METABOLISM",
    "proliferation_E2F": "HALLMARK_E2F_TARGETS",
    "proliferation_G2M": "HALLMARK_G2M_CHECKPOINT",
    "MYC": "HALLMARK_MYC_TARGETS_V1",
    "p53": "HALLMARK_P53_PATHWAY",
    "apoptosis": "HALLMARK_APOPTOSIS",
    "DNA_repair": "HALLMARK_DNA_REPAIR",
    "mTORC1": "HALLMARK_MTORC1_SIGNALING",
}

STRATEGY_CATEGORIES = {
    "immune_inflammatory": [
        "immune_IFN_gamma", "immune_IFN_alpha", "inflammatory", "TNFA_NFKB"
    ],
    "stromal_EMT_hypoxia": [
        "EMT_stromal", "angiogenesis", "hypoxia"
    ],
    "metabolic": [
        "glycolysis", "oxidative_phosphorylation", "fatty_acid_metabolism"
    ],
    "proliferation": [
        "proliferation_E2F", "proliferation_G2M", "MYC", "mTORC1"
    ],
    "stress_damage": [
        "p53", "apoptosis", "DNA_repair"
    ],
}


# ============================================================
# 2. BASIC HELPERS
# ============================================================

def tcga_patient_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s, flags=re.I)
    if m:
        return m.group(1).upper()
    return s.upper()


def clean_gene_symbol(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.split("|")[0].strip()
    return s.upper()


def detect_encoding_by_bom(path):
    with open(path, "rb") as f:
        raw = f.read(4)
    if raw.startswith(b"\xff\xfe"):
        return "utf-16"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return None


def read_table_safely(path, nrows=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    bom = detect_encoding_by_bom(path)
    encodings = []
    if bom:
        encodings.append(bom)
    encodings += ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "latin1"]
    encodings = list(dict.fromkeys(encodings))

    for enc in encodings:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc, low_memory=False, nrows=nrows)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
        try:
            df = pd.read_csv(path, sep=",", encoding=enc, low_memory=False, nrows=nrows)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc, nrows=nrows)
            return df
        except Exception:
            pass

    raise RuntimeError(f"Could not read table: {path}")


def looks_like_tcga_colnames(cols, min_frac=0.2):
    cols = list(map(str, cols))
    if len(cols) == 0:
        return False
    hits = sum(
        1 for c in cols
        if re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", c, flags=re.I)
    )
    return hits / max(len(cols), 1) >= min_frac


def looks_like_tcga_values(series, min_frac=0.2):
    vals = series.dropna().astype(str).head(300).tolist()
    if len(vals) == 0:
        return False
    hits = sum(
        1 for v in vals
        if re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", v, flags=re.I)
    )
    return hits / max(len(vals), 1) >= min_frac


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def zscore_series(s):
    s = pd.Series(s).astype(float)
    sd = s.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd


def vector_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def vector_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    r, _ = spearmanr(a[ok], b[ok])
    return float(r) if not pd.isna(r) else np.nan


# ============================================================
# 3. LOAD GE AND GMT
# ============================================================

def load_gmt(gmt_path):
    gene_sets = {}
    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = [clean_gene_symbol(g) for g in parts[2:] if clean_gene_symbol(g)]
            genes = sorted(set(genes))
            if genes:
                gene_sets[term] = genes
    return gene_sets


def infer_gene_column(df):
    possible = [
        "gene", "genes", "symbol", "gene_symbol",
        "hugo_symbol", "Hugo_Symbol", "Gene", "GENE",
        "sample"
    ]
    cols_lower = {str(c).lower(): c for c in df.columns}
    for p in possible:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    return df.columns[0]


def load_ge_matrix(ge_path):
    df = read_table_safely(ge_path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    if looks_like_tcga_colnames(df.columns):
        gene_col = infer_gene_column(df)
        genes = df[gene_col].map(clean_gene_symbol)

        mat = df.drop(columns=[gene_col])
        mat.columns = [tcga_patient_id(c) for c in mat.columns]
        mat.index = genes
        mat = mat[mat.index.notna()]
        mat = mat.apply(pd.to_numeric, errors="coerce")
        mat = mat.T.groupby(level=0).mean().T
        mat = mat.groupby(mat.index).mean()
        return mat

    first_col = df.columns[0]
    if looks_like_tcga_values(df[first_col]):
        samples = df[first_col].map(tcga_patient_id)

        mat = df.drop(columns=[first_col])
        mat.index = samples
        mat.columns = [clean_gene_symbol(c) for c in mat.columns]
        mat = mat.loc[mat.index.notna(), :]
        mat = mat.loc[:, pd.notna(mat.columns)]
        mat = mat.apply(pd.to_numeric, errors="coerce")
        mat = mat.groupby(mat.index).mean()
        mat = mat.T
        mat = mat.groupby(mat.index).mean()
        return mat

    gene_col = infer_gene_column(df)
    genes = df[gene_col].map(clean_gene_symbol)

    mat = df.drop(columns=[gene_col])
    mat.columns = [tcga_patient_id(c) for c in mat.columns]
    mat.index = genes
    mat = mat[mat.index.notna()]
    mat = mat.apply(pd.to_numeric, errors="coerce")
    mat = mat.T.groupby(level=0).mean().T
    mat = mat.groupby(mat.index).mean()
    return mat


def zscore_rows(mat):
    mean = mat.mean(axis=1, skipna=True)
    std = mat.std(axis=1, skipna=True).replace(0, np.nan)
    return mat.sub(mean, axis=0).div(std, axis=0)


def compute_gene_set_scores(ge, gene_sets, terms, min_matched=10):
    ge_genes = set(ge.index.astype(str))
    zge = zscore_rows(ge)

    rows = []
    scores = {}

    for term in terms:
        if term not in gene_sets:
            rows.append({
                "term": term,
                "n_defined": 0,
                "n_matched": 0,
                "matched_fraction": 0,
                "readiness_class": "missing_from_gmt"
            })
            continue

        defined = sorted(set(gene_sets[term]))
        matched = [g for g in defined if g in ge_genes]

        n_defined = len(defined)
        n_matched = len(matched)
        frac = n_matched / n_defined if n_defined > 0 else 0

        if n_matched >= min_matched:
            scores[term] = zge.loc[matched].mean(axis=0, skipna=True)
            cls = "observation_ready"
        elif n_matched > 0:
            cls = "low_resolution"
        else:
            cls = "near_unobservable"

        rows.append({
            "term": term,
            "n_defined": n_defined,
            "n_matched": n_matched,
            "matched_fraction": frac,
            "readiness_class": cls
        })

    score_df = pd.DataFrame(scores).T
    readiness_df = pd.DataFrame(rows)

    return score_df, readiness_df


# ============================================================
# 4. LOAD STAGE
# ============================================================

def roman_stage_to_group(x):
    if pd.isna(x):
        return None
    s0 = str(x).strip()
    if not s0:
        return None

    s = s0.upper()
    bad = [
        "", "NAN", "NA", "N/A", "NONE", "UNKNOWN",
        "NOT REPORTED", "NOT AVAILABLE", "[NOT AVAILABLE]",
        "[NOT EVALUATED]", "NOT EVALUATED", "TX", "TIS",
        "STAGE X", "X"
    ]
    if s in bad:
        return None

    sl = s.lower().strip()
    if sl in ["early", "stage_i_ii", "i_ii", "stage i/ii", "stage i-ii", "i/ii"]:
        return "Early"
    if sl in ["late", "stage_iii_iv", "iii_iv", "stage iii/iv", "stage iii-iv", "iii/iv"]:
        return "Late"

    s = s.replace("PATHOLOGIC", " ")
    s = s.replace("CLINICAL", " ")
    s = s.replace("STAGE", " ")
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if re.search(r"\bIV\b|IVA|IVB|IVC", s):
        return "Late"
    if re.search(r"\bIII\b|IIIA|IIIB|IIIC", s):
        return "Late"
    if re.search(r"\bII\b|IIA|IIB|IIC", s):
        return "Early"
    if re.search(r"\bI\b|IA|IB|IC", s):
        return "Early"

    if re.search(r"\b4\b", s):
        return "Late"
    if re.search(r"\b3\b", s):
        return "Late"
    if re.search(r"\b2\b", s):
        return "Early"
    if re.search(r"\b1\b", s):
        return "Early"

    return None


def infer_patient_col(df):
    possible = [
        "patient", "Patient", "PATIENT", "_PATIENT",
        "sample", "Sample", "SAMPLE",
        "sampleID", "SampleID",
        "barcode", "submitter_id", "case_id",
        "bcr_patient_barcode", "patient_id"
    ]
    cols_lower = {str(c).lower(): c for c in df.columns}
    for p in possible:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    for c in df.columns:
        try:
            if looks_like_tcga_values(df[c], min_frac=0.1):
                return c
        except Exception:
            pass
    return df.columns[0]


def infer_stage_cols(df):
    priority = [
        "stage_group",
        "stage_raw",
        "pathologic_stage",
        "ajcc_pathologic_stage",
        "clinical_stage",
        "tumor_stage",
        "pathologic_tumor_stage",
        "clinical_tumor_stage",
        "Converted_Stage_nature2012",
        "AJCC_Stage_nature2012",
        "stage"
    ]
    cols = list(df.columns)
    cols_lower = {str(c).lower(): c for c in cols}
    ordered = []

    for p in priority:
        if p.lower() in cols_lower and cols_lower[p.lower()] not in ordered:
            ordered.append(cols_lower[p.lower()])

    for c in cols:
        if "stage" in str(c).lower() and c not in ordered:
            ordered.append(c)

    return ordered


def parse_stage_file(stage_file):
    try:
        df = read_table_safely(stage_file)
    except Exception:
        return pd.DataFrame(columns=["patient", "stage_group"])

    if df.empty or df.shape[1] < 2:
        return pd.DataFrame(columns=["patient", "stage_group"])

    patient_col = infer_patient_col(df)
    stage_cols = infer_stage_cols(df)

    best = pd.DataFrame(columns=["patient", "stage_group"])

    for stage_col in stage_cols:
        out = pd.DataFrame()
        out["patient"] = df[patient_col].map(tcga_patient_id)
        out["stage_group"] = df[stage_col].map(roman_stage_to_group)
        out = out.dropna(subset=["patient", "stage_group"])
        out = out.drop_duplicates(subset=["patient"])

        counts = out["stage_group"].value_counts().to_dict()
        if set(counts.keys()) >= {"Early", "Late"}:
            print("[Stage] using", stage_file.name, counts)
            return out

        if out.shape[0] > best.shape[0]:
            best = out

    return best


def load_brca_stage():
    cancer_dir = DATA_ROOT / r"UCSC_XENA\Breast Cancer (BRCA)"
    files = []
    files += list(cancer_dir.glob("*stage_groups_from_survival*.tsv"))
    files += list(cancer_dir.glob("Phenotype.tsv"))
    files += list(cancer_dir.glob("*clinicalMatrix*"))
    files += list(cancer_dir.glob("*clinical*.tsv"))
    files += list(cancer_dir.glob("*phenotype*.tsv"))

    for f in files:
        st = parse_stage_file(f)
        counts = st["stage_group"].value_counts().to_dict() if not st.empty else {}
        if set(counts.keys()) >= {"Early", "Late"}:
            return st

    return pd.DataFrame(columns=["patient", "stage_group"])


# ============================================================
# 5. LOAD MODULE RESULTS
# ============================================================

def load_module_files():
    assignment_path = MODULE_DIR / "BRCA_module_assignment.csv"
    summary_path = TME_DIR / "BRCA_TME_conditioned_module_summary.csv"
    label_path = TME_DIR / "BRCA_module_TME_condition_labels.csv"

    assignment = pd.read_csv(assignment_path, low_memory=False)
    summary = pd.read_csv(summary_path, low_memory=False) if summary_path.exists() else pd.DataFrame()
    labels = pd.read_csv(label_path, low_memory=False) if label_path.exists() else pd.DataFrame()

    return assignment, summary, labels


def module_label_lookup(labels):
    out = {}
    if labels.empty:
        return out

    for _, r in labels.iterrows():
        key = (str(r["group"]), str(r["module_id"]))
        out[key] = {
            "module_condition_label": r.get("module_condition_label", "unknown"),
            "top_proxy": r.get("top_proxy", None),
            "top_proxy_r": r.get("top_proxy_r", np.nan),
            "top_category": r.get("top_category", None),
        }
    return out


# ============================================================
# 6. PATIENT MODULE / STRATEGY SCORES
# ============================================================

def compute_module_scores(bp_scores, module_assignment, group):
    sub = module_assignment[module_assignment["group"].astype(str) == group].copy()
    scores = {}
    module_terms = {}

    for module_id, df in sub.groupby("module_id"):
        terms = df["term"].dropna().astype(str).tolist()
        terms = [t for t in terms if t in bp_scores.index]
        if not terms:
            continue
        scores[module_id] = bp_scores.loc[terms].mean(axis=0, skipna=True)
        module_terms[module_id] = terms

    return pd.DataFrame(scores).T, module_terms


def zscore_scores_by_row(score_df):
    out = score_df.copy()
    for idx in out.index:
        out.loc[idx] = zscore_series(out.loc[idx])
    return out


def compute_stage_centroids(module_scores_z, stage_df):
    common = sorted(set(module_scores_z.columns).intersection(set(stage_df["patient"])))
    st = stage_df.set_index("patient").loc[common, "stage_group"]

    early = st[st == "Early"].index.tolist()
    late = st[st == "Late"].index.tolist()

    early_centroid = module_scores_z.loc[:, early].mean(axis=1)
    late_centroid = module_scores_z.loc[:, late].mean(axis=1)

    return early_centroid, late_centroid, early, late


def compute_module_risk_direction(module_scores_z, early_patients, late_patients):
    rows = []
    direction = {}

    for m in module_scores_z.index:
        mu_early = module_scores_z.loc[m, early_patients].mean()
        mu_late = module_scores_z.loc[m, late_patients].mean()
        delta = mu_late - mu_early

        # +1 means higher module score is late/adverse aligned.
        # -1 means lower module score is late/adverse aligned.
        sign = 1 if delta >= 0 else -1
        direction[m] = sign

        rows.append({
            "module_id": m,
            "mu_early_z": mu_early,
            "mu_late_z": mu_late,
            "delta_late_minus_early_z": delta,
            "risk_direction_sign": sign,
            "risk_direction": "high_is_late_aligned" if sign == 1 else "low_is_late_aligned"
        })

    return direction, pd.DataFrame(rows)


def compute_patient_state_metrics(module_scores_z, stage_df, labels):
    early_centroid, late_centroid, early_patients, late_patients = compute_stage_centroids(
        module_scores_z, stage_df
    )
    direction, direction_df = compute_module_risk_direction(
        module_scores_z, early_patients, late_patients
    )

    patients = module_scores_z.columns.tolist()

    label_map = module_label_lookup(labels)

    rows = []

    for p in patients:
        v = module_scores_z[p]

        dis_early = vector_distance(v.values, early_centroid.values)
        dis_late = vector_distance(v.values, late_centroid.values)
        sim_early = vector_corr(v.values, early_centroid.values)
        sim_late = vector_corr(v.values, late_centroid.values)

        risk_aligned = []
        for m in module_scores_z.index:
            risk_aligned.append(direction[m] * v.loc[m])
        risk_aligned = pd.Series(risk_aligned, index=module_scores_z.index)

        adverse_burden = risk_aligned.mean()
        adverse_max = risk_aligned.max()
        adverse_top_module = risk_aligned.sort_values(ascending=False).index[0]

        # Dominant raw active module
        dominant_active_module = v.sort_values(ascending=False).index[0]
        dominant_suppressed_module = v.sort_values(ascending=True).index[0]

        # Label for adverse top module
        group = MODULE_GROUP_FOR_PROFILE
        lab = label_map.get((group, str(adverse_top_module)), {})
        dominant_strategy = infer_strategy_from_label(lab)

        patient_stage = None
        if p in set(stage_df["patient"]):
            vals = stage_df.loc[stage_df["patient"] == p, "stage_group"].values
            if len(vals) > 0:
                patient_stage = vals[0]

        state_class = classify_patient_state(
            adverse_burden=adverse_burden,
            sim_late=sim_late,
            dis_late=dis_late,
            dis_early=dis_early,
            dominant_strategy=dominant_strategy
        )

        rows.append({
            "patient": p,
            "stage_group": patient_stage,
            "adverse_burden": adverse_burden,
            "adverse_max_module_score": adverse_max,
            "adverse_top_module": adverse_top_module,
            "dominant_active_module": dominant_active_module,
            "dominant_suppressed_module": dominant_suppressed_module,
            "dominant_strategy": dominant_strategy,
            "Sim_early_centroid": sim_early,
            "Sim_late_centroid": sim_late,
            "Dis_early_centroid": dis_early,
            "Dis_late_centroid": dis_late,
            "late_vs_early_similarity_delta": sim_late - sim_early if pd.notna(sim_late) and pd.notna(sim_early) else np.nan,
            "late_vs_early_distance_delta": dis_late - dis_early if pd.notna(dis_late) and pd.notna(dis_early) else np.nan,
            "patient_state_class": state_class,
        })

    metrics_df = pd.DataFrame(rows)

    return metrics_df, direction_df, early_centroid, late_centroid


def infer_strategy_from_label(label_info):
    if not label_info:
        return "unknown_strategy"

    label = str(label_info.get("module_condition_label", "unknown")).lower()
    proxy = str(label_info.get("top_proxy", "")).lower()
    cat = str(label_info.get("top_category", "")).lower()

    if "immune" in label or "inflammatory" in label or "immune" in proxy or "inflammatory" in proxy or "tnfa" in proxy or "ifn" in proxy:
        return "immune_inflammatory_route"

    if "stromal" in label or "hypoxia" in label or "emt" in proxy or "angiogenesis" in proxy or "hypoxia" in proxy:
        return "stromal_EMT_hypoxia_route"

    if "metabolic" in label or "glycolysis" in proxy or "oxidative" in proxy or "fatty" in proxy:
        return "metabolic_route"

    if "proliferation" in label or "e2f" in proxy or "g2m" in proxy or "myc" in proxy:
        return "proliferation_route"

    if "stress" in label or "damage" in label or "p53" in proxy or "apoptosis" in proxy or "repair" in proxy:
        return "stress_damage_route"

    return "mixed_or_unclear_route"


def classify_patient_state(adverse_burden, sim_late, dis_late, dis_early, dominant_strategy):
    closer_to_late = False
    if pd.notna(dis_late) and pd.notna(dis_early):
        closer_to_late = dis_late < dis_early

    if adverse_burden >= BURDEN_HIGH and closer_to_late:
        return f"late_adverse_like__{dominant_strategy}"

    if adverse_burden >= BURDEN_HIGH:
        return f"high_burden__{dominant_strategy}"

    if pd.notna(sim_late) and sim_late >= LATE_SIM_HIGH:
        return f"late_similarity__{dominant_strategy}"

    if adverse_burden <= -BURDEN_HIGH:
        return "favorable_or_opposite_to_late"

    return f"intermediate__{dominant_strategy}"


def compute_hallmark_strategy_scores(ge, hallmark_sets):
    terms = list(HALLMARK_STRATEGY_TERMS.values())
    raw_scores, readiness = compute_gene_set_scores(
        ge, hallmark_sets, terms, min_matched=MIN_MATCHED_GENES
    )

    term_to_name = {v: k for k, v in HALLMARK_STRATEGY_TERMS.items()}
    raw_scores.index = [term_to_name.get(t, t) for t in raw_scores.index]
    readiness["proxy_name"] = readiness["term"].map(term_to_name)

    # z-score each proxy across patients
    zscores = zscore_scores_by_row(raw_scores)

    # Category scores
    cat_scores = {}
    for cat, proxies in STRATEGY_CATEGORIES.items():
        present = [p for p in proxies if p in zscores.index]
        if present:
            cat_scores[cat] = zscores.loc[present].mean(axis=0, skipna=True)

    cat_scores = pd.DataFrame(cat_scores).T

    return raw_scores, zscores, cat_scores, readiness


# ============================================================
# 7. FIGURES
# ============================================================

def plot_patient_module_heatmap(module_scores_z, patient_metrics, out_path):
    # Order patients by state class then adverse burden
    order = patient_metrics.sort_values(
        ["patient_state_class", "adverse_burden"],
        ascending=[True, False]
    )["patient"].tolist()
    order = [p for p in order if p in module_scores_z.columns]

    data = module_scores_z.loc[:, order].copy()

    plt.figure(figsize=(16, max(5, data.shape[0] * 0.5)))
    plt.imshow(data.values, aspect="auto", interpolation="nearest", vmin=-2.5, vmax=2.5)
    plt.colorbar(label="Module score z")

    plt.yticks(range(data.shape[0]), data.index.tolist(), fontsize=8)
    plt.xticks([])

    plt.title("BRCA patient-level module activity profile")
    plt.xlabel("Patients ordered by patient-state class")
    plt.ylabel("Late-stage BP modules")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_strategy_heatmap(strategy_scores_z, patient_metrics, out_path):
    order = patient_metrics.sort_values(
        ["patient_state_class", "adverse_burden"],
        ascending=[True, False]
    )["patient"].tolist()
    order = [p for p in order if p in strategy_scores_z.columns]

    data = strategy_scores_z.loc[:, order].copy()

    plt.figure(figsize=(16, max(5, data.shape[0] * 0.5)))
    plt.imshow(data.values, aspect="auto", interpolation="nearest", vmin=-2.5, vmax=2.5)
    plt.colorbar(label="Strategy proxy z")

    plt.yticks(range(data.shape[0]), data.index.tolist(), fontsize=8)
    plt.xticks([])

    plt.title("BRCA patient-level Hallmark strategy proxy profile")
    plt.xlabel("Patients ordered by patient-state class")
    plt.ylabel("Strategy proxies")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pca(module_scores_z, patient_metrics, out_path):
    patients = module_scores_z.columns.tolist()
    X = module_scores_z.T.loc[patients].fillna(0)

    if X.shape[1] < 2 or X.shape[0] < 5:
        return

    X_scaled = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "patient": patients,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    }).merge(patient_metrics, on="patient", how="left")

    classes = plot_df["patient_state_class"].fillna("unknown").unique().tolist()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    colors = [class_to_idx[c] for c in plot_df["patient_state_class"].fillna("unknown")]

    plt.figure(figsize=(9, 7))
    plt.scatter(plot_df["PC1"], plot_df["PC2"], c=colors, s=20, alpha=0.75)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("BRCA patient map based on BP-module state")

    # Add compact legend manually
    handles = []
    for c, idx in class_to_idx.items():
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='', label=c, markersize=6))
    plt.legend(handles=handles, fontsize=7, loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    plot_df.to_csv(OUT_DIR / "BRCA_patient_module_PCA_coordinates.csv", index=False)


def plot_example_patient_dashboard(patient_id, module_scores_z, strategy_cat_scores_z, patient_metrics, out_path):
    if patient_id not in module_scores_z.columns:
        return

    row = patient_metrics[patient_metrics["patient"] == patient_id]
    if row.empty:
        return
    row = row.iloc[0]

    module_values = module_scores_z[patient_id].sort_values(ascending=False)
    strat_values = strategy_cat_scores_z[patient_id].sort_values(ascending=False)

    fig = plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.bar(range(len(module_values)), module_values.values)
    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.set_xticks(range(len(module_values)))
    ax1.set_xticklabels(module_values.index.tolist(), rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Module z-score")
    ax1.set_title(
        f"{patient_id} | {row['patient_state_class']} | burden={row['adverse_burden']:.2f}"
    )

    ax2 = plt.subplot(2, 1, 2)
    ax2.bar(range(len(strat_values)), strat_values.values)
    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.set_xticks(range(len(strat_values)))
    ax2.set_xticklabels(strat_values.index.tolist(), rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Strategy z-score")
    ax2.set_title(
        f"Dominant strategy: {row['dominant_strategy']} | Sim_late={row['Sim_late_centroid']:.2f} | Dis_late={row['Dis_late_centroid']:.2f}"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 100)
    print("MOATTERS BRCA Patient-level BP Strategy Profile V1")
    print("=" * 100)

    print("[Load] GE")
    ge = load_ge_matrix(BRCA_GE_PATH)
    ge = ge.loc[[g for g in ge.index if g not in ["?", "", None]], :]
    print("GE:", ge.shape)

    print("[Load] Stage")
    stage_df = load_brca_stage()
    print(stage_df["stage_group"].value_counts(dropna=False).to_string())

    common_patients = sorted(set(ge.columns).intersection(set(stage_df["patient"])))
    ge = ge.loc[:, common_patients]
    stage_df = stage_df[stage_df["patient"].isin(common_patients)].copy()
    print("Common patients:", len(common_patients))

    print("[Load] GMT")
    go_sets = load_gmt(GO_BP_GMT)
    hallmark_sets = load_gmt(HALLMARK_GMT)
    print("GO BP:", len(go_sets), "Hallmark:", len(hallmark_sets))

    print("[Load] Module files")
    assignment, module_summary, labels = load_module_files()

    # Use late-stage modules as the patient-level strategy basis
    profile_assignment = assignment[assignment["group"].astype(str) == MODULE_GROUP_FOR_PROFILE].copy()
    module_terms = sorted(set(profile_assignment["term"].dropna().astype(str)))
    print("Profile group:", MODULE_GROUP_FOR_PROFILE)
    print("Module BP terms:", len(module_terms))

    print("[Score] BP terms for profile modules")
    bp_scores, bp_readiness = compute_gene_set_scores(
        ge, go_sets, module_terms, min_matched=MIN_MATCHED_GENES
    )
    bp_readiness.to_csv(OUT_DIR / "BRCA_profile_BP_readiness.csv", index=False)
    print("BP scores:", bp_scores.shape)

    print("[Score] module scores")
    module_scores, module_terms_dict = compute_module_scores(
        bp_scores, assignment, MODULE_GROUP_FOR_PROFILE
    )
    module_scores_z = zscore_scores_by_row(module_scores)

    module_scores.to_csv(OUT_DIR / "BRCA_patient_module_scores_raw.csv")
    module_scores_z.to_csv(OUT_DIR / "BRCA_patient_module_scores_z.csv")

    # Save module composition
    module_composition_rows = []
    for m, terms in module_terms_dict.items():
        module_composition_rows.append({
            "module_id": m,
            "n_BP": len(terms),
            "BP_terms": " | ".join(terms)
        })
    pd.DataFrame(module_composition_rows).to_csv(
        OUT_DIR / "BRCA_profile_module_composition.csv",
        index=False
    )

    print("[Patient metrics] adverse burden / centroid similarity")
    patient_metrics, direction_df, early_centroid, late_centroid = compute_patient_state_metrics(
        module_scores_z, stage_df, labels
    )

    patient_metrics.to_csv(OUT_DIR / "BRCA_patient_centroid_similarity_distance.csv", index=False)
    direction_df.to_csv(OUT_DIR / "BRCA_module_late_alignment_direction.csv", index=False)

    pd.DataFrame({
        "module_id": early_centroid.index,
        "early_centroid": early_centroid.values,
        "late_centroid": late_centroid.values,
        "late_minus_early_centroid": late_centroid.values - early_centroid.values
    }).to_csv(OUT_DIR / "BRCA_early_late_module_centroids.csv", index=False)

    print("[Score] Hallmark strategy proxies")
    raw_hm, z_hm, cat_hm, hm_readiness = compute_hallmark_strategy_scores(ge, hallmark_sets)
    raw_hm.to_csv(OUT_DIR / "BRCA_patient_hallmark_strategy_scores_raw.csv")
    z_hm.to_csv(OUT_DIR / "BRCA_patient_hallmark_strategy_scores_z.csv")
    cat_hm.to_csv(OUT_DIR / "BRCA_patient_strategy_category_scores_z.csv")
    hm_readiness.to_csv(OUT_DIR / "BRCA_hallmark_strategy_readiness.csv", index=False)

    # Build patient master table
    print("[Build] master table")
    master = patient_metrics.copy()

    # Add module scores
    mod_pat = module_scores_z.T.reset_index().rename(columns={"index": "patient"})
    mod_pat.columns = ["patient"] + [f"module_{c}_z" for c in mod_pat.columns[1:]]
    master = master.merge(mod_pat, on="patient", how="left")

    # Add strategy category scores
    strat_pat = cat_hm.T.reset_index().rename(columns={"index": "patient"})
    strat_pat.columns = ["patient"] + [f"strategy_{c}_z" for c in strat_pat.columns[1:]]
    master = master.merge(strat_pat, on="patient", how="left")

    # Add all Hallmark proxy scores
    hm_pat = z_hm.T.reset_index().rename(columns={"index": "patient"})
    hm_pat.columns = ["patient"] + [f"hallmark_{c}_z" for c in hm_pat.columns[1:]]
    master = master.merge(hm_pat, on="patient", how="left")

    master = master.sort_values(["patient_state_class", "adverse_burden"], ascending=[True, False])
    master.to_csv(OUT_DIR / "BRCA_patient_strategy_master_table.csv", index=False)

    # Class summary
    class_summary = master.groupby("patient_state_class").agg(
        n_patients=("patient", "count"),
        mean_adverse_burden=("adverse_burden", "mean"),
        mean_Sim_late=("Sim_late_centroid", "mean"),
        mean_Dis_late=("Dis_late_centroid", "mean"),
    ).reset_index().sort_values("n_patients", ascending=False)

    class_summary.to_csv(OUT_DIR / "BRCA_patient_strategy_class_summary.csv", index=False)

    strategy_summary = master.groupby("dominant_strategy").agg(
        n_patients=("patient", "count"),
        mean_adverse_burden=("adverse_burden", "mean"),
        mean_Sim_late=("Sim_late_centroid", "mean"),
        mean_Dis_late=("Dis_late_centroid", "mean"),
    ).reset_index().sort_values("n_patients", ascending=False)

    strategy_summary.to_csv(OUT_DIR / "BRCA_patient_dominant_strategy_summary.csv", index=False)

    print("[Figures]")
    plot_patient_module_heatmap(
        module_scores_z,
        master,
        OUT_DIR / "BRCA_patient_module_activity_heatmap.png"
    )

    plot_strategy_heatmap(
        cat_hm,
        master,
        OUT_DIR / "BRCA_patient_strategy_category_heatmap.png"
    )

    plot_pca(
        module_scores_z,
        master,
        OUT_DIR / "BRCA_patient_module_state_PCA.png"
    )

    # Example patients:
    # pick top adverse, top late similarity, and one favorable/opposite
    example_patients = []

    if not master.empty:
        example_patients.append(master.sort_values("adverse_burden", ascending=False).iloc[0]["patient"])
        example_patients.append(master.sort_values("Sim_late_centroid", ascending=False).iloc[0]["patient"])
        example_patients.append(master.sort_values("adverse_burden", ascending=True).iloc[0]["patient"])

        # one per dominant strategy if possible
        for strat, sub in master.groupby("dominant_strategy"):
            p = sub.sort_values("adverse_burden", ascending=False).iloc[0]["patient"]
            example_patients.append(p)

    example_patients = list(dict.fromkeys(example_patients))

    example_dir = OUT_DIR / "example_patient_dashboards"
    example_dir.mkdir(exist_ok=True)

    for p in example_patients[:15]:
        plot_example_patient_dashboard(
            p,
            module_scores_z,
            cat_hm,
            master,
            example_dir / f"{p}_BP_strategy_dashboard.png"
        )

    # Write summary JSON
    summary = {
        "cancer": "BRCA",
        "n_patients": int(master.shape[0]),
        "n_modules_profile": int(module_scores_z.shape[0]),
        "profile_module_group": MODULE_GROUP_FOR_PROFILE,
        "stage_counts": stage_df["stage_group"].value_counts().to_dict(),
        "patient_state_class_counts": master["patient_state_class"].value_counts().to_dict(),
        "dominant_strategy_counts": master["dominant_strategy"].value_counts().to_dict(),
        "mean_adverse_burden": float(master["adverse_burden"].mean()),
        "max_adverse_burden": float(master["adverse_burden"].max()),
        "mean_Sim_late_centroid": float(master["Sim_late_centroid"].mean()),
        "mean_Dis_late_centroid": float(master["Dis_late_centroid"].mean()),
    }

    with open(OUT_DIR / "BRCA_patient_strategy_profile_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE]")
    print("Output folder:")
    print(OUT_DIR)

    print("\nClass summary:")
    print(class_summary.to_string(index=False))

    print("\nDominant strategy summary:")
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
