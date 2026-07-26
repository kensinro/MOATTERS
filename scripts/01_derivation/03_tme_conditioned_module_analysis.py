# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# ============================================================
# MOATTERS TME-conditioned BP Module Analysis V1
# Date: 2026-05-31
#
# Input:
#   1. Raw GE data:
#      D:\MOATTERS-Data\UCSC_XENA\<Cancer Folder>\GE.tsv
#
#   2. GMT files:
#      D:\MOATTERS-Data\GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt
#      D:\MOATTERS-Data\GSEA\h.all.v2026.1.Hs.symbols.gmt
#
#   3. Stage BP Attractor V3:
#      D:\MOATTERS-Output\MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530
#
#   4. Module Rewiring V1:
#      D:\MOATTERS-Output\MOATTERS_STAGE_BP_Module_Rewiring_V1_20260531
#
# Output:
#      D:\MOATTERS-Output\MOATTERS_TME_Conditioned_BP_Modules_V1_20260531
#
# Purpose:
#   Test whether stage-associated BP co-activity modules are associated
#   with tumor microenvironment-like functional proxies.
#
#   This version uses Hallmark signatures as internal TME/progression proxies:
#      immune / inflammatory
#      EMT / stromal
#      hypoxia / angiogenesis
#      metabolism
#      proliferation / cell cycle
#      DNA damage / p53
#
# No external TME score file is required for V1.
# ============================================================

import os
import re
import json
import math
import warnings
from pathlib import Path
from moatters.config import data_path, output_path
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, mannwhitneyu
import matplotlib.pyplot as plt


# ============================================================
# 0. PATH CONFIG
# ============================================================

DATA_ROOT = data_path()
TEMP_ROOT = output_path()

UCSC_XENA_DIR = DATA_ROOT / "UCSC_XENA"
GSEA_DIR = DATA_ROOT / "GSEA"

GO_BP_GMT = GSEA_DIR / "c5.go.bp.v2026.1.Hs.symbols.gmt"
HALLMARK_GMT = GSEA_DIR / "h.all.v2026.1.Hs.symbols.gmt"

V3_DIR = TEMP_ROOT / "MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530"
MODULE_DIR = TEMP_ROOT / "MOATTERS_STAGE_BP_Module_Rewiring_V1_20260531"

OUT_DIR = TEMP_ROOT / "MOATTERS_TME_Conditioned_BP_Modules_V1_20260531"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Data root:", DATA_ROOT)
print("V3 input:", V3_DIR)
print("Module input:", MODULE_DIR)
print("Output:", OUT_DIR)


# ============================================================
# 1. USER CONFIG
# ============================================================

CANCER_FOLDERS = {
    "BLCA": "Bladder Cancer (BLCA)",
    "BRCA": "Breast Cancer (BRCA)",
    "CESC": "Cervical Cancer (CESC)",
    "COAD": "Colon Cancer (COAD)",
    "LUAD": "Lung Adenocarcinoma (LUAD)",
    "LUSC": "Lung Squamous Cell Carcinoma (LUSC)",
    "KIRC": "Kidney Clear Cell Carcinoma (KIRC)",
    "LIHC": "Liver Cancer (LIHC)",
}

GROUPS_TO_ANALYZE = ["stage_I_II", "stage_III_IV"]

MIN_MATCHED_GENES = 10
MIN_PATIENTS_FOR_CORR = 30

# Hallmark proxy terms.
# These are internal functional proxies, not direct deconvolution estimates.
TME_PROXY_TERMS = {
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
    "cholesterol_homeostasis": "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
    "proliferation_E2F": "HALLMARK_E2F_TARGETS",
    "proliferation_G2M": "HALLMARK_G2M_CHECKPOINT",
    "MYC": "HALLMARK_MYC_TARGETS_V1",
    "p53": "HALLMARK_P53_PATHWAY",
    "apoptosis": "HALLMARK_APOPTOSIS",
    "DNA_repair": "HALLMARK_DNA_REPAIR",
    "mTORC1": "HALLMARK_MTORC1_SIGNALING",
}

# Category grouping for interpretation
TME_PROXY_CATEGORIES = {
    "TME_immune_inflammatory": [
        "immune_IFN_gamma",
        "immune_IFN_alpha",
        "inflammatory",
        "TNFA_NFKB",
    ],
    "TME_stromal_hypoxia": [
        "EMT_stromal",
        "angiogenesis",
        "hypoxia",
    ],
    "metabolic_adaptation": [
        "glycolysis",
        "oxidative_phosphorylation",
        "fatty_acid_metabolism",
        "cholesterol_homeostasis",
    ],
    "tumor_intrinsic_proliferation": [
        "proliferation_E2F",
        "proliferation_G2M",
        "MYC",
    ],
    "stress_damage_death": [
        "p53",
        "apoptosis",
        "DNA_repair",
        "mTORC1",
    ],
}

# Correlation thresholds for interpretation
STRONG_CORR = 0.50
MODERATE_CORR = 0.30
P_CUTOFF = 0.05

MAKE_FIGURES = True


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

    bom_enc = detect_encoding_by_bom(path)

    encodings = []
    if bom_enc:
        encodings.append(bom_enc)

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


def safe_neglog10(p):
    if p is None or pd.isna(p):
        return np.nan
    p = float(p)
    if p <= 0:
        return 300.0
    return -math.log10(p)


# ============================================================
# 3. LOAD GMT AND GE
# ============================================================

def load_gmt(gmt_path):
    gene_sets = {}

    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            term = parts[0].strip()
            genes = [
                clean_gene_symbol(g)
                for g in parts[2:]
                if clean_gene_symbol(g)
            ]
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

    # Case A: genes rows, sample columns
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

    # Case B: samples rows, genes columns
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

    # Case C: assume first column gene
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


def compute_scores_for_terms(ge, gene_sets, terms=None, min_matched=10):
    """
    Compute mean-z gene-set score.
    Return:
      score_df: terms x patients
      readiness_df
    """
    ge_genes = set(ge.index.astype(str))
    zge = zscore_rows(ge)

    if terms is None:
        terms = list(gene_sets.keys())

    scores = {}
    rows = []

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

        genes = sorted(set([g for g in gene_sets[term] if g in ge_genes]))
        n_defined = len(gene_sets[term])
        n_matched = len(genes)
        frac = n_matched / n_defined if n_defined > 0 else 0

        if n_matched >= min_matched:
            scores[term] = zge.loc[genes].mean(axis=0, skipna=True)
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
# 4. STAGE LOADING FROM V3 OUTPUT
# ============================================================

def load_stage_from_v3(cancer_code):
    """
    Use V3 selected BP result n_early/n_late only does not give patient labels.
    So reload the stage file from raw cancer folder using same logic as V3.
    """

    cancer_folder = CANCER_FOLDERS[cancer_code]
    cancer_dir = UCSC_XENA_DIR / cancer_folder

    files = []
    files += list(cancer_dir.glob("*stage_groups_from_survival*.tsv"))
    files += list(cancer_dir.glob("Phenotype.tsv"))
    files += list(cancer_dir.glob("*clinicalMatrix*"))
    files += list(cancer_dir.glob("*clinical*.tsv"))
    files += list(cancer_dir.glob("*phenotype*.tsv"))

    seen = set()
    files2 = []
    for f in files:
        if f.exists() and str(f) not in seen:
            files2.append(f)
            seen.add(str(f))

    for f in files2:
        stage_df, diag = parse_stage_file(f)
        counts = stage_df["stage_group"].value_counts().to_dict() if not stage_df.empty else {}
        if set(counts.keys()) >= {"Early", "Late"}:
            print(f"[Stage] {cancer_code}: using {f.name}, counts={counts}")
            return stage_df

    print(f"[Stage] {cancer_code}: no both Early/Late stage file found.")
    return pd.DataFrame(columns=["patient", "stage_group"])


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
    priority_patterns = [
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

    for p in priority_patterns:
        if p.lower() in cols_lower and cols_lower[p.lower()] not in ordered:
            ordered.append(cols_lower[p.lower()])

    for c in cols:
        if "stage" in str(c).lower() and c not in ordered:
            ordered.append(c)

    return ordered


def parse_stage_file(stage_file):
    try:
        df = read_table_safely(stage_file)
    except Exception as e:
        return pd.DataFrame(columns=["patient", "stage_group"]), {
            "file": str(stage_file),
            "status": "read_failed",
            "error": repr(e)[:300]
        }

    if df.empty or df.shape[1] < 2:
        return pd.DataFrame(columns=["patient", "stage_group"]), {
            "file": str(stage_file),
            "status": "empty"
        }

    patient_col = infer_patient_col(df)
    stage_cols = infer_stage_cols(df)

    if not stage_cols:
        return pd.DataFrame(columns=["patient", "stage_group"]), {
            "file": str(stage_file),
            "status": "no_stage_col"
        }

    best_out = pd.DataFrame(columns=["patient", "stage_group"])
    best_diag = None

    for stage_col in stage_cols:
        out = pd.DataFrame()
        out["patient"] = df[patient_col].map(tcga_patient_id)
        out["stage_group"] = df[stage_col].map(roman_stage_to_group)
        out = out.dropna(subset=["patient", "stage_group"])
        out = out.drop_duplicates(subset=["patient"])

        counts = out["stage_group"].value_counts().to_dict() if not out.empty else {}

        diag = {
            "file": str(stage_file),
            "patient_col": str(patient_col),
            "stage_col": str(stage_col),
            "n": int(out.shape[0]),
            "counts": counts
        }

        if set(counts.keys()) >= {"Early", "Late"}:
            return out, diag

        if out.shape[0] > best_out.shape[0]:
            best_out = out
            best_diag = diag

    return best_out, best_diag


# ============================================================
# 5. LOAD MODULE RESULTS
# ============================================================

def load_module_assignment(cancer_code):
    path = MODULE_DIR / cancer_code / f"{cancer_code}_module_assignment.csv"
    if not path.exists():
        print(f"[Missing] {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_module_summary(cancer_code):
    path = MODULE_DIR / cancer_code / f"{cancer_code}_module_summary.csv"
    if not path.exists():
        print(f"[Missing] {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_dstage_results(cancer_code):
    path = V3_DIR / cancer_code / f"{cancer_code}_DStage_BP_results.csv"
    if not path.exists():
        print(f"[Missing] {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


# ============================================================
# 6. MODULE SCORE AND TME CORRELATION
# ============================================================

def compute_module_scores(bp_scores, module_assignment, group_name):
    """
    module_assignment has:
      group, module_id, term

    Return:
      modules x patients
    """
    sub = module_assignment[module_assignment["group"].astype(str) == group_name].copy()

    if sub.empty:
        return pd.DataFrame()

    scores = {}

    for module_id, df in sub.groupby("module_id"):
        terms = df["term"].dropna().astype(str).tolist()
        terms = [t for t in terms if t in bp_scores.index]

        if len(terms) == 0:
            continue

        scores[module_id] = bp_scores.loc[terms].mean(axis=0, skipna=True)

    return pd.DataFrame(scores).T


def compute_stage_group_test(score_series, stage_df):
    common = sorted(set(score_series.index).intersection(set(stage_df["patient"])))
    if len(common) < 20:
        return {
            "n_early": 0,
            "n_late": 0,
            "mu_early": np.nan,
            "mu_late": np.nan,
            "delta_late_minus_early": np.nan,
            "p_stage": np.nan,
            "D_stage": np.nan,
        }

    tmp = stage_df.set_index("patient").loc[common, "stage_group"]
    early = tmp[tmp == "Early"].index.tolist()
    late = tmp[tmp == "Late"].index.tolist()

    x_early = score_series.loc[early].dropna().astype(float)
    x_late = score_series.loc[late].dropna().astype(float)

    if len(x_early) < 10 or len(x_late) < 10:
        return {
            "n_early": len(x_early),
            "n_late": len(x_late),
            "mu_early": np.nan,
            "mu_late": np.nan,
            "delta_late_minus_early": np.nan,
            "p_stage": np.nan,
            "D_stage": np.nan,
        }

    try:
        stat, p = mannwhitneyu(x_late, x_early, alternative="two-sided")
    except Exception:
        p = np.nan

    mu_early = float(x_early.mean())
    mu_late = float(x_late.mean())

    return {
        "n_early": int(len(x_early)),
        "n_late": int(len(x_late)),
        "mu_early": mu_early,
        "mu_late": mu_late,
        "delta_late_minus_early": mu_late - mu_early,
        "p_stage": p,
        "D_stage": safe_neglog10(p),
    }


def correlate_modules_with_tme(module_scores, tme_scores, patients=None):
    """
    module_scores: modules x patients
    tme_scores: proxies x patients
    """
    rows = []

    if module_scores.empty or tme_scores.empty:
        return pd.DataFrame()

    if patients is None:
        patients = sorted(set(module_scores.columns).intersection(set(tme_scores.columns)))
    else:
        patients = sorted(set(patients).intersection(set(module_scores.columns)).intersection(set(tme_scores.columns)))

    if len(patients) < MIN_PATIENTS_FOR_CORR:
        return pd.DataFrame()

    for module_id in module_scores.index:
        x = module_scores.loc[module_id, patients].astype(float)

        for proxy in tme_scores.index:
            y = tme_scores.loc[proxy, patients].astype(float)

            ok = x.notna() & y.notna()
            if ok.sum() < MIN_PATIENTS_FOR_CORR:
                continue

            r, p = spearmanr(x[ok], y[ok])

            rows.append({
                "module_id": module_id,
                "tme_proxy": proxy,
                "n": int(ok.sum()),
                "spearman_r": float(r) if not pd.isna(r) else np.nan,
                "p_corr": p,
                "abs_r": abs(float(r)) if not pd.isna(r) else np.nan,
                "corr_strength": classify_corr_strength(r, p),
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["module_id", "abs_r"], ascending=[True, False])

    return out


def classify_corr_strength(r, p):
    if pd.isna(r):
        return "NA"
    if pd.isna(p):
        sig = False
    else:
        sig = p <= P_CUTOFF

    ar = abs(float(r))

    if sig and ar >= STRONG_CORR:
        return "strong"
    if sig and ar >= MODERATE_CORR:
        return "moderate"
    if sig:
        return "weak_significant"
    return "not_significant"


def assign_tme_condition_label(corr_df):
    """
    Label a module by strongest correlated proxy category.
    """
    if corr_df.empty:
        return {
            "module_condition_label": "unknown",
            "top_proxy": None,
            "top_proxy_r": np.nan,
            "top_proxy_p": np.nan,
            "top_proxy_strength": None,
            "top_category": None,
        }

    sig = corr_df[corr_df["p_corr"] <= P_CUTOFF].copy()
    if sig.empty:
        top = corr_df.sort_values("abs_r", ascending=False).iloc[0]
        return {
            "module_condition_label": "weak_or_unclear",
            "top_proxy": top["tme_proxy"],
            "top_proxy_r": top["spearman_r"],
            "top_proxy_p": top["p_corr"],
            "top_proxy_strength": top["corr_strength"],
            "top_category": infer_proxy_category(top["tme_proxy"]),
        }

    top = sig.sort_values("abs_r", ascending=False).iloc[0]
    cat = infer_proxy_category(top["tme_proxy"])

    # Broad labels
    if cat in ["TME_immune_inflammatory", "TME_stromal_hypoxia"]:
        label = f"TME_conditioned__{cat}"
    elif cat == "tumor_intrinsic_proliferation":
        label = "tumor_intrinsic_proliferation_like"
    elif cat == "metabolic_adaptation":
        label = "metabolic_adaptation_like"
    elif cat == "stress_damage_death":
        label = "stress_damage_like"
    else:
        label = "mixed_or_unclear"

    return {
        "module_condition_label": label,
        "top_proxy": top["tme_proxy"],
        "top_proxy_r": top["spearman_r"],
        "top_proxy_p": top["p_corr"],
        "top_proxy_strength": top["corr_strength"],
        "top_category": cat,
    }


def infer_proxy_category(proxy):
    for cat, proxies in TME_PROXY_CATEGORIES.items():
        if proxy in proxies:
            return cat
    return "other"


# ============================================================
# 7. FIGURES
# ============================================================

def plot_module_tme_heatmap(corr_df, cancer_code, group_name, out_path):
    if corr_df.empty:
        return

    mat = corr_df.pivot_table(
        index="module_id",
        columns="tme_proxy",
        values="spearman_r",
        aggfunc="mean"
    )

    if mat.empty:
        return

    plt.figure(figsize=(max(10, mat.shape[1] * 0.55), max(4, mat.shape[0] * 0.55)))
    plt.imshow(mat.values, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman r")

    plt.xticks(range(mat.shape[1]), mat.columns.tolist(), rotation=90, fontsize=7)
    plt.yticks(range(mat.shape[0]), mat.index.tolist(), fontsize=8)

    plt.title(f"{cancer_code} {group_name}: module vs TME/progression proxies")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_module_stage_delta(module_stage_df, cancer_code, out_path):
    if module_stage_df.empty:
        return

    df = module_stage_df.copy()
    df["label"] = df["group"].astype(str) + ":" + df["module_id"].astype(str)

    plt.figure(figsize=(max(10, df.shape[0] * 0.45), 5))
    x = np.arange(df.shape[0])
    plt.bar(x, df["delta_late_minus_early"].values)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, df["label"].tolist(), rotation=90, fontsize=7)
    plt.ylabel("Module score change: Late - Early")
    plt.title(f"{cancer_code}: module stage shift")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# 8. PER-CANCER ANALYSIS
# ============================================================

def run_one_cancer(cancer_code, go_gene_sets, hallmark_gene_sets):
    print("\n" + "=" * 100)
    print(f"[Cancer] {cancer_code}")

    cancer_folder = CANCER_FOLDERS[cancer_code]
    ge_path = UCSC_XENA_DIR / cancer_folder / "GE.tsv"

    if not ge_path.exists():
        print("[Skip] missing GE:", ge_path)
        return None

    out_dir = OUT_DIR / cancer_code
    out_dir.mkdir(parents=True, exist_ok=True)

    module_assignment = load_module_assignment(cancer_code)
    module_summary = load_module_summary(cancer_code)
    dstage = load_dstage_results(cancer_code)

    if module_assignment.empty:
        print("[Skip] missing module assignment")
        return None

    print("[Load] GE")
    ge = load_ge_matrix(ge_path)
    ge = ge.loc[[g for g in ge.index if g not in ["?", "", None]], :]
    print("GE shape:", ge.shape)

    print("[Load] stage")
    stage_df = load_stage_from_v3(cancer_code)
    if stage_df.empty:
        print("[Skip] no stage")
        return None

    common_patients = sorted(set(ge.columns).intersection(set(stage_df["patient"])))
    if len(common_patients) < 40:
        print("[Skip] too few common patients")
        return None

    ge = ge.loc[:, common_patients]
    stage_df = stage_df[stage_df["patient"].isin(common_patients)].copy()

    print("Common patients:", len(common_patients))
    print(stage_df["stage_group"].value_counts().to_string())

    # Compute GO BP scores only for module BP terms
    module_terms = sorted(set(module_assignment["term"].dropna().astype(str)))
    print("Module BP terms:", len(module_terms))

    print("[Score] GO BP module terms")
    bp_scores, bp_readiness = compute_scores_for_terms(
        ge,
        go_gene_sets,
        terms=module_terms,
        min_matched=MIN_MATCHED_GENES
    )
    bp_readiness.to_csv(out_dir / f"{cancer_code}_module_BP_readiness_recomputed.csv", index=False)

    print("BP scores:", bp_scores.shape)

    # Hallmark TME proxy scores
    hallmark_terms = list(TME_PROXY_TERMS.values())

    print("[Score] Hallmark TME proxies")
    hallmark_scores_raw, hallmark_readiness = compute_scores_for_terms(
        ge,
        hallmark_gene_sets,
        terms=hallmark_terms,
        min_matched=MIN_MATCHED_GENES
    )

    # Rename Hallmark term rows to proxy names
    term_to_proxy = {v: k for k, v in TME_PROXY_TERMS.items()}
    hallmark_scores = hallmark_scores_raw.copy()
    hallmark_scores.index = [term_to_proxy.get(t, t) for t in hallmark_scores.index]

    hallmark_readiness["proxy_name"] = hallmark_readiness["term"].map(term_to_proxy)
    hallmark_readiness.to_csv(out_dir / f"{cancer_code}_hallmark_TME_proxy_readiness.csv", index=False)
    hallmark_scores.to_csv(out_dir / f"{cancer_code}_hallmark_TME_proxy_scores.csv")

    print("TME proxy scores:", hallmark_scores.shape)

    all_corrs = []
    all_module_stage_rows = []
    all_module_labels = []

    for group_name in GROUPS_TO_ANALYZE:
        print(f"[Group] {group_name}")

        module_scores = compute_module_scores(bp_scores, module_assignment, group_name)

        if module_scores.empty:
            print("  no module scores")
            continue

        module_scores.to_csv(out_dir / f"{cancer_code}_{group_name}_module_scores.csv")

        # Correlation all patients
        corr_all = correlate_modules_with_tme(module_scores, hallmark_scores, patients=common_patients)
        if not corr_all.empty:
            corr_all["cancer"] = cancer_code
            corr_all["group"] = group_name
            corr_all["patient_subset"] = "all_stage_patients"
            all_corrs.append(corr_all)

            plot_module_tme_heatmap(
                corr_all,
                cancer_code,
                group_name,
                out_dir / f"{cancer_code}_{group_name}_module_TME_proxy_corr_heatmap.png"
            )

        # Correlation within early and late separately
        st = stage_df.set_index("patient").loc[common_patients, "stage_group"]
        early_patients = st[st == "Early"].index.tolist()
        late_patients = st[st == "Late"].index.tolist()

        corr_early = correlate_modules_with_tme(module_scores, hallmark_scores, patients=early_patients)
        if not corr_early.empty:
            corr_early["cancer"] = cancer_code
            corr_early["group"] = group_name
            corr_early["patient_subset"] = "early_only"
            all_corrs.append(corr_early)

        corr_late = correlate_modules_with_tme(module_scores, hallmark_scores, patients=late_patients)
        if not corr_late.empty:
            corr_late["cancer"] = cancer_code
            corr_late["group"] = group_name
            corr_late["patient_subset"] = "late_only"
            all_corrs.append(corr_late)

        # Stage shift per module
        for module_id in module_scores.index:
            s = module_scores.loc[module_id]
            stat = compute_stage_group_test(s, stage_df)
            row = {
                "cancer": cancer_code,
                "group": group_name,
                "module_id": module_id,
                **stat
            }
            all_module_stage_rows.append(row)

        # Label module by strongest all-patient TME/proxy correlation
        if not corr_all.empty:
            for module_id, sub_corr in corr_all.groupby("module_id"):
                label_info = assign_tme_condition_label(sub_corr)
                all_module_labels.append({
                    "cancer": cancer_code,
                    "group": group_name,
                    "module_id": module_id,
                    **label_info
                })

    corr_df = pd.concat(all_corrs, ignore_index=True) if all_corrs else pd.DataFrame()
    module_stage_df = pd.DataFrame(all_module_stage_rows)
    module_label_df = pd.DataFrame(all_module_labels)

    if not corr_df.empty:
        corr_df = corr_df[
            [
                "cancer", "group", "patient_subset", "module_id",
                "tme_proxy", "n", "spearman_r", "p_corr", "abs_r", "corr_strength"
            ]
        ]
        corr_df.to_csv(out_dir / f"{cancer_code}_module_TME_proxy_correlations.csv", index=False)

    if not module_stage_df.empty:
        module_stage_df.to_csv(out_dir / f"{cancer_code}_module_stage_shift.csv", index=False)
        plot_module_stage_delta(
            module_stage_df,
            cancer_code,
            out_dir / f"{cancer_code}_module_stage_shift_barplot.png"
        )

    if not module_label_df.empty:
        module_label_df.to_csv(out_dir / f"{cancer_code}_module_TME_condition_labels.csv", index=False)

    # Merge label + module summary for easy interpretation
    merged_summary = module_summary.copy()
    if not module_label_df.empty and not merged_summary.empty:
        merged_summary = merged_summary.merge(
            module_label_df,
            on=["cancer", "group", "module_id"],
            how="left"
        )

    if not module_stage_df.empty and not merged_summary.empty:
        merged_summary = merged_summary.merge(
            module_stage_df[
                [
                    "cancer", "group", "module_id",
                    "n_early", "n_late", "mu_early", "mu_late",
                    "delta_late_minus_early", "p_stage", "D_stage"
                ]
            ],
            on=["cancer", "group", "module_id"],
            how="left",
            suffixes=("", "_module_score")
        )

    if not merged_summary.empty:
        merged_summary.to_csv(out_dir / f"{cancer_code}_TME_conditioned_module_summary.csv", index=False)

    # Compact cancer summary
    summary = summarize_cancer_tme_modules(
        cancer_code,
        corr_df,
        module_stage_df,
        module_label_df,
        module_summary
    )

    with open(out_dir / f"{cancer_code}_TME_conditioned_module_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[Summary]")
    for k, v in summary.items():
        print(" ", k, ":", v)

    return {
        "summary": summary,
        "corr": corr_df,
        "module_stage": module_stage_df,
        "module_label": module_label_df,
        "merged_summary": merged_summary,
    }


def summarize_cancer_tme_modules(cancer_code, corr_df, module_stage_df, module_label_df, module_summary):
    out = {"cancer": cancer_code}

    if not module_label_df.empty:
        counts = module_label_df["module_condition_label"].value_counts().to_dict()
        for k, v in counts.items():
            out[f"module_label_count__{k}"] = int(v)

        out["n_labeled_modules"] = int(module_label_df.shape[0])

        strong_or_mod = module_label_df[
            module_label_df["top_proxy_strength"].isin(["strong", "moderate"])
        ]
        out["n_modules_moderate_or_strong_TME_proxy"] = int(strong_or_mod.shape[0])

    if not corr_df.empty:
        sig = corr_df[corr_df["p_corr"] <= P_CUTOFF]
        out["n_significant_module_proxy_correlations"] = int(sig.shape[0])
        out["max_abs_module_proxy_r"] = float(corr_df["abs_r"].max())

        top = corr_df.sort_values("abs_r", ascending=False).iloc[0]
        out["top_module_proxy_pair"] = f"{top['group']}:{top['module_id']}~{top['tme_proxy']}"
        out["top_module_proxy_r"] = float(top["spearman_r"])
        out["top_module_proxy_p"] = float(top["p_corr"])

    if not module_stage_df.empty:
        out["n_module_stage_tests"] = int(module_stage_df.shape[0])
        out["max_module_D_stage"] = float(module_stage_df["D_stage"].max())
        out["mean_abs_module_delta_stage"] = float(module_stage_df["delta_late_minus_early"].abs().mean())

        n_late_high = int((module_stage_df["delta_late_minus_early"] > 0).sum())
        n_late_low = int((module_stage_df["delta_late_minus_early"] < 0).sum())

        out["n_modules_late_high"] = n_late_high
        out["n_modules_late_low"] = n_late_low

    if not module_summary.empty:
        out["n_original_modules"] = int(module_summary.shape[0])
        if "redundancy_warning" in module_summary.columns:
            out["n_original_modules_with_redundancy_warning"] = int(module_summary["redundancy_warning"].sum())

    return out


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("=" * 100)
    print("MOATTERS TME-conditioned BP Module Analysis V1")
    print("=" * 100)

    if not GO_BP_GMT.exists():
        raise FileNotFoundError(f"GO BP GMT not found: {GO_BP_GMT}")
    if not HALLMARK_GMT.exists():
        raise FileNotFoundError(f"Hallmark GMT not found: {HALLMARK_GMT}")

    print("[Load] GO BP GMT")
    go_gene_sets = load_gmt(GO_BP_GMT)
    print("GO BP gene sets:", len(go_gene_sets))

    print("[Load] Hallmark GMT")
    hallmark_gene_sets = load_gmt(HALLMARK_GMT)
    print("Hallmark gene sets:", len(hallmark_gene_sets))

    all_summaries = []
    all_corrs = []
    all_module_stage = []
    all_labels = []
    all_merged = []

    for cancer_code in CANCER_FOLDERS.keys():
        try:
            res = run_one_cancer(cancer_code, go_gene_sets, hallmark_gene_sets)

            if res is None:
                continue

            all_summaries.append(res["summary"])

            if not res["corr"].empty:
                all_corrs.append(res["corr"])

            if not res["module_stage"].empty:
                all_module_stage.append(res["module_stage"])

            if not res["module_label"].empty:
                all_labels.append(res["module_label"])

            if not res["merged_summary"].empty:
                all_merged.append(res["merged_summary"])

        except Exception as e:
            print(f"[ERROR] {cancer_code}: {repr(e)}")
            import traceback
            traceback.print_exc()

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(
            OUT_DIR / "MOATTERS_TME_conditioned_module_all_cancer_summary.csv",
            index=False
        )

    if all_corrs:
        pd.concat(all_corrs, ignore_index=True).to_csv(
            OUT_DIR / "MOATTERS_module_TME_proxy_correlations_all_cancers.csv",
            index=False
        )

    if all_module_stage:
        pd.concat(all_module_stage, ignore_index=True).to_csv(
            OUT_DIR / "MOATTERS_module_stage_shift_all_cancers.csv",
            index=False
        )

    if all_labels:
        pd.concat(all_labels, ignore_index=True).to_csv(
            OUT_DIR / "MOATTERS_module_TME_condition_labels_all_cancers.csv",
            index=False
        )

    if all_merged:
        pd.concat(all_merged, ignore_index=True).to_csv(
            OUT_DIR / "MOATTERS_TME_conditioned_module_summary_all_cancers.csv",
            index=False
        )

    print("\n[DONE]")
    print("Output folder:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
