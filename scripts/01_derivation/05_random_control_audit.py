# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# ============================================================
# MOATTERS BRCA Patient-level BP Strategy RANDOM TEST V1
# Date: 2026-05-31
#
# Purpose:
#   Stress-test the BRCA patient-level BP/module-state idea.
#
# Tests:
#   1. Random BP modules:
#      Keep real module sizes, randomly sample BP terms from candidate BP pool.
#
#   2. Stage-label shuffle:
#      Keep real module scores, shuffle Early/Late labels.
#
#   3. Patient scramble:
#      Keep each module distribution, randomly permute patients per module,
#      destroying patient-level co-activity structure.
#
# Input:
#   Raw:
#     D:\MOATTERS-Data\UCSC_XENA\Breast Cancer (BRCA)\GE.tsv
#     D:\MOATTERS-Data\GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt
#
#   Previous outputs:
#     D:\MOATTERS-Output\MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA
#     D:\MOATTERS-Output\MOATTERS_STAGE_BP_Module_Rewiring_V1_20260531\BRCA
#     D:\MOATTERS-Output\MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531
#
# Output:
#     D:\MOATTERS-Output\MOATTERS_BRCA_PATIENT_STATE_RANDOM_TEST_V1_20260531
# ============================================================

import re
import math
import json
from pathlib import Path
from moatters.config import data_path, output_path
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, spearmanr
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

V3_BRCA_DIR = TEMP_ROOT / r"MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530\BRCA"
MODULE_BRCA_DIR = TEMP_ROOT / r"MOATTERS_STAGE_BP_Module_Rewiring_V1_20260531\BRCA"
PROFILE_BRCA_DIR = TEMP_ROOT / r"MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531"

OUT_DIR = TEMP_ROOT / "MOATTERS_BRCA_PATIENT_STATE_RANDOM_TEST_V1_20260531"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Output:", OUT_DIR)


# ============================================================
# 1. USER CONFIG
# ============================================================

RANDOM_SEED = 20260531
rng = np.random.default_rng(RANDOM_SEED)

N_RANDOM_MODULES = 500          # Random BP module repeats
N_STAGE_SHUFFLE = 1000          # Stage-label shuffle repeats
N_PATIENT_SCRAMBLE = 500        # Patient scramble repeats

MIN_MATCHED_GENES = 10

MODULE_GROUP_FOR_PROFILE = "stage_III_IV"

# Candidate BP pool:
#   "all_ready" = all observation-ready GO BP from V3 readiness table
#   "tested"    = all BP tested in DStage results
#   "selected"  = only selected top BP; not recommended for random baseline
CANDIDATE_POOL_MODE = "tested"

# Real summary thresholds
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


def zscore_series(s):
    s = pd.Series(s).astype(float)
    sd = s.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd


def zscore_rows(mat):
    mean = mat.mean(axis=1, skipna=True)
    std = mat.std(axis=1, skipna=True).replace(0, np.nan)
    return mat.sub(mean, axis=0).div(std, axis=0)


def safe_neglog10(p):
    if p is None or pd.isna(p):
        return np.nan
    p = float(p)
    if p <= 0:
        return 300.0
    return -math.log10(p)


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


def empirical_p_greater_equal(real_value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0 or not np.isfinite(real_value):
        return np.nan
    return float((np.sum(null_values >= real_value) + 1) / (len(null_values) + 1))


def empirical_p_less_equal(real_value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0 or not np.isfinite(real_value):
        return np.nan
    return float((np.sum(null_values <= real_value) + 1) / (len(null_values) + 1))


# ============================================================
# 3. LOAD GE / GMT / STAGE
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
        "stage_group", "stage_raw", "pathologic_stage", "ajcc_pathologic_stage",
        "clinical_stage", "tumor_stage", "pathologic_tumor_stage",
        "clinical_tumor_stage", "Converted_Stage_nature2012",
        "AJCC_Stage_nature2012", "stage"
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
# 4. BP SCORE COMPUTATION
# ============================================================

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
# 5. LOAD PREVIOUS RESULTS
# ============================================================

def load_previous_results():
    profile_master_path = PROFILE_BRCA_DIR / "BRCA_patient_strategy_master_table.csv"
    real_module_scores_path = PROFILE_BRCA_DIR / "BRCA_patient_module_scores_z.csv"
    direction_path = PROFILE_BRCA_DIR / "BRCA_module_late_alignment_direction.csv"

    assignment_path = MODULE_BRCA_DIR / "BRCA_module_assignment.csv"
    dstage_path = V3_BRCA_DIR / "BRCA_DStage_BP_results.csv"
    selected_path = V3_BRCA_DIR / "BRCA_selected_BP_for_network.csv"
    readiness_path = V3_BRCA_DIR / "BRCA_BP_readiness.csv"

    profile_master = pd.read_csv(profile_master_path, low_memory=False)
    real_module_scores_z = pd.read_csv(real_module_scores_path, index_col=0)
    direction_df = pd.read_csv(direction_path, low_memory=False)

    assignment = pd.read_csv(assignment_path, low_memory=False)
    dstage = pd.read_csv(dstage_path, low_memory=False)
    selected = pd.read_csv(selected_path, low_memory=False)
    readiness = pd.read_csv(readiness_path, low_memory=False)

    return {
        "profile_master": profile_master,
        "real_module_scores_z": real_module_scores_z,
        "direction_df": direction_df,
        "assignment": assignment,
        "dstage": dstage,
        "selected": selected,
        "readiness": readiness,
    }


def get_candidate_bp_pool(prev):
    if CANDIDATE_POOL_MODE == "all_ready":
        readiness = prev["readiness"]
        pool = readiness.loc[
            readiness["readiness_class"] == "observation_ready",
            "term"
        ].dropna().astype(str).tolist()

    elif CANDIDATE_POOL_MODE == "tested":
        dstage = prev["dstage"]
        pool = dstage["term"].dropna().astype(str).tolist()

    elif CANDIDATE_POOL_MODE == "selected":
        selected = prev["selected"]
        pool = selected["term"].dropna().astype(str).tolist()

    else:
        raise ValueError(f"Unknown CANDIDATE_POOL_MODE: {CANDIDATE_POOL_MODE}")

    pool = sorted(set(pool))
    return pool


def get_real_module_composition(prev):
    assignment = prev["assignment"]
    sub = assignment[assignment["group"].astype(str) == MODULE_GROUP_FOR_PROFILE].copy()

    comp = {}
    for module_id, df in sub.groupby("module_id"):
        terms = df["term"].dropna().astype(str).tolist()
        comp[str(module_id)] = sorted(set(terms))

    return comp


# ============================================================
# 6. CORE METRICS
# ============================================================

def compute_stage_metrics_for_scores(module_scores_z, stage_df):
    """
    module_scores_z: modules x patients
    Return summary metrics and module-level stats.
    """
    common = sorted(set(module_scores_z.columns).intersection(set(stage_df["patient"])))
    st = stage_df.set_index("patient").loc[common, "stage_group"]

    early = st[st == "Early"].index.tolist()
    late = st[st == "Late"].index.tolist()

    rows = []
    for m in module_scores_z.index:
        x_early = module_scores_z.loc[m, early].dropna().astype(float)
        x_late = module_scores_z.loc[m, late].dropna().astype(float)

        if len(x_early) < 10 or len(x_late) < 10:
            p = np.nan
            delta = np.nan
        else:
            _, p = mannwhitneyu(x_late, x_early, alternative="two-sided")
            delta = float(x_late.mean() - x_early.mean())

        rows.append({
            "module_id": m,
            "n_early": len(x_early),
            "n_late": len(x_late),
            "delta_late_minus_early": delta,
            "p_stage": p,
            "D_stage": safe_neglog10(p),
            "abs_delta": abs(delta) if not pd.isna(delta) else np.nan,
        })

    module_stats = pd.DataFrame(rows)

    # Centroid metrics
    early_centroid = module_scores_z.loc[:, early].mean(axis=1)
    late_centroid = module_scores_z.loc[:, late].mean(axis=1)

    patient_rows = []
    for p in common:
        v = module_scores_z[p]
        dis_early = vector_distance(v.values, early_centroid.values)
        dis_late = vector_distance(v.values, late_centroid.values)
        sim_early = vector_corr(v.values, early_centroid.values)
        sim_late = vector_corr(v.values, late_centroid.values)

        patient_rows.append({
            "patient": p,
            "stage_group": st.loc[p],
            "Dis_early": dis_early,
            "Dis_late": dis_late,
            "Sim_early": sim_early,
            "Sim_late": sim_late,
            "late_minus_early_similarity": sim_late - sim_early if pd.notna(sim_late) and pd.notna(sim_early) else np.nan,
            "late_minus_early_distance": dis_late - dis_early if pd.notna(dis_late) and pd.notna(dis_early) else np.nan,
        })

    patient_metrics = pd.DataFrame(patient_rows)

    # How well similarity-to-late separates stage
    e = patient_metrics[patient_metrics["stage_group"] == "Early"]["Sim_late"].dropna()
    l = patient_metrics[patient_metrics["stage_group"] == "Late"]["Sim_late"].dropna()
    if len(e) >= 10 and len(l) >= 10:
        _, p_sim_late = mannwhitneyu(l, e, alternative="two-sided")
        delta_sim_late = float(l.mean() - e.mean())
    else:
        p_sim_late = np.nan
        delta_sim_late = np.nan

    # How well Dis_late separates stage: lower Dis_late should be late-like
    e = patient_metrics[patient_metrics["stage_group"] == "Early"]["Dis_late"].dropna()
    l = patient_metrics[patient_metrics["stage_group"] == "Late"]["Dis_late"].dropna()
    if len(e) >= 10 and len(l) >= 10:
        _, p_dis_late = mannwhitneyu(l, e, alternative="two-sided")
        delta_dis_late = float(l.mean() - e.mean())
    else:
        p_dis_late = np.nan
        delta_dis_late = np.nan

    summary = {
        "n_modules": int(module_scores_z.shape[0]),
        "n_patients": int(len(common)),
        "n_early": int(len(early)),
        "n_late": int(len(late)),
        "mean_module_D_stage": float(module_stats["D_stage"].mean()),
        "max_module_D_stage": float(module_stats["D_stage"].max()),
        "mean_abs_module_delta": float(module_stats["abs_delta"].mean()),
        "n_modules_p_le_0_05": int((module_stats["p_stage"] <= P_CUTOFF).sum()),
        "delta_Sim_late_late_minus_early": delta_sim_late,
        "p_Sim_late_stage": p_sim_late,
        "D_Sim_late_stage": safe_neglog10(p_sim_late),
        "delta_Dis_late_late_minus_early": delta_dis_late,
        "p_Dis_late_stage": p_dis_late,
        "D_Dis_late_stage": safe_neglog10(p_dis_late),
    }

    return summary, module_stats, patient_metrics


def zscore_scores_by_row(score_df):
    out = score_df.copy()
    for idx in out.index:
        out.loc[idx] = zscore_series(out.loc[idx])
    return out


# ============================================================
# 7. RANDOM BP MODULE TEST
# ============================================================

def build_module_scores_from_composition(bp_scores, module_comp):
    scores = {}
    for m, terms in module_comp.items():
        present = [t for t in terms if t in bp_scores.index]
        if not present:
            continue
        scores[m] = bp_scores.loc[present].mean(axis=0, skipna=True)

    out = pd.DataFrame(scores).T
    return zscore_scores_by_row(out)


def sample_random_module_composition(real_comp, candidate_pool, rng):
    """
    Keep the same module IDs and module sizes.
    Sample without replacement within each module.
    Across modules, reuse is allowed to match simple null.
    """
    candidate_pool = list(candidate_pool)
    random_comp = {}

    for m, terms in real_comp.items():
        k = len(terms)
        if k <= len(candidate_pool):
            sampled = rng.choice(candidate_pool, size=k, replace=False).tolist()
        else:
            sampled = rng.choice(candidate_pool, size=k, replace=True).tolist()
        random_comp[m] = sorted(set(sampled))

        # if set() shrinks because duplicate under replacement, fill back
        while len(random_comp[m]) < k:
            extra = rng.choice(candidate_pool, size=1, replace=True).tolist()[0]
            random_comp[m].append(extra)
            random_comp[m] = sorted(set(random_comp[m]))

    return random_comp


def run_random_bp_module_test(bp_scores_pool, real_comp, candidate_pool, stage_df, real_summary):
    rows = []

    for i in range(N_RANDOM_MODULES):
        rand_comp = sample_random_module_composition(real_comp, candidate_pool, rng)
        rand_scores_z = build_module_scores_from_composition(bp_scores_pool, rand_comp)

        if rand_scores_z.empty:
            continue

        summary, _, _ = compute_stage_metrics_for_scores(rand_scores_z, stage_df)
        summary["random_id"] = i
        rows.append(summary)

        if (i + 1) % 50 == 0:
            print(f"[Random BP modules] {i+1}/{N_RANDOM_MODULES}")

    null_df = pd.DataFrame(rows)

    compare_rows = []
    metrics = [
        "mean_module_D_stage",
        "max_module_D_stage",
        "mean_abs_module_delta",
        "n_modules_p_le_0_05",
        "D_Sim_late_stage",
        "D_Dis_late_stage",
    ]

    for metric in metrics:
        real_val = real_summary.get(metric, np.nan)
        null_vals = null_df[metric].values if metric in null_df.columns else []
        p_emp = empirical_p_greater_equal(real_val, null_vals)

        compare_rows.append({
            "test": "random_BP_modules",
            "metric": metric,
            "real_value": real_val,
            "null_mean": float(np.nanmean(null_vals)) if len(null_vals) else np.nan,
            "null_sd": float(np.nanstd(null_vals)) if len(null_vals) else np.nan,
            "null_p95": float(np.nanpercentile(null_vals, 95)) if len(null_vals) else np.nan,
            "null_p99": float(np.nanpercentile(null_vals, 99)) if len(null_vals) else np.nan,
            "empirical_p_greater_equal": p_emp,
        })

    compare_df = pd.DataFrame(compare_rows)
    return null_df, compare_df


# ============================================================
# 8. STAGE LABEL SHUFFLE TEST
# ============================================================

def shuffle_stage_labels(stage_df, rng):
    out = stage_df.copy()
    labels = out["stage_group"].values.copy()
    rng.shuffle(labels)
    out["stage_group"] = labels
    return out


def run_stage_label_shuffle_test(real_module_scores_z, stage_df, real_summary):
    rows = []

    for i in range(N_STAGE_SHUFFLE):
        shuf_stage = shuffle_stage_labels(stage_df, rng)
        summary, _, _ = compute_stage_metrics_for_scores(real_module_scores_z, shuf_stage)
        summary["shuffle_id"] = i
        rows.append(summary)

        if (i + 1) % 100 == 0:
            print(f"[Stage shuffle] {i+1}/{N_STAGE_SHUFFLE}")

    null_df = pd.DataFrame(rows)

    compare_rows = []
    metrics = [
        "mean_module_D_stage",
        "max_module_D_stage",
        "mean_abs_module_delta",
        "n_modules_p_le_0_05",
        "D_Sim_late_stage",
        "D_Dis_late_stage",
    ]

    for metric in metrics:
        real_val = real_summary.get(metric, np.nan)
        null_vals = null_df[metric].values if metric in null_df.columns else []
        p_emp = empirical_p_greater_equal(real_val, null_vals)

        compare_rows.append({
            "test": "stage_label_shuffle",
            "metric": metric,
            "real_value": real_val,
            "null_mean": float(np.nanmean(null_vals)) if len(null_vals) else np.nan,
            "null_sd": float(np.nanstd(null_vals)) if len(null_vals) else np.nan,
            "null_p95": float(np.nanpercentile(null_vals, 95)) if len(null_vals) else np.nan,
            "null_p99": float(np.nanpercentile(null_vals, 99)) if len(null_vals) else np.nan,
            "empirical_p_greater_equal": p_emp,
        })

    compare_df = pd.DataFrame(compare_rows)
    return null_df, compare_df


# ============================================================
# 9. PATIENT SCRAMBLE TEST
# ============================================================

def scramble_patients_per_module(module_scores_z, rng):
    out = module_scores_z.copy()
    patients = out.columns.tolist()

    for m in out.index:
        vals = out.loc[m, patients].values.copy()
        rng.shuffle(vals)
        out.loc[m, patients] = vals

    return out


def compute_patient_coactivity_metrics(module_scores_z):
    """
    Metrics for patient-level co-activity structure.
    """
    X = module_scores_z.T.fillna(0)

    # Mean absolute module-module correlation
    if X.shape[1] >= 2:
        corr = X.corr(method="spearman")
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        mean_abs_corr = float(np.nanmean(np.abs(upper.values)))
        max_abs_corr = float(np.nanmax(np.abs(upper.values)))
    else:
        mean_abs_corr = np.nan
        max_abs_corr = np.nan

    # PCA concentration
    if X.shape[0] >= 5 and X.shape[1] >= 2:
        Xs = StandardScaler().fit_transform(X.values)
        pca = PCA(n_components=min(3, Xs.shape[1]))
        pca.fit(Xs)
        pc1_var = float(pca.explained_variance_ratio_[0])
        pc2_var = float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else np.nan
    else:
        pc1_var = np.nan
        pc2_var = np.nan

    return {
        "mean_abs_module_module_corr": mean_abs_corr,
        "max_abs_module_module_corr": max_abs_corr,
        "PCA_PC1_variance": pc1_var,
        "PCA_PC2_variance": pc2_var,
    }


def run_patient_scramble_test(real_module_scores_z):
    real_coact = compute_patient_coactivity_metrics(real_module_scores_z)

    rows = []

    for i in range(N_PATIENT_SCRAMBLE):
        scrambled = scramble_patients_per_module(real_module_scores_z, rng)
        metrics = compute_patient_coactivity_metrics(scrambled)
        metrics["scramble_id"] = i
        rows.append(metrics)

        if (i + 1) % 50 == 0:
            print(f"[Patient scramble] {i+1}/{N_PATIENT_SCRAMBLE}")

    null_df = pd.DataFrame(rows)

    compare_rows = []
    for metric in [
        "mean_abs_module_module_corr",
        "max_abs_module_module_corr",
        "PCA_PC1_variance",
        "PCA_PC2_variance",
    ]:
        real_val = real_coact.get(metric, np.nan)
        null_vals = null_df[metric].values if metric in null_df.columns else []
        p_emp = empirical_p_greater_equal(real_val, null_vals)

        compare_rows.append({
            "test": "patient_scramble",
            "metric": metric,
            "real_value": real_val,
            "null_mean": float(np.nanmean(null_vals)) if len(null_vals) else np.nan,
            "null_sd": float(np.nanstd(null_vals)) if len(null_vals) else np.nan,
            "null_p95": float(np.nanpercentile(null_vals, 95)) if len(null_vals) else np.nan,
            "null_p99": float(np.nanpercentile(null_vals, 99)) if len(null_vals) else np.nan,
            "empirical_p_greater_equal": p_emp,
        })

    compare_df = pd.DataFrame(compare_rows)
    return null_df, compare_df, real_coact


# ============================================================
# 10. FIGURES
# ============================================================

def plot_null_distribution(null_df, compare_df, test_name, out_prefix):
    if null_df.empty or compare_df.empty:
        return

    for _, r in compare_df.iterrows():
        metric = r["metric"]
        if metric not in null_df.columns:
            continue

        vals = null_df[metric].dropna().values
        if len(vals) == 0:
            continue

        real_val = r["real_value"]

        plt.figure(figsize=(7, 5))
        plt.hist(vals, bins=35, alpha=0.8)
        plt.axvline(real_val, linestyle="--", linewidth=2)
        plt.xlabel(metric)
        plt.ylabel("Random count")
        plt.title(f"{test_name}: {metric}\nreal={real_val:.3g}, emp_p={r['empirical_p_greater_equal']:.4f}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{out_prefix}_{metric}.png", dpi=300)
        plt.close()


# ============================================================
# 11. MAIN
# ============================================================

def main():
    print("=" * 100)
    print("MOATTERS BRCA Patient-level BP Strategy RANDOM TEST V1")
    print("=" * 100)

    print("[Load] previous outputs")
    prev = load_previous_results()

    real_module_scores_z = prev["real_module_scores_z"]
    real_comp = get_real_module_composition(prev)
    candidate_pool = get_candidate_bp_pool(prev)

    print("Real module scores:", real_module_scores_z.shape)
    print("Real modules:", {k: len(v) for k, v in real_comp.items()})
    print("Candidate BP pool:", len(candidate_pool), "mode=", CANDIDATE_POOL_MODE)

    print("[Load] GE / stage / GMT")
    ge = load_ge_matrix(BRCA_GE_PATH)
    ge = ge.loc[[g for g in ge.index if g not in ["?", "", None]], :]

    stage_df = load_brca_stage()
    common_patients = sorted(set(ge.columns).intersection(set(stage_df["patient"])))

    ge = ge.loc[:, common_patients]
    stage_df = stage_df[stage_df["patient"].isin(common_patients)].copy()

    print("GE:", ge.shape)
    print(stage_df["stage_group"].value_counts().to_string())

    go_sets = load_gmt(GO_BP_GMT)

    # Compute BP scores for candidate pool once
    print("[Compute] BP scores for candidate pool")
    bp_scores_pool, bp_readiness = compute_gene_set_scores(
        ge,
        go_sets,
        candidate_pool,
        min_matched=MIN_MATCHED_GENES
    )

    bp_readiness.to_csv(OUT_DIR / "BRCA_random_test_candidate_BP_readiness.csv", index=False)
    print("BP score pool:", bp_scores_pool.shape)

    # Align real module score patients
    real_module_scores_z = real_module_scores_z.loc[:, common_patients]

    # Real metrics
    print("[Real metrics]")
    real_summary, real_module_stats, real_patient_metrics = compute_stage_metrics_for_scores(
        real_module_scores_z,
        stage_df
    )

    real_module_stats.to_csv(OUT_DIR / "BRCA_real_module_stage_stats.csv", index=False)
    real_patient_metrics.to_csv(OUT_DIR / "BRCA_real_patient_centroid_metrics.csv", index=False)

    with open(OUT_DIR / "BRCA_real_random_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(real_summary, f, indent=2)

    print(json.dumps(real_summary, indent=2))

    # Test 1: random BP modules
    print("\n" + "=" * 80)
    print("[TEST 1] Random BP modules")
    rand_module_null, rand_module_compare = run_random_bp_module_test(
        bp_scores_pool=bp_scores_pool,
        real_comp=real_comp,
        candidate_pool=bp_scores_pool.index.tolist(),
        stage_df=stage_df,
        real_summary=real_summary
    )

    rand_module_null.to_csv(OUT_DIR / "BRCA_random_BP_module_null_distribution.csv", index=False)
    rand_module_compare.to_csv(OUT_DIR / "BRCA_random_BP_module_compare.csv", index=False)

    if MAKE_FIGURES:
        plot_null_distribution(
            rand_module_null,
            rand_module_compare,
            "Random BP modules",
            "random_BP_modules"
        )

    # Test 2: stage-label shuffle
    print("\n" + "=" * 80)
    print("[TEST 2] Stage-label shuffle")
    stage_shuffle_null, stage_shuffle_compare = run_stage_label_shuffle_test(
        real_module_scores_z=real_module_scores_z,
        stage_df=stage_df,
        real_summary=real_summary
    )

    stage_shuffle_null.to_csv(OUT_DIR / "BRCA_stage_label_shuffle_null_distribution.csv", index=False)
    stage_shuffle_compare.to_csv(OUT_DIR / "BRCA_stage_label_shuffle_compare.csv", index=False)

    if MAKE_FIGURES:
        plot_null_distribution(
            stage_shuffle_null,
            stage_shuffle_compare,
            "Stage-label shuffle",
            "stage_label_shuffle"
        )

    # Test 3: patient scramble
    print("\n" + "=" * 80)
    print("[TEST 3] Patient scramble")
    patient_scramble_null, patient_scramble_compare, real_coact = run_patient_scramble_test(
        real_module_scores_z=real_module_scores_z
    )

    patient_scramble_null.to_csv(OUT_DIR / "BRCA_patient_scramble_null_distribution.csv", index=False)
    patient_scramble_compare.to_csv(OUT_DIR / "BRCA_patient_scramble_compare.csv", index=False)

    with open(OUT_DIR / "BRCA_real_patient_coactivity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(real_coact, f, indent=2)

    if MAKE_FIGURES:
        plot_null_distribution(
            patient_scramble_null,
            patient_scramble_compare,
            "Patient scramble",
            "patient_scramble"
        )

    # Combined compare table
    all_compare = pd.concat(
        [rand_module_compare, stage_shuffle_compare, patient_scramble_compare],
        ignore_index=True
    )
    all_compare.to_csv(OUT_DIR / "BRCA_RANDOM_TEST_all_compare.csv", index=False)

    summary = {
        "random_seed": RANDOM_SEED,
        "n_random_modules": N_RANDOM_MODULES,
        "n_stage_shuffle": N_STAGE_SHUFFLE,
        "n_patient_scramble": N_PATIENT_SCRAMBLE,
        "candidate_pool_mode": CANDIDATE_POOL_MODE,
        "n_candidate_BP": int(bp_scores_pool.shape[0]),
        "n_patients": int(len(common_patients)),
        "stage_counts": stage_df["stage_group"].value_counts().to_dict(),
        "real_summary": real_summary,
        "real_patient_coactivity": real_coact,
    }

    with open(OUT_DIR / "BRCA_RANDOM_TEST_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE]")
    print("Output:")
    print(OUT_DIR)
    print("\nCombined compare:")
    print(all_compare.to_string(index=False))


if __name__ == "__main__":
    main()
