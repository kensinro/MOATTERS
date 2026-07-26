# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
from pathlib import Path
import json
import pandas as pd

# ============================================================
# Stage 5B-2D — Finalize Pathifier outputs without rerunning
# The expensive Pathifier computation is already complete.
# This cell only validates files and writes analysis_manifest.json.
# ============================================================

OUTPUT_DIR = Path(
    r"D:\MOATTERS-Output\MOATTERS_STAGE5B_GSVA_PATHIFIER_SCORES"
)

INPUT_DIR = Path(
    r"D:\MOATTERS-Output\MOATTERS_STAGE5B_BENCHMARK_INPUTS"
)

gsva_file = OUTPUT_DIR / "GSVA_scores_sample_by_BP.csv"
pathifier_file = OUTPUT_DIR / "Pathifier_scores_sample_by_BP.csv"
rds_file = OUTPUT_DIR / "Pathifier_full_result_attempts20.rds"
structure_file = OUTPUT_DIR / "Pathifier_object_structure.txt"
sample_manifest_file = INPUT_DIR / "sample_manifest.csv"
manifest_file = OUTPUT_DIR / "analysis_manifest.json"

required = [
    gsva_file,
    pathifier_file,
    rds_file,
    structure_file,
    sample_manifest_file,
]

missing = [str(p) for p in required if not p.exists()]

if missing:
    raise FileNotFoundError(
        "Missing required file(s):\n" + "\n".join(missing)
    )

gsva = pd.read_csv(gsva_file, index_col=0)
pathifier = pd.read_csv(pathifier_file, index_col=0)
sample_manifest = pd.read_csv(sample_manifest_file)

def as_bool_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
    )

primary = as_bool_series(sample_manifest["is_primary_tumor"])
normal = as_bool_series(sample_manifest["is_adjacent_normal"])

if primary.isna().any():
    raise ValueError(
        "Could not parse all is_primary_tumor values as Boolean."
    )

if normal.isna().any():
    raise ValueError(
        "Could not parse all is_adjacent_normal values as Boolean."
    )

if gsva.shape[1] != 30:
    raise ValueError(
        f"GSVA matrix has {gsva.shape[1]} pathways; expected 30."
    )

if pathifier.shape[1] != 30:
    raise ValueError(
        f"Pathifier matrix has {pathifier.shape[1]} pathways; expected 30."
    )

if gsva.shape[0] != pathifier.shape[0]:
    raise ValueError(
        f"Sample count mismatch: GSVA={gsva.shape[0]}, "
        f"Pathifier={pathifier.shape[0]}"
    )

manifest = {
    "status": "PASS",
    "n_gene_sets_requested": 30,
    "n_pathways_processed": int(pathifier.shape[1]),
    "n_samples": int(pathifier.shape[0]),
    "n_primary_tumors": int(primary.sum()),
    "n_adjacent_normals": int(normal.sum()),
    "Pathifier_attempts": 20,
    "Pathifier_maximize_stability": True,
    "GSVA_file_present": True,
    "GSVA_shape_sample_by_pathway": list(gsva.shape),
    "Pathifier_shape_sample_by_pathway": list(pathifier.shape),
    "full_Pathifier_object_saved": True,
    "full_Pathifier_object_file": rds_file.name,
    "Pathifier_structure_file": structure_file.name,
}

manifest_file.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("Final output check:")
for p in [
    gsva_file,
    pathifier_file,
    manifest_file,
    rds_file,
    structure_file,
]:
    print(f"{p.name}: {'FOUND' if p.exists() else 'MISSING'}")

print()
print("GSVA shape:", gsva.shape)
print("Pathifier shape:", pathifier.shape)
print("Primary tumors:", int(primary.sum()))
print("Adjacent normals:", int(normal.sum()))
print()
print("PASS — Stage 5B-2 is complete. Proceed to benchmark evaluation.")
