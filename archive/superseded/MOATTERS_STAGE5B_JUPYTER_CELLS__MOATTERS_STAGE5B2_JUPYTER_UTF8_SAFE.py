# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
from pathlib import Path
import subprocess
import os
import time

# ============================================================
# Stage 5B-2 — Run GSVA and Pathifier from Jupyter
# UTF-8 / Windows console-safe version
# ============================================================

rscript = r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe"

r_file = Path(
    r"D:\MOATTERS-Output\MOATTERS_STAGE5B2_RUN_GSVA_PATHIFIER.R"
)

if not Path(rscript).exists():
    raise FileNotFoundError(f"Rscript not found: {rscript}")

if not r_file.exists():
    raise FileNotFoundError(
        f"R script not found: {r_file}\n"
        "Please run the previous complete Stage 5B-2 Jupyter cell once so the R file is created."
    )

print("Rscript:", rscript)
print("R file:", r_file)
print("Running GSVA and Pathifier...")
print("Note: non-ASCII console bytes will be replaced safely.")

# Verify Rscript.
version_check = subprocess.run(
    [rscript, "--version"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=60,
)

print("Rscript return code:", version_check.returncode)
if version_check.stdout:
    print(version_check.stdout)
if version_check.stderr:
    print(version_check.stderr)

if version_check.returncode != 0:
    raise RuntimeError("Rscript could not be executed.")

# Run R and decode output safely as UTF-8.
process = subprocess.Popen(
    [rscript, str(r_file)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)

for line in iter(process.stdout.readline, ""):
    if line == "" and process.poll() is not None:
        break
    print(line, end="")

return_code = process.wait()

if return_code != 0:
    raise RuntimeError(
        f"Stage 5B-2 failed with exit code {return_code}"
    )

print("PASS — Stage 5B-2 completed successfully")

output_dir = Path(
    r"D:\MOATTERS-Output\MOATTERS_STAGE5B_GSVA_PATHIFIER_SCORES"
)

print("Output folder:", output_dir)

expected = [
    output_dir / "GSVA_scores_sample_by_BP.csv",
    output_dir / "Pathifier_scores_sample_by_BP.csv",
    output_dir / "analysis_manifest.json",
]

for p in expected:
    print(f"{p.name}: {'FOUND' if p.exists() else 'MISSING'}")
