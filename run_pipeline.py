#!/usr/bin/env python3
"""Run or inspect the ordered MOATTERS workflow.

This runner preserves the individual locked scripts while providing one
portable entry point. It sets the repository root on PYTHONPATH so scripts can
import ``moatters.config`` without an editable install.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

STAGES = {
    "derivation": [
        "scripts/01_derivation/02_stage_attractor_reconstruction.py",
        "scripts/01_derivation/01_bp_module_rewiring.py",
        "scripts/01_derivation/03_tme_conditioned_module_analysis.py",
        "scripts/01_derivation/04_patient_state_profile.py",
        "scripts/01_derivation/05_random_control_audit.py",
        "scripts/01_derivation/06_downstream_validation.py",
        "scripts/02_reference/01_tcga_brca_reference_audit.py",
    ],
    "metabric": [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts/03_external_validation/metabric").glob("*.py"))],
    "gse96058": [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts/03_external_validation/gse96058").glob("*.py"))],
    "kirc": [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts/04_cross_cancer/kirc").glob("*.py"))],
    "luad": [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts/04_cross_cancer/luad").glob("*.py"))],
    "synthesis": [
        "scripts/05_synthesis/01_three_cohort_synthesis.py",
        "scripts/05_synthesis/02_joint_evidence_synthesis.py",
    ],
    "singleton": ["scripts/06_audits/01_singleton_component_audit.py"],
    "benchmark": [
        "scripts/07_benchmark/01_prepare_benchmark_inputs.py",
        "scripts/07_benchmark/02_run_gsva_pathifier.py",
        "scripts/07_benchmark/03_evaluate_benchmark.py",
    ],
}
STAGES["all"] = sum((STAGES[k] for k in ["derivation", "metabric", "gse96058", "kirc", "luad", "synthesis", "singleton", "benchmark"]), [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    failures = []
    for rel in STAGES[args.stage]:
        cmd = [sys.executable, str(ROOT / rel)]
        print("$", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        if result.returncode:
            failures.append((rel, result.returncode))
            if not args.continue_on_error:
                break
    if failures:
        for rel, code in failures:
            print(f"FAILED ({code}): {rel}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
