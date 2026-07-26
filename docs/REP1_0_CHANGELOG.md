# MOATTERS REP1.0 release-hardening changelog

Date: 2026-07-25

## Changes completed without rerunning scientific analyses

1. Corrected `docs/DATA_AND_INPUTS.md` so it reflects the active centralized configuration architecture rather than obsolete per-script hard-coded path blocks.
2. Hardened `moatters/config.py`:
   - validates that the JSON configuration is an object;
   - reports the active configuration path;
   - resolves relative JSON paths from the repository root;
   - preserves conventional current-working-directory resolution for relative environment-variable paths.
3. Replaced the binary preflight with tiered scopes:
   - `structure`;
   - `core`;
   - `benchmark`;
   - `full`.
4. Added explicit output-directory write testing.
5. Added Rscript resolution and optional direct checks for the `GSVA` and `pathifier` namespaces.
6. Restored fresh-run benchmark completeness: the active R stage now generates locked GSVA scores when absent, retains an existing locked GSVA file when present, and then executes the corrected Pathifier extraction.
7. Added component-level PASS/FAIL/NOT_REQUIRED states and retained machine-readable JSON output and shell-safe exit codes.
8. Updated README, configuration documentation, input documentation, and release checklist to match the revised behaviour.
9. Re-ran Python compilation, static release audit, runner dry-run, cross-working-directory configuration checks, and manifest generation.

## Not completed in REP1.0 hardening

- cohort-level numerical reruns;
- expected-versus-observed metric comparison;
- R package execution against the locked benchmark inputs;
- final manuscript/repository/Zenodo DOI lock;
- final software-license selection.

These items require the locked external datasets and/or the designated R environment.
