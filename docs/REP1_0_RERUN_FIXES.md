# MOATTERS REP1.0 rerun repairs

This release incorporates defects observed during the 2026-07-26 clean rerun.

## Blocking defects repaired

1. Corrected derivation order: attractor reconstruction now precedes module rewiring.
2. Attractor outputs now use the configured output root.
3. Added the Hallmark GMT to preflight input checks.
4. Updated Matplotlib boxplot argument from `labels` to `tick_labels`.
5. METABRIC locked coverage now consumes active portable outputs; V2/V3 equivalence is reported as not tested when V2 was not generated.
6. Explicitly aligned KIRC and LUAD eligibility masks to the results index.
7. Singleton audit now consumes current portable outputs instead of a historical ZIP.
8. Benchmark preparation now consumes current portable outputs instead of a historical ZIP.
9. Benchmark R-process output is relayed using UTF-8-safe Python streams.
10. R manifest generation explicitly coerces textual sample flags to booleans.
11. Downstream random-control summaries recognize directional empirical-p column names.

## Non-blocking improvements

- Benchmark numeric conversion is assembled in one block to avoid DataFrame fragmentation warnings.
- The package version is `1.0.0`; supported Python is explicitly constrained to 3.10–3.13.

## Rerun results observed before release locking

- TCGA-BRCA derivation/reference: PASS.
- METABRIC external validation: PASS, N=1980.
- GSE96058 external validation: PASS, N=3069.
- KIRC cross-cancer demonstration: PASS, held-out AUC approximately 0.713.
- LUAD cross-cancer demonstration: PASS, held-out AUC approximately 0.659.
- Singleton audit: PASS.
- GSVA/Pathifier benchmark: PASS.
