# GitHub and Zenodo release checklist

## Identity and licensing

- [ ] Confirm the final public method name and manuscript title.
- [ ] Replace placeholders in `CITATION.cff`.
- [ ] Choose and add a software license.
- [ ] Add the final article citation and DOI when available.

## Reproducibility readiness

- [ ] Configure `data_root`, `output_root`, and `rscript` without editing active scripts.
- [ ] Run `python tools/preflight.py --scope structure`.
- [ ] Run `python tools/preflight.py --scope core` after mounting the locked inputs.
- [ ] Run `python tools/preflight.py --scope benchmark` before GSVA/Pathifier execution.
- [ ] Record Python, R, Bioconductor, GSVA, and Pathifier versions.
- [ ] Record input filenames, source versions, sample denominators, and checksums.
- [ ] Run the required locked stages and compare machine-readable outputs with expected results.
- [ ] Classify each comparison as EXACT, WITHIN_TOLERANCE, EXPECTED_ENVIRONMENTAL_DIFFERENCE, MATERIAL_MISMATCH, or NOT_REPRODUCIBLE.
- [ ] Lock the final result manifest and discrepancy registry.

## Public release

- [ ] Create a tagged GitHub release matching the submitted revision.
- [ ] Archive that exact tag on Zenodo.
- [ ] Record the Zenodo DOI in the manuscript, README, and `CITATION.cff`.
- [ ] Verify that no raw patient-level data or restricted files are committed.
- [ ] Verify that no private paths, credentials, email tokens, or internal project names remain.
- [ ] Run `python tools/static_release_audit.py`.
- [ ] Regenerate `MANIFEST.tsv` after the final edit and verify every checksum.
- [ ] Attach machine-readable result tables only when redistribution is permitted.
