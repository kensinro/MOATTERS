# Configuration

Active scripts import paths from `moatters.config`. Configuration precedence is:

1. explicit environment variables;
2. the JSON file named by `MOATTERS_CONFIG`;
3. repository-local `config/moatters_config.json`;
4. repository-local `data/` and `outputs/` defaults.

## JSON configuration

Copy `config/moatters_config.example.json` to `config/moatters_config.json` and edit only the values:

```json
{
  "data_root": "D:/MOATTERS-Data",
  "output_root": "D:/MOATTERS-Output",
  "rscript": "C:/Program Files/R/R-4.6.0/bin/Rscript.exe"
}
```

Relative JSON paths are resolved from the repository root, not from the current working directory.

## Environment variables

PowerShell example:

```powershell
$env:MOATTERS_DATA_ROOT = "D:\AIDO-Data"
$env:MOATTERS_OUTPUT_ROOT = "D:\MOATTERS-Output"
$env:MOATTERS_RSCRIPT = "C:\Program Files\R\R-4.6.0\bin\Rscript.exe"
python tools/preflight.py --scope full
```

Optional custom configuration file:

```powershell
$env:MOATTERS_CONFIG = "D:\configs\moatters_config.json"
python tools/preflight.py --scope full
```

The local directory name may differ from the public method name; only the configured paths matter.

## Preflight scopes

- `structure`: Python dependencies and writable output directory.
- `core`: structure checks plus all declared cohort and GMT inputs.
- `benchmark`: structure checks plus Rscript, GSVA, and Pathifier.
- `full`: core and benchmark checks together; this is the default.

`preflight.py` returns exit code `0` on PASS and `2` on INCOMPLETE, making it suitable for shell scripts and CI.
