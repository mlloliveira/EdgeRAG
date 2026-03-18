# Results and outputs

This repository separates **tracked paper-supporting artifacts** from **regenerable experiment outputs**.

## Runtime outputs from the experiment runner

The canonical config writes runtime outputs to `results_phase1/`.

Important files:

- `results.jsonl` — one row per evaluated trial, including failures
- `resume_state.json` — hashed resume checkpoints per run configuration
- `run_metadata.json` — sidecar runtime metadata such as start/update timestamps, elapsed runtime, active phase, and resolved run context
- `live_streams/` — optional streamed token traces when `--verbose_stream` is enabled
- `runtime_estimate.json` — optional post-hoc session estimate derived from live stream timestamps

## Base analysis outputs

Typical base analysis outputs include:

- `canonical_results.jsonl`
- summary CSVs such as retrieval and per-generator tables
- publication CSV tables
- `publication_tables.xlsx`
- figure exports in PNG / SVG / PDF
- `summary.txt`

## SAS analysis outputs

The SAS path adds semantic-scoring outputs on top of the base analysis shape, including:

- `canonical_results_with_sas.jsonl`
- SAS summary CSVs
- SAS publication tables
- SAS-specific figures
- `summary.txt`

## Tracked paper-supporting artifacts

Under `artifacts/paper/` the repository keeps project files that support the paper narrative but are not the normal live outputs of the CLI:

- `manuscript/` — paper snapshots and bibliography material
- `spreadsheets/` — uploaded result workbooks
- `reports/` — uploaded PDF report snapshots
- `tables/` and `figures/` — reserved locations for curated paper assets
- `sample_outputs/` — suggested home for regenerated analysis examples

## What should be committed vs regenerated

Usually commit:

- configs
- docs
- source code
- preserved legacy snapshots
- tracked paper-supporting artifacts
- lightweight sample outputs only when useful for illustration

Usually regenerate locally instead of committing:

- full `results_phase1/` experiment outputs
- FAISS indices
- KB SQLite databases
- downloaded KILT corpora
- large temporary logs and caches
