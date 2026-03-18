# EdgeRAG

EdgeRAG is a **reproducibility-first, publication-facing refactor** of the local RAG experiment used in the paper *Resource-Constrained Evaluation of Quantized Local LLMs for Retrieval-Augmented Generation*.

The repository is intentionally conservative. It does **not** replace the original experiment with a new design. Instead, it reorganizes the already-run study into a cleaner package with stable entry points, preserved legacy wrappers, documented outputs, and paper-friendly artifact locations.

## What this repository evaluates

The experiment evaluates four pipelines under a fixed local-stack workflow:

- **P0** — retrieval-only
- **P1** — closed-book answering
- **P2** — standard retrieve-then-answer RAG
- **P3** — same-model query-rewrite RAG

The study is **generator-centric**. Retrieval still matters and is swept across multiple embedding models and retrieval depths, but the central question is how local quantized generators behave once lexical quality, grounding, latency, semantic adequacy, and runtime failures are measured together.

## Relation to the paper

This repository is the runnable companion to the paper’s workstation-scale evaluation story.

The manuscript describes a **single personal workstation** with:

- Windows 10 (version 10.0)
- AMD64 Family 25 Model 33 CPU, 32 physical cores / 32 threads
- 64.0 GB RAM
- one NVIDIA GeForce RTX 3090 GPU with 24.0 GB VRAM
- Ollama 0.17.1
- FAISS 1.13.2
- Python 3.11.14

The reported experiment workload required **more than 160 GPU-hours** of execution time on that local machine. The purpose of this repository is therefore not to claim universal model rankings, but to make this exact local-stack workflow inspectable, repeatable, and extendable.

## What is intentionally preserved

The refactor preserves the parts of the original experiment that matter for scientific continuity:

- P0 / P1 / P2 / P3 semantics
- prompt text and prompt roles
- timeout defaults and watchdog logic
- how failures are recorded in `results.jsonl`
- resume-state hashing and progression logic
- base lexical and provenance-aware metrics
- reduced-KILT workflow structure
- optional SAS as an extension rather than a base dependency

## Quickstart

Install the runner and base-analysis dependencies:

```bash
pip install -e ".[runner,analysis]"
```

Start Ollama locally, then run the canonical paper-style command:

```bash
python -m edgerag.cli.run   --config configs/phase1.json   --verbose_stream   --first_token_timeout_s 300   --stream_timeout_s 600
```

Regenerate the base analysis:

```bash
python -m edgerag.analysis.base   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/base_analysis
```

Add SAS only when needed:

```bash
pip install -e ".[runner,analysis,sas]"
python -m edgerag.analysis.sas   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/sas_analysis
```

## Repository layout

```text
.
├── README.md
├── CITATION.cff
├── LICENSE-TODO.txt
├── pyproject.toml
├── configs/
├── docs/
├── artifacts/
├── legacy/
├── src/edgerag/
└── tests/
```

Key locations:

- `configs/phase1.json` — canonical stable config
- `src/edgerag/pipelines/runner.py` — main experiment orchestration
- `src/edgerag/analysis/base.py` — base analysis
- `src/edgerag/analysis/sas.py` — optional SAS extension
- `artifacts/paper/` — paper-facing tracked artifacts and sample output locations
- `legacy/original_snapshot/` — preserved uploaded originals for audit comparison

## Installation

Core package only:

```bash
pip install -e .
```

Experiment runner + base analysis:

```bash
pip install -e ".[runner,analysis]"
```

Add optional SAS:

```bash
pip install -e ".[runner,analysis,sas]"
```

Add test tooling:

```bash
pip install -e ".[dev]"
```

## External software

The main experiment runner expects a local Ollama server. This dependency is deliberate and part of the scientific setup rather than an implementation detail.

## Canonical paper-style workflow

### 1. Inspect the stable config

The canonical config is:

```text
configs/phase1.json
```

It preserves the uploaded phase-1 settings, including:

- 500 deterministic KILT-NQ development questions
- `subset_mode=random`
- `subset_seed=123`
- reduced `gold_plus_random` KB construction
- 100,000 random background passages plus all gold provenance passages
- three embedders
- twelve generator models in the current snapshot
- first-token timeout of 300 seconds
- stream timeout of 600 seconds

### 2. Resolve the plan safely before running

```bash
python -m edgerag.cli.run --config configs/phase1.json --dry_run
```

Dry-run now avoids Ollama execution, FAISS building, and KILT downloads. If the local KILT files are already present, it will also resolve the exact sampled question count and KB tag. If the local KILT files are absent, it reports the planned paths and configuration without downloading the dataset.

### 3. Run the experiment

```bash
python -m edgerag.cli.run   --config configs/phase1.json   --verbose_stream   --first_token_timeout_s 300   --stream_timeout_s 600
```

Installed console-script equivalent:

```bash
edgerag-run   --config configs/phase1.json   --verbose_stream   --first_token_timeout_s 300   --stream_timeout_s 600
```

### 4. Regenerate the base analysis outputs

```bash
python -m edgerag.analysis.base   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/base_analysis
```

### 5. Regenerate SAS outputs only when needed

```bash
python -m edgerag.analysis.sas   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/sas_analysis
```

Skip SAS scoring but keep a SAS-shaped output path:

```bash
python -m edgerag.analysis.sas   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/sas_analysis   --skip_sas
```

### 6. Rebuild resume state only if needed

```bash
python -m edgerag.cli.rebuild_resume --config configs/phase1.json
```

### 7. Estimate runtime from live stream logs

```bash
python -m edgerag.cli.estimate_runtime --results_dir results_phase1
```

## Expected outputs

The canonical run writes to the configured `results_dir` (default: `results_phase1/`). Important files include:

- `results_phase1/results.jsonl` — raw trial records, including failures
- `results_phase1/resume_state.json` — resume checkpoints keyed by run configuration
- `results_phase1/run_metadata.json` — sidecar runtime metadata for resumability and reporting
- `results_phase1/live_streams/` — verbose token-stream traces when `--verbose_stream` is enabled
- `results_phase1/runtime_estimate.json` — optional post-hoc session estimate from live stream timestamps

Base analysis writes publication-style exports such as:

- canonicalized JSONL
- CSV summary tables
- publication CSV tables
- `publication_tables.xlsx`
- figure PNG / SVG / PDF files
- `summary.txt`

See `docs/RESULTS.md` for a fuller output map.

## Tracked artifacts vs regenerated outputs

Tracked under `artifacts/paper/`:

- manuscript snapshots
- uploaded spreadsheets
- uploaded PDF reports
- sample output locations for regenerated analysis

Regenerated locally:

- results JSONL
- resume state
- runtime metadata
- live stream logs
- runtime estimates
- regenerated tables and figures
- KB SQLite files and FAISS indices
- downloaded KILT corpora

## Backward compatibility and migration

The following old root-level filenames remain as thin wrappers:

- `edge_rag_experiment_fix10.py`
- `edge_rag_final_analysis_v2_5.py`
- `edge_rag_final_analysis_v2_7_sas.py`
- `rebuild_resume_from_results.py`
- `estimate_gpu_runtime.py`
- `phase1_config_fix9_v4.json`

They are preserved for migration and reproducibility, while the stable public-facing paths live under `src/`, `configs/`, and `docs/`.

## Scope and limitations

This is a **local-stack** evaluation repository. Interpret the reported results as specific to:

- the reduced KILT-NQ setup
- the local Ollama serving stack
- the selected timeout policy
- the selected generators and embedders
- the single-workstation hardware budget

The rankings are not meant to be universal.

## Troubleshooting

- If generation repeatedly stalls, restart `ollama serve` and rerun the same command.
- If a run stopped mid-sweep, rerun the same command first; use the rebuild utility only when `resume_state.json` and `results.jsonl` drift apart.
- If base analysis works but SAS fails, install the optional SAS dependencies and rerun.
- If you want to validate the resolved plan without touching Ollama or FAISS, use `--dry_run`.
- If `--dry_run` reports missing local KILT files, that is expected when the dataset has not been downloaded yet.

## Citation and license status for this snapshot

The manuscript files provided for the refactor use blinded author placeholders, so `CITATION.cff` keeps anonymized review metadata rather than inventing names. The source files also did not specify a final project license, so `LICENSE-TODO.txt` remains intentionally conservative until the authors choose a release license.

## Further documentation

- `docs/REPRODUCIBILITY.md` — step-by-step paper-style workflow
- `docs/COMMANDS.md` — command reference with examples
- `docs/PROJECT_STRUCTURE.md` — module-level layout explanation
- `docs/RESULTS.md` — output files and artifact expectations
- `docs/MIGRATION.md` — old-to-new command mapping
