# Reproducibility workflow

This document gives the step-by-step workflow for reproducing the local-stack experiment and regenerating the paper-facing outputs.

## 1. Install dependencies

For the main runner and base-analysis path:

```bash
pip install -e ".[runner,analysis]"
```

Add SAS only when needed:

```bash
pip install -e ".[runner,analysis,sas]"
```

## 2. Start Ollama locally

Confirm that the Ollama server is running and reachable at the configured URL.

Default URL:

```text
http://localhost:11434
```

## 3. Use the stable config

The canonical paper-style configuration is:

```text
configs/phase1.json
```

It preserves the uploaded phase-1 setup:

- deterministic subset of 500 KILT-NQ dev questions
- `subset_mode=random`
- `subset_seed=123`
- reduced `gold_plus_random` KB construction
- 100,000 random background passages plus all gold provenance passages
- three embedders
- twelve generators in the current snapshot
- first-token timeout of 300 seconds
- stream timeout of 600 seconds

## 4. Validate the plan without touching Ollama

```bash
python -m edgerag.cli.run --config configs/phase1.json --dry_run
```

Dry-run is intended as a safe planning step. It does not execute Ollama, build FAISS, or download KILT. If the local KILT files already exist, it also resolves the exact sampled question count and KB tag.

## 5. Run the main experiment

```bash
python -m edgerag.cli.run   --config configs/phase1.json   --verbose_stream   --first_token_timeout_s 300   --stream_timeout_s 600
```

Expected runtime outputs in `results_phase1/`:

- `results.jsonl`
- `resume_state.json`
- `run_metadata.json`
- `live_streams/` when verbose streaming is enabled

## 6. Rebuild resume state only if needed

Use this only when `results.jsonl` is present but `resume_state.json` is missing or stale.

```bash
python -m edgerag.cli.rebuild_resume --config configs/phase1.json
```

## 7. Run base analysis

```bash
python -m edgerag.analysis.base   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/base_analysis
```

This generates canonicalized results, summary CSVs, publication tables, figure files, and a text summary.

## 8. Run SAS analysis only when desired

```bash
python -m edgerag.analysis.sas   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/sas_analysis
```

SAS is optional. The base analysis path does not require SAS-only dependencies.

## 9. Estimate runtime from live stream logs

```bash
python -m edgerag.cli.estimate_runtime --results_dir results_phase1
```

This writes `results_phase1/runtime_estimate.json` and does not mutate `resume_state.json`.

## 10. Interpreting reproducibility claims

This repository is reproducible in the sense intended by the paper:

- same config and prompts
- same local retrieval and generation workflow
- same timeout logic
- same failure accounting
- same resume semantics
- same analysis exports from the same `results.jsonl`

It is **not** a claim that another machine, another Ollama build, another quantized checkpoint revision, or another driver stack will necessarily reproduce numerically identical results.
