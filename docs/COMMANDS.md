# Commands

## Install

Core package only:

```bash
pip install -e .
```

Runner + base analysis:

```bash
pip install -e ".[runner,analysis]"
```

Add optional SAS:

```bash
pip install -e ".[runner,analysis,sas]"
```

Add tests:

```bash
pip install -e ".[dev]"
```

## Experiment runner

Minimal canonical run:

```bash
python -m edgerag.cli.run --config configs/phase1.json
```

This downloads KILT if needed, constructs or reuses the KB and FAISS indices, and writes runtime outputs under the configured `results_dir`.

Paper-style run with verbose stream logging and explicit watchdog overrides:

```bash
python -m edgerag.cli.run   --config configs/phase1.json   --verbose_stream   --first_token_timeout_s 300   --stream_timeout_s 600
```

Dry-run only:

```bash
python -m edgerag.cli.run --config configs/phase1.json --dry_run
```

Dry-run does **not** execute Ollama, build FAISS, or download KILT. If local KILT files are present, it resolves the exact sampled question count; otherwise it reports the expected paths and planned configuration.

## Base analysis

Uses `results_phase1/results.jsonl` by default through the canonical config:

```bash
python -m edgerag.analysis.base --config configs/phase1.json
```

Explicit outputs:

```bash
python -m edgerag.analysis.base   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/base_analysis
```

This generates canonicalized results, summary CSVs, publication tables, figures, and `summary.txt`.

## SAS analysis

```bash
python -m edgerag.analysis.sas   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/sas_analysis
```

Skip SAS scoring but preserve the SAS-shaped output structure:

```bash
python -m edgerag.analysis.sas   --config configs/phase1.json   --results results_phase1/results.jsonl   --outdir artifacts/paper/sample_outputs/sas_analysis   --skip_sas
```

## Resume rebuild

```bash
python -m edgerag.cli.rebuild_resume --config configs/phase1.json
```

With explicit files:

```bash
python -m edgerag.cli.rebuild_resume   --config configs/phase1.json   --results results_phase1/results.jsonl   --resume results_phase1/resume_state.json
```

This reconstructs hashed progress checkpoints from `results.jsonl` without changing the result rows themselves.

## Runtime estimator

```bash
python -m edgerag.cli.estimate_runtime --results_dir results_phase1
```

Explicit folders:

```bash
python -m edgerag.cli.estimate_runtime   --results_dir results_phase1   --folders results_phase1/live_streams results_phase1/live_streams_0
```

This writes `runtime_estimate.json` as a sidecar and does not mutate the resume schema.

## Legacy wrappers

The old filenames still work as deprecation wrappers, for example:

```bash
python edge_rag_experiment_fix10.py --config phase1_config_fix9_v4.json --verbose_stream
```
