# Project structure

## Top level

- `README.md` — project overview and primary workflows
- `CITATION.cff` — citation metadata for the repository
- `pyproject.toml` — packaging, dependencies, and console scripts
- `configs/` — stable JSON configs
- `docs/` — reproducibility, outputs, structure, and migration docs
- `artifacts/` — paper-facing supporting artifacts and sample output locations
- `legacy/original_snapshot/` — preserved uploaded source snapshots for auditability
- `tests/` — lightweight smoke and unit tests

## Package modules

### `src/edgerag/core/`

Low-level shared helpers used by the runner and analysis:

- `config.py` — config loading, defaults, and CLI/config resolution helpers
- `metrics.py` — lexical, retrieval, and think-tag scoring helpers
- `prompts.py` — preserved prompt text
- `resume.py` — `RunKey` and resume-state helpers
- `results.py` — failed-trial record writer
- `runtime.py` — timing, budget checks, runtime-session estimation, and sidecar run metadata helpers
- `utils.py` — atomic IO, JSONL helpers, hashing, and filesystem utilities

### `src/edgerag/data/`

- `kilt.py` — KILT download and question-subset loading
- `kb.py` — SQLite KB builders, validators, and passage fetching

### `src/edgerag/retrieval/`

- `embeddings.py` — Ollama vs sentence-transformers embedding helpers
- `faiss_index.py` — FAISS build, load, and retrieval logic

### `src/edgerag/llm/`

- `ollama_client.py` — local Ollama API client, retries, and stream watchdog handling

### `src/edgerag/pipelines/`

- `runner.py` — orchestration for P0, P1, P2, and P3 under the preserved local-stack semantics

### `src/edgerag/analysis/`

- `common.py` — shared path and config-resolution helpers for analysis entry points
- `base.py` — base publication analysis path
- `sas.py` — optional SAS extension layer built on top of the packaged base analysis module

### `src/edgerag/cli/`

- `run.py` — stable experiment entry point
- `analyze.py` — base-analysis wrapper
- `analyze_sas.py` — SAS-analysis wrapper
- `rebuild_resume.py` — resume rebuild utility
- `estimate_runtime.py` — log-derived GPU runtime estimator
