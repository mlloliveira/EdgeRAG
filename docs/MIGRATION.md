# Migration notes

## Old names to new entry points

| Old file / command | New canonical entry point |
|---|---|
| `python edge_rag_experiment_fix10.py --config phase1_config_fix9_v4.json ...` | `python -m edgerag.cli.run --config configs/phase1.json ...` |
| `python edge_rag_final_analysis_v2_5.py ...` | `python -m edgerag.analysis.base --config configs/phase1.json ...` |
| `python edge_rag_final_analysis_v2_7_sas.py ...` | `python -m edgerag.analysis.sas --config configs/phase1.json ...` |
| `python rebuild_resume_from_results.py` | `python -m edgerag.cli.rebuild_resume --config configs/phase1.json` |
| `python estimate_gpu_runtime.py` | `python -m edgerag.cli.estimate_runtime --results_dir results_phase1` |

## Compatibility preserved

- The old Python filenames remain in the repository root as thin deprecation wrappers.
- The old config filename `phase1_config_fix9_v4.json` is preserved as a compatibility copy.
- The original uploaded scripts remain under `legacy/original_snapshot/` for audit comparison.

## Output locations

- Runtime outputs are expected under the configured `results_dir`, typically `results_phase1/`.
- Regenerated analysis outputs should go under `artifacts/paper/sample_outputs/`.
- Tracked paper-supporting spreadsheets and manuscript files live under `artifacts/paper/`.
