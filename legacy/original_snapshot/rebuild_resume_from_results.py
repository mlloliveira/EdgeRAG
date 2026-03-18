from pathlib import Path
import json
import shutil
import importlib.util
import sys

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
EXPERIMENT_SCRIPT = "edge_rag_experiment_fix9.py"
CONFIG_FILE = "phase1_config_fix9_v4.json"

# --------------------------------------------------
# LOAD EXPERIMENT MODULE
# --------------------------------------------------
spec = importlib.util.spec_from_file_location("edge_fix9", EXPERIMENT_SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["edge_fix9"] = mod
spec.loader.exec_module(mod)

# --------------------------------------------------
# SMALL CONFIG HELPER
# --------------------------------------------------
def cfg_get(cfg: dict, keys, default):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default

# --------------------------------------------------
# LOAD CONFIG
# --------------------------------------------------
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = json.load(f)

data_dir = Path(cfg_get(cfg, ["data_dir"], "data"))
results_dir = Path(cfg_get(cfg, ["results_dir"], "results_phase1"))
split = cfg_get(cfg, ["split"], "dev")
max_questions = int(cfg_get(cfg, ["max_questions"], 500))
subset_mode = cfg_get(cfg, ["subset_mode"], "random")
seed = int(cfg_get(cfg, ["seed"], 123))

results_path = results_dir / "results.jsonl"
resume_path = results_dir / "resume_state.json"

if not results_path.exists():
    raise FileNotFoundError(f"Could not find results file: {results_path}")

print(f"[info] data_dir      = {data_dir}")
print(f"[info] results_dir   = {results_dir}")
print(f"[info] split         = {split}")
print(f"[info] max_questions = {max_questions}")
print(f"[info] subset_mode   = {subset_mode}")
print(f"[info] seed          = {seed}")

# --------------------------------------------------
# LOAD THE SAME QUESTION SUBSET USED BY THE RUN
# --------------------------------------------------
nq_path, _ = mod.ensure_kilt_nq(data_dir, split=split)
examples = mod.load_kilt_nq_examples(
    nq_path,
    max_questions=max_questions,
    subset_mode=subset_mode,
    seed=seed,
)

qid_to_idx = {}
for i, ex in enumerate(examples):
    qid_to_idx[str(ex.qid)] = i

print(f"[info] Loaded {len(examples)} examples from subset")
print(f"[info] Found {len(qid_to_idx)} qids in subset")

# --------------------------------------------------
# REBUILD RESUME STATE
# --------------------------------------------------
resume_state = {}
bad_rows = 0
used_rows = 0
skipped_rows = 0

with open(results_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except Exception:
            bad_rows += 1
            continue

        qid = str(row.get("question_id", ""))
        if qid not in qid_to_idx:
            skipped_rows += 1
            continue

        pipeline = str(row.get("pipeline", ""))
        generator = str(row.get("generator", ""))
        embedder = str(row.get("embedder", ""))
        top_k = int(row.get("top_k", 0))
        context_length = int(row.get("context_length", 0))

        idx = qid_to_idx[qid]

        key = mod.RunKey(
            pipeline=pipeline,
            generator=generator,
            embedder=embedder,
            top_k=top_k,
            context_length=context_length,
        )
        key_hash = key.to_str()

        # Mark this question as completed for that config
        resume_state[key_hash] = max(resume_state.get(key_hash, 0), idx + 1)
        used_rows += 1

print(f"[info] Used {used_rows} rows from results.jsonl")
print(f"[info] Ignored {bad_rows} malformed rows")
print(f"[info] Skipped {skipped_rows} rows whose qid was not in current subset")
print(f"[info] Reconstructed {len(resume_state)} resume entries")

# --------------------------------------------------
# BACKUP OLD RESUME FILE
# --------------------------------------------------
if resume_path.exists():
    backup_path = resume_path.with_suffix(".json.bak")
    shutil.copy2(resume_path, backup_path)
    print(f"[info] Backed up old resume file to: {backup_path}")

# --------------------------------------------------
# WRITE NEW RESUME FILE
# --------------------------------------------------
tmp_path = resume_path.with_suffix(".json.tmp")
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(resume_state, f, indent=2, ensure_ascii=False)

tmp_path.replace(resume_path)

print(f"[ok] Rebuilt resume file: {resume_path}")