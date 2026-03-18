from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from edgerag.core.utils import safe_mkdir

KILT_BASE = "http://dl.fbaipublicfiles.com/KILT"


def resolve_kilt_nq_paths(data_dir: Path, split: str = "dev") -> Tuple[Path, Path]:
    split = split.lower()
    if split not in {"train", "dev"}:
        raise ValueError("split must be 'train' or 'dev'")
    nq_name = f"nq-{split}-kilt.jsonl"
    ks_name = "kilt_knowledgesource.json"
    return data_dir / nq_name, data_dir / ks_name


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    import requests

    safe_mkdir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        if resume_pos > 0 and response.status_code == 200:
            resume_pos = 0
            tmp.unlink(missing_ok=True)
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        mode = "ab" if resume_pos > 0 else "wb"
        with open(tmp, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    tmp.replace(out_path)


def ensure_kilt_nq(data_dir: Path, split: str = "dev") -> Tuple[Path, Path]:
    nq_path, ks_path = resolve_kilt_nq_paths(data_dir, split=split)
    if not nq_path.exists():
        print(f"[data] Downloading {nq_path.name} ...")
        download_file(f"{KILT_BASE}/{nq_path.name}", nq_path)
    if not ks_path.exists():
        print(f"[data] Downloading {ks_path.name} (large: ~35GiB) ...")
        download_file(f"{KILT_BASE}/{ks_path.name}", ks_path)
    return nq_path, ks_path


@dataclass
class KiltExample:
    qid: str
    question: str
    gold_answers: List[str]
    gold_provenance: List[str]


def load_kilt_nq_examples(
    nq_jsonl: Path,
    max_questions: Optional[int] = None,
    subset_mode: str = "head",
    seed: int = 123,
) -> List[KiltExample]:
    subset_mode = (subset_mode or "head").strip().lower()
    if subset_mode not in {"head", "random"}:
        raise ValueError("subset_mode must be 'head' or 'random'")

    examples_all: List[KiltExample] = []
    with open(nq_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if subset_mode == "head" and max_questions is not None and i >= max_questions:
                break
            obj = json.loads(line)
            qid = str(obj.get("id", i))
            question = (obj.get("input") or "").strip()
            gold_answers: List[str] = []
            gold_prov: List[str] = []
            for out in obj.get("output", []) or []:
                ans = out.get("answer")
                if isinstance(ans, str) and ans.strip():
                    gold_answers.append(ans.strip())
                for prov in out.get("provenance", []) or []:
                    wid = prov.get("wikipedia_id")
                    sp = prov.get("start_paragraph_id")
                    ep = prov.get("end_paragraph_id")
                    if wid is None:
                        continue
                    if sp is not None and ep is not None:
                        try:
                            sp_i = int(sp)
                            ep_i = int(ep)
                        except Exception:
                            sp_i = None
                            ep_i = None
                        if sp_i is not None and ep_i is not None and ep_i >= sp_i:
                            for pid_idx in range(sp_i, ep_i + 1):
                                gold_prov.append(f"{wid}:{pid_idx}")
                            continue
                    if sp is not None:
                        gold_prov.append(f"{wid}:{sp}")
                    else:
                        gold_prov.append(str(wid))
            gold_answers = list(dict.fromkeys(gold_answers))
            gold_prov = list(dict.fromkeys(gold_prov))
            examples_all.append(KiltExample(qid=qid, question=question, gold_answers=gold_answers, gold_provenance=gold_prov))

    if subset_mode == "random" and max_questions is not None:
        if max_questions >= len(examples_all):
            return examples_all
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(len(examples_all), size=int(max_questions), replace=False)
        return [examples_all[i] for i in sorted(int(i) for i in indices)]

    if max_questions is not None:
        return examples_all[: int(max_questions)]
    return examples_all
