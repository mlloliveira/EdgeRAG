#!/usr/bin/env python3
"""
EdgeRAG Phase-1 analysis extended with SAS (Semantic Answer Similarity).

This script wraps the current analysis pipeline in edge_rag_final_analysis_v2_5.py and adds
QA-specific semantic evaluation using SAS, motivated by Risch et al. (MRQA 2021), which showed
that transformer-based semantic similarity metrics correlate better with human judgments than
lexical metrics such as EM/F1 for QA answers.

Implementation choices:
- primary semantic metric: SAS(answer, gold_answers) = max similarity(answer, gold_i)
- model default: cross-encoder/stsb-roberta-large (cross-encoder as proposed in the SAS paper)
- grounded SAS variants mirror existing grounded EM/F1 variants:
    metric_kilt_sas_hit = SAS * provenance_hit
    metric_kilt_sas_all = SAS * provenance_all

This preserves the existing publication outputs and adds SAS tables/figures on top.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_base_module(base_path: Path):
    spec = importlib.util.spec_from_file_location("edge_rag_final_analysis_v2_5", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base analysis module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def strip_and_validate_answer(base, answer: Any) -> Optional[str]:
    if not isinstance(answer, str):
        return None
    cleaned = base.strip_think_prefix_for_scoring(answer)
    cleaned = cleaned.strip()
    if cleaned == "":
        return None
    return cleaned


def _normalize_sas_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if np.isnan(scores).any():
        scores = np.nan_to_num(scores, nan=0.0)
    # Cross-encoders for STS often emit raw regression scores not constrained to [0, 1].
    if (scores < 0).any() or (scores > 1).any():
        scores = sigmoid(scores)
    scores = np.clip(scores, 0.0, 1.0)
    return scores


def build_sas_pairs(base, canon: pd.DataFrame) -> Tuple[List[Tuple[str, str]], List[int], List[List[str]]]:
    pairs: List[Tuple[str, str]] = []
    pair_row_idx: List[int] = []
    grouped_refs: List[List[str]] = []

    for row_idx, row in canon.iterrows():
        if row.get("pipeline") == "P0":
            grouped_refs.append([])
            continue
        if row.get("status_norm", "ok") == "failed":
            grouped_refs.append([])
            continue
        answer = strip_and_validate_answer(base, row.get("answer"))
        gold_answers = row.get("gold_answers")
        if answer is None or not isinstance(gold_answers, list) or len(gold_answers) == 0:
            grouped_refs.append([])
            continue
        refs = [str(g).strip() for g in gold_answers if isinstance(g, str) and str(g).strip()]
        grouped_refs.append(refs)
        for ref in refs:
            pairs.append((answer, ref))
            pair_row_idx.append(row_idx)
    return pairs, pair_row_idx, grouped_refs


def compute_sas_scores(
    base,
    canon: pd.DataFrame,
    model_name: str = "cross-encoder/stsb-roberta-large",
    batch_size: int = 32,
    device: Optional[str] = None,
    progress_bar: bool = True,
    fake_model: Any = None,
) -> pd.DataFrame:
    """Compute SAS for each row as max similarity(answer, any gold answer).

    By default this uses a sentence-transformers CrossEncoder model, which aligns with the
    cross-encoder formulation used in the SAS paper.
    """

    canon = canon.copy()
    canon["metric_sas"] = np.nan
    canon["metric_kilt_sas_hit"] = np.nan
    canon["metric_kilt_sas_all"] = np.nan

    pairs, pair_row_idx, grouped_refs = build_sas_pairs(base, canon)
    if not pairs:
        return canon

    if fake_model is not None:
        model = fake_model
    else:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "SAS evaluation requires sentence-transformers. Install it with: pip install sentence-transformers"
            ) from e
        model_kwargs = {}
        if device:
            model_kwargs["device"] = device
        model = CrossEncoder(model_name, **model_kwargs)

    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=progress_bar)
    raw_scores = np.asarray(raw_scores)
    scores = _normalize_sas_scores(raw_scores)

    by_row: Dict[int, List[float]] = {}
    for row_idx, score in zip(pair_row_idx, scores):
        by_row.setdefault(int(row_idx), []).append(float(score))

    for row_idx, vals in by_row.items():
        sas = max(vals) if vals else np.nan
        canon.at[row_idx, "metric_sas"] = sas

        retrieved_ids = canon.at[row_idx, "retrieved_ids"] if "retrieved_ids" in canon.columns else []
        gold_provenance = canon.at[row_idx, "gold_provenance"] if "gold_provenance" in canon.columns else []
        retrieved_ids = retrieved_ids if isinstance(retrieved_ids, list) else []
        gold_provenance = gold_provenance if isinstance(gold_provenance, list) else []
        gold_set = set(gold_provenance)
        retrieved_set = set(retrieved_ids)
        prov_hit = 1.0 if gold_set and gold_set.intersection(retrieved_set) else 0.0
        prov_all = 1.0 if gold_set and gold_set.issubset(retrieved_set) else 0.0
        canon.at[row_idx, "metric_kilt_sas_hit"] = float(sas) * prov_hit
        canon.at[row_idx, "metric_kilt_sas_all"] = float(sas) * prov_all

    return canon


def _best_metric_by_generator_topk(canon: pd.DataFrame, pipeline: str, metric_col: str) -> pd.DataFrame:
    sub = (
        canon[canon["pipeline"] == pipeline]
        .groupby(["generator_label", "top_k", "embedder_label"], as_index=False)
        .agg(metric=(metric_col, "mean"))
        .sort_values(["generator_label", "top_k", "metric"], ascending=[True, True, False])
        .drop_duplicates(["generator_label", "top_k"], keep="first")
    )
    return sub


def _build_generator_metric_table(canon: pd.DataFrame, generator_order: List[str], metric_col: str, metric_label: str, include_p1: bool = True) -> pd.DataFrame:
    table = pd.DataFrame({"Generator": generator_order})
    if include_p1:
        p1_metric = (
            canon[canon["pipeline"] == "P1"]
            .groupby("generator_label", as_index=False)
            .agg(value=(metric_col, "mean"))
            .rename(columns={"value": f"P1 {metric_label}"})
        )
        table = table.merge(p1_metric.rename(columns={"generator_label": "Generator"}), on="Generator", how="left")

    for pipeline in ["P2", "P3"]:
        pivot = _best_metric_by_generator_topk(canon, pipeline, metric_col).pivot(
            index="generator_label", columns="top_k", values="metric"
        )
        for k in sorted(k for k in pivot.columns.tolist() if pd.notna(k)):
            table[f"{pipeline} {metric_label} k = {int(k)}"] = table["Generator"].map(pivot[k])
    return table


def build_sas_publication_tables(base, canon: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    generator_order = base.ordered_present(
        canon.loc[canon["generator"] != "__shared_p0__", "generator_label"].dropna().astype(str).unique().tolist(),
        meta["gen_order_label"],
    )

    generator_sas_table = _build_generator_metric_table(canon, generator_order, "metric_sas", "SAS", include_p1=True)
    generator_grounded_sas_hit_table = _build_generator_metric_table(canon, generator_order, "metric_kilt_sas_hit", "gSAS-hit", include_p1=False)
    generator_grounded_sas_all_table = _build_generator_metric_table(canon, generator_order, "metric_kilt_sas_all", "gSAS-all", include_p1=False)

    return {
        "table_generator_sas": generator_sas_table,
        "table_generator_grounded_sas_hit": generator_grounded_sas_hit_table,
        "table_generator_grounded_sas_all": generator_grounded_sas_all_table,
    }


def build_sas_summary_tables(canon: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    p1_sas = (
        canon[canon["pipeline"] == "P1"]
        .groupby(["generator", "generator_label"], as_index=False)
        .agg(sas=("metric_sas", "mean"))
        .sort_values("generator_label")
    )
    p23_sas_cfg = (
        canon[canon["pipeline"].isin(["P2", "P3"])]
        .groupby(["pipeline", "generator", "generator_label", "embedder", "embedder_label", "top_k"], as_index=False)
        .agg(
            sas=("metric_sas", "mean"),
            grounded_sas_hit=("metric_kilt_sas_hit", "mean"),
            grounded_sas_all=("metric_kilt_sas_all", "mean"),
        )
        .sort_values(["pipeline", "generator_label", "sas"], ascending=[True, True, False])
    )
    return {
        "p1_sas_summary": p1_sas,
        "p23_sas_by_config": p23_sas_cfg,
    }




def _figure_heatmap_metric(base, canon: pd.DataFrame, outdir: Path, meta: Dict[str, Any], pipeline: str, metric_col: str, metric_label: str, stem: str, title: str) -> None:
    gen_order_raw = base.ordered_present(
        canon.loc[canon["pipeline"] == pipeline, "generator"].dropna().astype(str).unique().tolist(),
        meta["gen_order_raw"],
    )
    gen_order_label = [base.label_generator(g, meta) for g in gen_order_raw]
    col_order = base._heatmap_column_order(canon, pipeline, meta)
    if not gen_order_raw or not col_order:
        return

    heat_df = (
        canon[canon["pipeline"] == pipeline]
        .groupby(["generator", "embedder_label", "top_k"], as_index=False)
        .agg(metric=(metric_col, "mean"))
    )
    heat_df["col"] = heat_df["embedder_label"] + " k=" + heat_df["top_k"].astype(str)
    heat = heat_df.pivot(index="generator", columns="col", values="metric").reindex(index=gen_order_raw, columns=col_order) * 100

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(col_order)), max(4.8, 0.55 * len(gen_order_raw) + 2.5)))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(gen_order_raw)))
    ax.set_yticklabels(gen_order_label)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{metric_label} (%)")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    base.save_all_formats(fig, outdir, stem)


def figure9_p2_heatmap_sas(base, canon: pd.DataFrame, outdir: Path, meta: Dict[str, Any]) -> None:
    _figure_heatmap_metric(base, canon, outdir, meta, pipeline="P2", metric_col="metric_sas", metric_label="SAS", stem="fig9_p2_heatmap_sas", title="P2 SAS by generator, embedder, and top-k")


def figure10_p3_heatmap_sas(base, canon: pd.DataFrame, outdir: Path, meta: Dict[str, Any]) -> None:
    _figure_heatmap_metric(base, canon, outdir, meta, pipeline="P3", metric_col="metric_sas", metric_label="SAS", stem="fig10_p3_heatmap_sas", title="P3 SAS by generator, embedder, and top-k")

def figure7_generator_scaling_sas(base, canon: pd.DataFrame, outdir: Path, meta: Dict[str, Any]) -> None:
    gen_order_raw = base.ordered_present(
        canon.loc[canon["generator"] != "__shared_p0__", "generator"].dropna().astype(str).unique().tolist(),
        meta["gen_order_raw"],
    )
    if not gen_order_raw:
        return

    p1_sas = (
        canon[canon["pipeline"] == "P1"]
        .groupby("generator", as_index=False)["metric_sas"]
        .mean()
        .rename(columns={"metric_sas": "p1_sas"})
    )
    best_p2 = (
        canon[canon["pipeline"] == "P2"]
        .groupby(["generator", "embedder_label", "top_k"], as_index=False)
        .agg(sas=("metric_sas", "mean"))
        .sort_values(["generator", "sas"])
        .groupby("generator", as_index=False)
        .tail(1)
        .rename(columns={"sas": "p2_sas"})
    )
    best_p3 = (
        canon[canon["pipeline"] == "P3"]
        .groupby(["generator", "embedder_label", "top_k"], as_index=False)
        .agg(sas=("metric_sas", "mean"))
        .sort_values(["generator", "sas"])
        .groupby("generator", as_index=False)
        .tail(1)
        .rename(columns={"sas": "p3_sas"})
    )

    scale_df = pd.DataFrame({"generator": gen_order_raw})
    scale_df["generator_label"] = scale_df["generator"].map(lambda x: base.label_generator(x, meta))
    scale_df = scale_df.merge(p1_sas, on="generator", how="left")
    scale_df = scale_df.merge(best_p2[["generator", "p2_sas"]], on="generator", how="left")
    scale_df = scale_df.merge(best_p3[["generator", "p3_sas"]], on="generator", how="left")
    for col in ["p1_sas", "p2_sas", "p3_sas"]:
        scale_df[col] = scale_df[col] * 100

    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(gen_order_raw)), 4.8))
    x = np.arange(len(gen_order_raw))
    w = 0.24
    bar_specs = [("p1_sas", "P1", -w), ("p2_sas", "Best P2", 0), ("p3_sas", "Best P3", w)]
    for col, label, offset in bar_specs:
        bars = ax.bar(x + offset, scale_df[col], w, label=label)
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.1f}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    ax.set_title("Generator scaling: semantic answer similarity (SAS)")
    ax.set_ylabel("SAS (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(scale_df["generator_label"], rotation=20)
    ax.legend()
    base.save_all_formats(fig, outdir, "fig7_generator_scaling_sas")


def figure8_sas_vs_f1(base, canon: pd.DataFrame, outdir: Path, meta: Dict[str, Any]) -> None:
    sub = canon[canon["pipeline"].isin(["P1", "P2", "P3"])].copy()
    agg = (
        sub.groupby(["pipeline", "generator", "generator_label"], as_index=False)
        .agg(f1=("metric_f1", "mean"), sas=("metric_sas", "mean"))
    )
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    markers = {"P1": "^", "P2": "o", "P3": "s"}
    for pipe in ["P1", "P2", "P3"]:
        s = agg[agg["pipeline"] == pipe]
        ax.scatter(s["f1"] * 100, s["sas"] * 100, marker=markers[pipe], label=pipe)
        for _, r in s.iterrows():
            ax.annotate(r["generator_label"], (r["f1"]*100, r["sas"]*100), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean F1 (%)")
    ax.set_ylabel("Mean SAS (%)")
    ax.set_title("Semantic vs lexical answer quality")
    ax.legend()
    base.save_all_formats(fig, outdir, "fig8_sas_vs_f1")


def run_self_tests() -> None:
    # helper tests independent of sentence-transformers download
    class FakeModel:
        def predict(self, pairs, batch_size=32, show_progress_bar=True):
            out = []
            for a, b in pairs:
                if a.lower() == b.lower():
                    out.append(3.0)  # will be sigmoid-normalized close to 1
                elif a.lower() in b.lower() or b.lower() in a.lower():
                    out.append(1.0)
                else:
                    out.append(-2.0)
            return np.array(out, dtype=np.float32)

    # dynamically load base normalize behavior for think tags etc.
    base = _load_base_module(Path(__file__).with_name('edge_rag_final_analysis_v2_5.py'))
    raw = pd.DataFrame([
        {
            "pipeline": "P1", "generator": "g1", "generator_label": "G1", "embedder": "__none__", "embedder_label": "__none__",
            "top_k": 0, "context_length": 4096, "status_norm": "ok", "answer": "<think>stuff</think> Paris",
            "gold_answers": ["Paris", "The city of Paris"], "retrieved_ids": [], "gold_provenance": []
        },
        {
            "pipeline": "P2", "generator": "g1", "generator_label": "G1", "embedder": "e1", "embedder_label": "E1",
            "top_k": 5, "context_length": 4096, "status_norm": "ok", "answer": "capital of france",
            "gold_answers": ["Paris"], "retrieved_ids": ["a"], "gold_provenance": ["a"]
        },
        {
            "pipeline": "P3", "generator": "g1", "generator_label": "G1", "embedder": "e1", "embedder_label": "E1",
            "top_k": 10, "context_length": 4096, "status_norm": "failed", "answer": "", "gold_answers": ["Paris"],
            "retrieved_ids": ["a"], "gold_provenance": ["a"]
        },
    ])
    scored = compute_sas_scores(base, raw, fake_model=FakeModel(), progress_bar=False)
    assert scored.loc[0, 'metric_sas'] > 0.9
    assert scored.loc[1, 'metric_sas'] < scored.loc[0, 'metric_sas']
    assert np.isnan(scored.loc[2, 'metric_sas'])
    assert scored.loc[1, 'metric_kilt_sas_hit'] == scored.loc[1, 'metric_sas']
    assert scored.loc[1, 'metric_kilt_sas_all'] == scored.loc[1, 'metric_sas']
    print('[tests] SAS helper tests passed')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='EdgeRAG analysis with SAS extension')
    p.add_argument('--results', type=Path, default=Path('/mnt/data/results.jsonl'))
    p.add_argument('--config', type=Path, default=Path('/mnt/data/phase1_config_fix9_v4.json'))
    p.add_argument('--outdir', type=Path, default=Path('final_results_analysis_sas'))
    p.add_argument('--base_script', type=Path, default=None, help='Path to edge_rag_final_analysis_v2_5.py')
    p.add_argument('--sas_model', type=str, default='cross-encoder/stsb-roberta-large')
    p.add_argument('--sas_batch_size', type=int, default=32)
    p.add_argument('--sas_device', type=str, default=None)
    p.add_argument('--skip_sas', action='store_true', help='Skip SAS computation and only run base analysis outputs')
    p.add_argument('--run_tests', action='store_true')
    return p.parse_args()


def render_publication_table_large(base, name: str, df: pd.DataFrame, outdir: Path, figw: float, fontsize: float = 13.0) -> None:
    """Render a dense publication table to PNG/SVG/PDF with larger default sizing.

    This is used for the SAS generator tables, which are too wide/dense for the
    default PNG rendering inherited from the base analysis script.
    """
    render_df = base._format_publication_table_for_png(name, df)
    nrows, ncols = render_df.shape
    fig_w = max(10.0, figw * ncols)
    fig_h = max(1.6 + 0.52 * (nrows + 1), 2.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=render_df.values,
        colLabels=render_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.08, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.0)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_edgecolor("black")
            cell.set_linewidth(0.9)
        elif row == nrows:
            cell.visible_edges = "B"
            cell.set_edgecolor("black")
            cell.set_linewidth(0.9)
        else:
            cell.visible_edges = ""
        if col == 0 and row > 0:
            cell._loc = "left"

    # Slight padding helps readability when zoomed or included in a paper draft.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

    for ext in ["png", "svg", "pdf"]:
        dpi = 450 if ext == "png" else None
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_sas_publication_tables_large(base, publication_tables: Dict[str, pd.DataFrame], outdir: Path) -> None:
    """Export the SAS publication tables in larger raster/vector formats.

    Tuning knob:
      - increase figw below if a table is still too cramped
      - increase fontsize below if you want larger text
    """
    width_overrides = {
        "table_generator_sas": 2.7,
        "table_generator_grounded_sas_hit": 2.9,
        "table_generator_grounded_sas_all": 3.1,
    }
    fontsize_overrides = {
        "table_generator_sas": 13.0,
        "table_generator_grounded_sas_hit": 13.0,
        "table_generator_grounded_sas_all": 13.0,
    }

    for name, figw in width_overrides.items():
        if name in publication_tables:
            render_publication_table_large(
                base,
                name,
                publication_tables[name],
                outdir,
                figw=figw,
                fontsize=fontsize_overrides.get(name, 13.0),
            )


def main() -> None:
    args = parse_args()
    if args.run_tests:
        run_self_tests()
        return

    base_path = args.base_script or Path(__file__).with_name('edge_rag_final_analysis_v2_5.py')
    base = _load_base_module(base_path)

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = base.load_results(args.results)
    df = base.ensure_columns(df, [
        'error_stage', 'metric_recall_at_k', 'metric_r_precision', 'metric_em', 'metric_f1',
        'metric_kilt_em_hit', 'metric_kilt_f1_hit', 'metric_kilt_em_all', 'metric_kilt_f1_all',
        'time_retrieve_s', 'time_total_s', 'metric_sas', 'metric_kilt_sas_hit', 'metric_kilt_sas_all'
    ])
    cfg = base.load_config(args.config)
    meta = base.build_model_metadata(cfg, df)
    canon = base.canonicalize(df, meta)

    if not args.skip_sas:
        print(f"[sas] Computing SAS with model={args.sas_model}, batch_size={args.sas_batch_size}, device={args.sas_device or 'default'}")
        canon = compute_sas_scores(base, canon, model_name=args.sas_model, batch_size=args.sas_batch_size, device=args.sas_device)
    else:
        print('[sas] Skipping SAS computation')

    canon.to_json(args.outdir / 'canonical_results_with_sas.jsonl', orient='records', lines=True, force_ascii=False)

    # Existing outputs
    tables = base.build_summary_tables(canon)
    sas_tables = build_sas_summary_tables(canon)
    all_tables = {**tables, **sas_tables}
    for name, table in all_tables.items():
        table.to_csv(args.outdir / f'{name}.csv', index=False)

    publication_tables = base.build_publication_tables(canon, meta)
    publication_tables.update(build_sas_publication_tables(base, canon, meta))
    for name, table in publication_tables.items():
        table.to_csv(args.outdir / f'{name}.csv', index=False)
    base.export_publication_tables_xlsx(publication_tables, args.outdir / 'publication_tables.xlsx')
    base.export_publication_tables_png(publication_tables, args.outdir)
    export_sas_publication_tables_large(base, publication_tables, args.outdir)

    # Existing figures
    base.figure1_p0(canon, args.outdir, meta)
    base.figure2_generator_scaling(canon, args.outdir, meta)
    base.figure3_p2_heatmap(canon, args.outdir, meta)
    base.figure4_p3_heatmap(canon, args.outdir, meta)
    base.figure5_fail_rate(canon, args.outdir, meta)
    base.figure5b_fail_count(canon, args.outdir, meta)
    base.figure6_pareto(canon, args.outdir, meta)
    # New SAS figures
    figure7_generator_scaling_sas(base, canon, args.outdir, meta)
    figure8_sas_vs_f1(base, canon, args.outdir, meta)
    figure9_p2_heatmap_sas(base, canon, args.outdir, meta)
    figure10_p3_heatmap_sas(base, canon, args.outdir, meta)

    summary_path = args.outdir / 'summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        f.write('EdgeRAG analysis with SAS\n')
        f.write(f'Raw rows: {len(df)}\n')
        f.write(f'Canonical rows: {len(canon)}\n')
        f.write(f'Config used: {args.config}\n')
        f.write(f'Base analysis script: {base_path}\n')
        if args.skip_sas:
            f.write('SAS: skipped\n')
        else:
            f.write(f'SAS model: {args.sas_model}\n')
            f.write('SAS rationale: QA-specific semantic metric (cross-encoder) following Risch et al., MRQA 2021.\n')
        f.write('\nGenerated extra SAS outputs:\n')
        for name in ['p1_sas_summary', 'p23_sas_by_config', 'table_generator_sas', 'table_generator_grounded_sas_hit', 'table_generator_grounded_sas_all']:
            f.write(f'- {name}.csv/png (where applicable)\n')
        f.write('- fig7_generator_scaling_sas.png/.svg/.pdf\n')
        f.write('- fig8_sas_vs_f1.png/.svg/.pdf\n')
        f.write('- fig9_p2_heatmap_sas.png/.svg/.pdf\n')
        f.write('- fig10_p3_heatmap_sas.png/.svg/.pdf\n')

    print(f'Analysis with SAS complete. Outputs written to: {args.outdir}')


if __name__ == '__main__':
    main()
