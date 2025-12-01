#!/usr/bin/env python3
"""
COVID Structural Shift Analysis (Environmental Intensity)
--------------------------------------------------------

Implements Hypothesis 1 using an existing global BERTopic model.
Steps:
1) Load BERTopic model (no retraining).
2) Identify environmental topics via topic embeddings vs keyword centroid.
3) Compute environmental intensity per company-year from doc_topics_bertopic.csv.
4) Build control-adjusted series D = company intensity - peer average.
5) Fit interrupted time-series (ITS) regressions per company.
6) Compute counterfactual (pre-2020 trend) cumulative divergence 2020–2024.
7) Save tidy CSVs and plots.

Defaults assume paths inside this repo; override via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

# Ensure writable caches for matplotlib/numba before heavy imports
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"
MPL_DIR = CACHE_DIR / "matplotlib"
NUMBA_DIR = CACHE_DIR / "numba"
MPL_DIR.mkdir(parents=True, exist_ok=True)
NUMBA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_DIR))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # avoid caching issues in restricted envs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity


ENV_KEYWORDS = [
    "emissions",
    "ghg",
    "greenhouse gas",
    "carbon",
    "carbon neutral",
    "carbon negative",
    "carbon free",
    "carbon removal",
    "decarbonization",
    "climate",
    "climate risk",
    "renewable energy",
    "clean energy",
    "solar",
    "wind",
    "geothermal",
    "hydro",
    "battery storage",
    "energy storage",
    "energy efficiency",
    "energy intensity",
    "electrification",
    "fleet electrification",
    "ev",
    "ppas",
    "power purchase agreement",
    "virtual ppa",
    "green tariff",
    "net zero",
    "scope 1",
    "scope 2",
    "scope 3",
    "supplier emissions",
    "value chain",
    "water use",
    "water stewardship",
    "water intensity",
    "water discharge",
    "waste reduction",
    "waste diversion",
    "circular",
    "circular economy",
    "reuse",
    "recycling",
    "compost",
    "biodiversity",
    "habitat",
    "restoration",
    "reforestation",
    "afforestation",
    "nature based",
    "land use",
    "carbon price",
    "carbon tax",
    "offset",
    "credit",
    "rec",
    "eac",
    "science based",
    "sbti",
    "tcfd",
    "cdp",
    "ghg protocol",
    "footprint",
    "lca",
    "24/7",
    "cfe",
    "clean grid",
    "grid mix",
]


# ---------------------------- Embedding helpers ---------------------------- #
def embed_texts(embedding_model, texts: Iterable[str]) -> np.ndarray:
    """
    Embed a list of texts using the same embedding model as BERTopic.
    Supports SentenceTransformer-like .encode or .transform.
    """
    if hasattr(embedding_model, "encode"):
        return np.asarray(
            embedding_model.encode(list(texts), show_progress_bar=False)
        )
    if hasattr(embedding_model, "transform"):
        return np.asarray(embedding_model.transform(list(texts)))
    raise ValueError("Unsupported embedding model interface.")


# ---------------------------- Core functionality --------------------------- #
def load_model_and_topics(model_path: Path) -> Tuple[BERTopic, pd.DataFrame, np.ndarray]:
    """Load BERTopic model and return model, topic_info, topic_embeddings."""
    model = BERTopic.load(model_path)
    topic_info = model.get_topic_info()
    topic_embeddings = np.asarray(model.topic_embeddings_)
    return model, topic_info, topic_embeddings


def build_environmental_topic_set(
    topic_model: BERTopic,
    topic_embeddings: np.ndarray,
    sim_threshold: float = 0.60,
    keywords: List[str] | None = None,
) -> Tuple[List[int], pd.DataFrame]:
    """
    Create an environmental topic set by cosine similarity to a keyword centroid.
    Returns (env_topic_ids, similarity_df).
    """
    if keywords is None:
        keywords = ENV_KEYWORDS

    key_emb = embed_texts(topic_model.embedding_model, keywords)
    env_centroid = key_emb.mean(axis=0)

    # normalize
    env_norm = env_centroid / (np.linalg.norm(env_centroid) + 1e-9)
    topic_norms = np.linalg.norm(topic_embeddings, axis=1, keepdims=True) + 1e-9
    topic_unit = topic_embeddings / topic_norms

    sims = cosine_similarity(topic_unit, env_norm.reshape(1, -1)).ravel()

    topic_ids = list(range(len(sims)))
    sim_df = pd.DataFrame({"topic_id": topic_ids, "similarity": sims})
    env_topics = sim_df.loc[
        (sim_df["similarity"] > sim_threshold) & (sim_df["topic_id"] != -1),
        "topic_id",
    ].astype(int).tolist()
    return env_topics, sim_df


def load_doc_topics(path: Path) -> pd.DataFrame:
    """Load doc_topics_bertopic.csv and ensure required columns."""
    df = pd.read_csv(path)
    needed = {"company", "year", "topic_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df["company"] = df["company"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "topic_id"])
    df["topic_id"] = df["topic_id"].astype(int)
    return df


def compute_env_intensity(doc_df: pd.DataFrame, env_topics: List[int]) -> pd.DataFrame:
    """Flag environmental docs and aggregate intensity per company-year."""
    df = doc_df.copy()
    env_set = set(env_topics)
    df["is_env"] = df["topic_id"].apply(lambda x: x in env_set)

    agg = (
        df.groupby(["company", "year"], dropna=False)
        .agg(env_docs=("is_env", "sum"), total_docs=("topic_id", "count"))
        .reset_index()
    )
    agg["env_intensity"] = agg["env_docs"] / agg["total_docs"].replace(0, np.nan)
    return agg


def add_control_and_design(intensity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute control intensity (mean of other companies per year),
    deviation D, and ITS design variables.
    """
    df = intensity_df.copy()

    # Control intensity: mean of other companies in same year
    control_list = []
    for _, row in df.iterrows():
        mask = (df["year"] == row["year"]) & (df["company"] != row["company"])
        peers = df.loc[mask, "env_intensity"]
        control_val = peers.mean() if not peers.empty else np.nan
        control_list.append(control_val)
    df["control_intensity"] = control_list
    df["D"] = df["env_intensity"] - df["control_intensity"]

    # Time variables (2015 -> 1, 2016 -> 2, ...)
    df["time"] = df["year"] - 2014
    df["break_dummy"] = (df["year"] >= 2020).astype(int)
    df["time_after_break"] = (df["year"] - 2019).clip(lower=0)
    return df


def fit_its_per_company(design_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit ITS model per company: D_t = a + b1*time + b2*break + b3*time_after_break.
    Returns summary rows with betas and p-values.
    """
    rows = []
    companies = sorted(design_df["company"].unique())
    for comp in companies:
        sub = design_df.loc[
            design_df["company"] == comp, ["D", "time", "break_dummy", "time_after_break"]
        ].dropna()
        sub = sub.astype(float)
        if len(sub) < 4:
            continue  # not enough points to fit
        y = sub["D"]
        X = sm.add_constant(sub[["time", "break_dummy", "time_after_break"]])
        model = sm.OLS(y, X).fit()
        params = model.params
        pvals = model.pvalues
        rows.append(
            {
                "company": comp,
                "beta_time": params.get("time", np.nan),
                "p_time": pvals.get("time", np.nan),
                "beta_break": params.get("break_dummy", np.nan),
                "p_break": pvals.get("break_dummy", np.nan),
                "beta_post_slope": params.get("time_after_break", np.nan),
                "p_post_slope": pvals.get("time_after_break", np.nan),
                "r2": model.rsquared,
                "n": len(sub),
            }
        )
    return pd.DataFrame(rows)


def compute_cumulative_divergence(intensity_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each company:
      - Fit pre-2020 trend env_intensity ~ time (2015–2019).
      - Predict counterfactual 2020–2024.
      - Compute divergence and cumulative divergence.
    """
    records = []
    companies = sorted(intensity_df["company"].unique())
    for comp in companies:
        sub = intensity_df[intensity_df["company"] == comp].copy()
        sub["time"] = sub["year"] - 2014
        sub["env_intensity"] = pd.to_numeric(sub["env_intensity"], errors="coerce")

        pre = sub[sub["year"] < 2020].dropna(subset=["env_intensity", "time"])
        if len(pre) == 0:
            continue  # no pre-break data
        if len(pre) == 1:
            # Flat counterfactual using the single pre value
            const_val = float(pre["env_intensity"].iloc[0])
            sub["counterfactual_env_intensity"] = const_val
        else:
            y = pre["env_intensity"].astype(float)
            X = sm.add_constant(pre["time"].astype(float))
            trend = sm.OLS(y, X).fit()
            sub["counterfactual_env_intensity"] = trend.predict(
                sm.add_constant(sub["time"].astype(float))
            )

        # Predict for all years (keep actual for reporting)
        sub["counterfactual_env_intensity"] = trend.predict(
            sm.add_constant(sub["time"].astype(float))
        )
        sub["divergence"] = sub["env_intensity"] - sub["counterfactual_env_intensity"]

        # Cumulative divergence 2020–2024
        post_mask = sub["year"] >= 2020
        cumulative = sub.loc[post_mask, "divergence"].sum()
        sub["cumulative_divergence_per_company"] = cumulative

        records.append(sub[[
            "company",
            "year",
            "env_intensity",
            "counterfactual_env_intensity",
            "divergence",
            "cumulative_divergence_per_company",
        ]])

    if not records:
        return pd.DataFrame(
            columns=[
                "company",
                "year",
                "env_intensity",
                "counterfactual_env_intensity",
                "divergence",
                "cumulative_divergence_per_company",
            ]
        )
    return pd.concat(records, ignore_index=True)


# --------------------------------- Plots ---------------------------------- #
def plot_env_intensity(intensity_df: pd.DataFrame, out_dir: Path):
    """Small multiples: env intensity over time per company (cleaner than a single noisy plot)."""
    companies = sorted(intensity_df["company"].unique())
    n = len(companies)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = axes.flatten()
    for ax, comp in zip(axes, companies):
        sub = intensity_df[intensity_df["company"] == comp]
        sns.lineplot(data=sub, x="year", y="env_intensity", marker="o", ax=ax, color="#1f77b4")
        ax.axvline(2020, color="red", linestyle="--", alpha=0.6)
        ax.set_title(comp)
        ax.set_xlabel("Year")
        ax.set_ylabel("Env intensity")
        ax.grid(True, alpha=0.2)
    # Turn off unused axes
    for ax in axes[len(companies):]:
        ax.set_visible(False)
    plt.suptitle("Environmental Intensity Over Time (per company)", y=0.99)
    plt.tight_layout()
    out_path = out_dir / "env_intensity_over_time.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_D_with_fit(design_df: pd.DataFrame, its_results: pd.DataFrame, out_dir: Path):
    """Small multiples: D series with fitted ITS line per company."""
    companies = sorted(design_df["company"].unique())
    n = len(companies)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = axes.flatten()
    for ax, comp in zip(axes, companies):
        sub = design_df[design_df["company"] == comp].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub[["D", "time", "break_dummy", "time_after_break"]] = sub[
            ["D", "time", "break_dummy", "time_after_break"]
        ].astype(float)
        sns.lineplot(data=sub, x="year", y="D", marker="o", ax=ax, color="#1f77b4")
        ax.axvline(2020, color="red", linestyle="--", alpha=0.6)
        # Fit line for this company
        X = sm.add_constant(sub[["time", "break_dummy", "time_after_break"]])
        model = sm.OLS(sub["D"], X).fit()
        fitted = model.predict(X)
        ax.plot(sub["year"], fitted, linestyle="--", color="#ff7f0e", alpha=0.8, label="ITS fit")
        ax.set_title(comp)
        ax.set_xlabel("Year")
        ax.set_ylabel("D")
        ax.grid(True, alpha=0.2)
    # Turn off unused axes
    for ax in axes[len(companies):]:
        ax.set_visible(False)
    plt.suptitle("D = EnvIntensity - Control (per company)", y=0.99)
    plt.tight_layout()
    out_path = out_dir / "D_over_time.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_cumulative_divergence(div_df: pd.DataFrame, out_dir: Path):
    """Bar chart of cumulative divergence 2020–2024 per company."""
    # One row per company
    summary = (
        div_df.groupby("company")["cumulative_divergence_per_company"]
        .first()
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x="company", y="cumulative_divergence_per_company")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Cumulative Divergence from Pre-2020 Trend (2020–2024)")
    plt.ylabel("Cumulative divergence (env_intensity)")
    plt.xlabel("Company")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = out_dir / "cumulative_divergence.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# ---------------------------------- Main ---------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="COVID structural shift analysis using BERTopic outputs.")
    ap.add_argument(
        "--model-path",
        default="models/bertopic_model",
        help="Path to saved BERTopic model (global model, no retraining).",
    )
    ap.add_argument(
        "--topics-csv",
        default=None,
        help="Path to topics_bertopic.csv (used in fallback mode when model is unavailable).",
    )
    ap.add_argument(
        "--doc-topics",
        default="data/processed/doc_topics_bertopic.csv",
        help="Path to doc_topics_bertopic.csv (page-level topics).",
    )
    ap.add_argument(
        "--sim-threshold",
        type=float,
        default=0.60,
        help="Cosine similarity threshold for selecting environmental topics.",
    )
    ap.add_argument(
        "--fallback-top-terms",
        action="store_true",
        help="Use keyword hits on topics_bertopic.csv top_terms instead of loading a BERTopic model.",
    )
    ap.add_argument(
        "--min-keyword-hits",
        type=int,
        default=1,
        help="Minimum keyword hits in top_terms to mark a topic as environmental in fallback mode.",
    )
    ap.add_argument(
        "--output-dir",
        default="reports/covid_structural_shift",
        help="Directory to write outputs (CSVs, plots).",
    )
    ap.add_argument(
        "--save-similarities",
        action="store_true",
        help="Also save environmental topic similarity scores to CSV.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive environmental topic set
    if args.fallback_top_terms:
        if not args.topics_csv:
            raise SystemExit("Fallback mode requires --topics-csv pointing to topics_bertopic.csv")
        print(f"[info] Fallback mode: deriving environmental topics from {args.topics_csv}")
        topics_df = pd.read_csv(args.topics_csv)
        if "topic_id" not in topics_df.columns or "top_terms" not in topics_df.columns:
            raise SystemExit("topics_bertopic.csv must contain topic_id and top_terms columns")
        def keyword_hits(s: str) -> int:
            tokens = [w.strip().lower() for w in str(s).replace(",", " ").split()]
            return sum(1 for t in tokens for kw in ENV_KEYWORDS if kw in t)
        topics_df["keyword_hits"] = topics_df["top_terms"].apply(keyword_hits)
        env_topics = topics_df.loc[
            topics_df["keyword_hits"] >= args.min_keyword_hits, "topic_id"
        ].astype(int).tolist()
        sim_df = topics_df[["topic_id", "keyword_hits"]]
        print(f"[info] Environmental topics selected (>= {args.min_keyword_hits} keyword hits): {env_topics}")
        if args.save_similarities:
            sim_out = out_dir / "environmental_topic_keyword_hits.csv"
            sim_df.to_csv(sim_out, index=False)
            print(f"[info] Saved topic keyword hits → {sim_out}")
    else:
        print(f"[info] Loading BERTopic model from {args.model_path}")
        model, topic_info, topic_embs = load_model_and_topics(Path(args.model_path))
        print("[info] Building environmental topic set")
        env_topics, sim_df = build_environmental_topic_set(
            model, topic_embs, sim_threshold=args.sim_threshold
        )
        print(f"[info] Environmental topics selected (threshold {args.sim_threshold}): {env_topics}")
        if args.save_similarities:
            sim_out = out_dir / "environmental_topic_similarities.csv"
            sim_df.to_csv(sim_out, index=False)
            print(f"[info] Saved topic similarities → {sim_out}")

    print(f"[info] Loading doc topics from {args.doc_topics}")
    docs = load_doc_topics(Path(args.doc_topics))

    print("[info] Computing environmental intensity per company-year")
    intensity = compute_env_intensity(docs, env_topics)
    intensity_out = out_dir / "env_intensity.csv"
    intensity.to_csv(intensity_out, index=False)
    print(f"[done] Saved env intensity → {intensity_out}")

    print("[info] Building control-adjusted design matrix")
    design_df = add_control_and_design(intensity)
    design_out = out_dir / "its_design.csv"
    design_df.to_csv(design_out, index=False)
    print(f"[done] Saved ITS design → {design_out}")

    print("[info] Fitting ITS per company")
    its_results = fit_its_per_company(design_df)
    its_out = out_dir / "its_results.csv"
    its_results.to_csv(its_out, index=False)
    print(f"[done] Saved ITS results → {its_out}")

    print("[info] Computing cumulative divergence (counterfactual 2020–2024)")
    divergence_df = compute_cumulative_divergence(intensity)
    div_out = out_dir / "cumulative_divergence.csv"
    divergence_df.to_csv(div_out, index=False)
    print(f"[done] Saved divergence → {div_out}")

    print("[info] Making plots")
    p1 = plot_env_intensity(intensity, out_dir)
    p2 = plot_D_with_fit(design_df, its_results, out_dir)
    p3 = plot_cumulative_divergence(divergence_df, out_dir)
    print(f"[done] Plots saved: {p1}, {p2}, {p3}")


if __name__ == "__main__":
    main()
