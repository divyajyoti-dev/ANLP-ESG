#!/usr/bin/env python3
# bt_tune_refined.py
# Stable, refined BERTopic tuner for your ESG corpus (per company or all)

import os, sys, csv, json, random, argparse, itertools
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd

# ---- macOS / tokenizer stability (put BEFORE any HF imports) ----
import multiprocessing as mp
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    mp.set_start_method("spawn", force=True)  # safer than fork on macOS
except RuntimeError:
    pass

# --------- dependencies ----------
# pip install bertopic sentence-transformers umap-learn hdbscan gensim pandas scikit-learn matplotlib
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from umap import UMAP
from hdbscan import HDBSCAN
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt


# ------------------- IO -------------------

def read_docs(data_path: str, text_col: str) -> List[str]:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() not in [".txt", ".csv", ".jsonl", ".json"]:
        raise ValueError("Use .txt, .csv, .jsonl, or .json")

    if p.suffix.lower() == ".txt":
        docs = [line.strip() for line in p.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if text_col not in df.columns:
            raise ValueError(f"--text-col '{text_col}' not found. Columns: {list(df.columns)[:12]}")
        docs = df[text_col].dropna().astype(str).tolist()
    else:  # json/jsonl
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        docs = []
        for l in lines:
            try:
                obj = json.loads(l)
            except Exception:
                obj = {}
            docs.append(str(obj.get(text_col, "")))
        docs = [d for d in docs if d.strip()]

    # light cleanup + drop tiny segments
    docs = [d.replace("\n", " ").strip() for d in docs if len(d.split()) >= 5]
    return docs


# --------------- metrics ------------------

def _topic_words_dict(model: BERTopic, top_k: int = 10) -> Dict[int, List[str]]:
    topics = {}
    for tid, words in model.get_topics().items():
        if tid == -1:
            continue
        topics[tid] = [w for (w, _) in words[:top_k]]
    return topics

def topic_diversity(model: BERTopic, top_k: int = 10) -> float:
    topics = _topic_words_dict(model, top_k)
    if not topics: return 0.0
    words = [w for lst in topics.values() for w in lst]
    return len(set(words)) / (len(topics) * top_k)

def coherence_cv(model: BERTopic, docs: List[str], top_k: int = 10) -> float:
    topics = _topic_words_dict(model, top_k)
    if not topics: return 0.0
    texts = [d.lower().split() for d in docs]
    dictionary = Dictionary(texts)
    topic_list = list(topics.values())
    cm = CoherenceModel(topics=topic_list, texts=texts, dictionary=dictionary, coherence='c_v')
    return float(cm.get_coherence())

def silhouette_umap(model: BERTopic, labels: np.ndarray) -> float:
    try:
        umap_embeds = np.array(model.umap_model_.embedding_)
    except Exception:
        return 0.0
    mask = (labels != -1)
    if mask.sum() < 10:  # too few points
        return 0.0
    lbls = labels[mask]
    if len(np.unique(lbls)) < 2:
        return 0.0
    return float(silhouette_score(umap_embeds[mask], lbls, metric="euclidean"))

def combo_score(div, coh, sil):
    sil01 = (sil + 1.0) / 2.0
    return 0.4*coh + 0.4*div + 0.2*sil01


# --------------- search space (refined) -------------

def param_space(seed: int = 7):
    random.seed(seed)
    space = list(itertools.product(
        [(1,2)],       # bigrams help ESG phrasing ("carbon neutral")
        [3,5],         # min_df trims noise
        [15],          # richer top words
        [10,20,30],    # min_topic_size
        [15,25],       # UMAP n_neighbors
        [5,8],         # UMAP n_components
        [0.0,0.05],    # UMAP min_dist
        [10,15,25],    # HDBSCAN min_cluster_size
        [5,10]         # HDBSCAN min_samples
    ))
    random.shuffle(space)
    return space


# --------------- trial --------------

def run_trial(docs: List[str], embedder_name: str, params: tuple, verbose=False):
    ngr, mindf, topn, mts, nnei, ndim, mdist, mcls, msamp = params

    embedder = SentenceTransformer(embedder_name)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=ngr, min_df=mindf, max_df=0.9)
    rep_model = MaximalMarginalRelevance(diversity=0.3)

    umap_model = UMAP(n_neighbors=nnei, n_components=ndim, min_dist=mdist, metric="cosine", random_state=42)
    hdb_model  = HDBSCAN(min_cluster_size=mcls, min_samples=msamp, metric="euclidean", prediction_data=True)

    model = BERTopic(
        embedding_model=embedder,
        vectorizer_model=vectorizer,
        representation_model=rep_model,
        top_n_words=topn,
        min_topic_size=mts,
        nr_topics="auto",
        umap_model=umap_model,
        hdbscan_model=hdb_model,
        calculate_probabilities=False,
        low_memory=True,
        verbose=verbose
    )

    topics, _ = model.fit_transform(docs)

    div = topic_diversity(model, top_k=min(10, topn))
    coh = coherence_cv(model, docs, top_k=min(10, topn))
    sil = silhouette_umap(model, np.array(topics))
    score = combo_score(div, coh, sil)

    result = {
        "ngram_range": str(ngr), "min_df": mindf, "top_n_words": topn,
        "min_topic_size": mts, "umap_n_neighbors": nnei, "umap_n_components": ndim,
        "umap_min_dist": mdist, "hdb_min_cluster_size": mcls, "hdb_min_samples": msamp,
        "topic_diversity": round(div, 6),
        "coherence_cv": round(coh, 6),
        "silhouette": round(sil, 6),
        "score": round(score, 6),
        "n_topics": int(model.get_topic_info().shape[0]-1)
    }
    return model, topics, result


# --------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Refined BERTopic tuner (Mac-safe)")
    ap.add_argument("--data", required=True, help="Path to .txt/.csv/.jsonl/.json")
    ap.add_argument("--text-col", required=True, help="Text column (e.g., page_text_final)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--trials", type=int, default=20, help="Number of trials (20 recommended)")
    ap.add_argument("--embedder", default="all-mpnet-base-v2", help="SentenceTransformer name")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    docs = read_docs(args.data, args.text_col)
    print(f"Loaded {len(docs)} docs.")

    space = param_space(seed=args.seed)
    trials = min(args.trials, len(space))

    all_rows = []
    best = None  # (score, model, topics, result)
    for i in range(trials):
        print(f"[{i+1}/{trials}] Trying params:", space[i])
        try:
            model, topics, res = run_trial(docs, args.embedder, space[i], verbose=False)
        except Exception as e:
            print("  -> trial failed:", e)
            continue

        all_rows.append(res)
        if (best is None) or (res["score"] > best[0]):
            best = (res["score"], model, topics, res)

        pd.DataFrame(all_rows).sort_values("score", ascending=False).to_csv(
            os.path.join(args.outdir, "bt_tuning_history.csv"), index=False
        )

    if best is None:
        print("No successful trials. Exiting.")
        sys.exit(1)

    best_score, best_model, best_topics, best_res = best
    print("\n=== BEST SETTINGS ===")
    print(best_res)

    # Save artifacts
    best_model.save(os.path.join(args.outdir, "bertopic_model"))
    pd.DataFrame(all_rows).sort_values("score", ascending=False).to_csv(
        os.path.join(args.outdir, "bt_tuning_history.csv"), index=False
    )

    info = best_model.get_topic_info()
    info.to_csv(os.path.join(args.outdir, "topic_info.csv"), index=False)

    topics_dict = {}
    for tid in info["Topic"].tolist():
        if tid == -1:
            continue
        topics_dict[int(tid)] = best_model.get_topic(int(tid))
    with open(os.path.join(args.outdir, "topics_top_words.json"), "w") as f:
        json.dump({k: [(w, float(s)) for (w, s) in v] for k, v in topics_dict.items()}, f, indent=2)

    # Visuals
    try:
        fig_html = best_model.visualize_topics()
        fig_html.write_html(os.path.join(args.outdir, "viz_topics.html"))
        fig_html = best_model.visualize_hierarchy()
        fig_html.write_html(os.path.join(args.outdir, "viz_hierarchy.html"))
        fig_html = best_model.visualize_barchart()
        fig_html.write_html(os.path.join(args.outdir, "viz_barchart.html"))
    except Exception as e:
        print("Visualizations skipped:", e)

    # Topic sizes
    try:
        sizes = info[info["Topic"]!=-1][["Topic","Count"]].sort_values("Count", ascending=False)
        plt.figure()
        plt.bar(sizes["Topic"].astype(str), sizes["Count"])
        plt.xticks(rotation=90)
        plt.title("Topic Sizes (best BERTopic)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "topic_sizes.png"), dpi=200)
        plt.close()
    except Exception as e:
        print("Matplotlib plot skipped:", e)

    print(f"\nSaved best model + metrics to: {args.outdir}")
    print("Files: bertopic_model/, bt_tuning_history.csv, topic_info.csv, topics_top_words.json, viz_*.html")

if __name__ == "__main__":
    main()
