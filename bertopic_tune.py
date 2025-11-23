#!/usr/bin/env python3
"""
Hyperparameter search for BERTopic (macOS-friendly, single process).
- Tries multiple configs (embedding, UMAP, HDBSCAN, min_topic_size)
- Scores each run with topic_diversity and silhouette (optionally coherence Cv)
- Writes a CSV leaderboard and (optionally) saves the best model's artifacts

Usage (joint):
  python bertopic_tune.py \
    --inputs data/processed/*_layout_all/corpus_pages_cleaned.csv \
    --out-dir reports/bt_tuning_all \
    --max-runs 12 \
    --save-best

Usage (one company):
  python bertopic_tune.py \
    --inputs data/processed/apple_layout_all/corpus_pages_cleaned.csv \
    --out-dir reports/bt_tuning_apple \
    --max-runs 8 --save-best
"""
import argparse, os, re, random, numpy as np, pandas as pd, time, itertools, json
from typing import List, Dict, Any

def infer_company(p: str) -> str:
    m = re.search(r"/processed/([^/]+)_layout_all/", p.replace("\\","/"))
    return m.group(1) if m else "Unknown"

def topic_diversity(word_lists: List[List[str]], topn: int = 10) -> float:
    flat = [w for lst in word_lists for w in lst[:topn]]
    return len(set(flat)) / max(1, len(flat))

def top_terms_from_topic(topic_obj, k=15):
    if isinstance(topic_obj, list):
        return [w for (w, _score) in topic_obj[:k]]
    if hasattr(topic_obj, "head") and "Word" in getattr(topic_obj, "columns", []):
        return topic_obj["Word"].head(k).astype(str).tolist()
    return [str(x) for x in list(topic_obj)[:k]]

def load_corpus(paths: List[str]) -> pd.DataFrame:
    frames=[]
    for p in paths:
        df = pd.read_csv(p)
        if "company" not in df.columns:
            df["company"] = infer_company(p)
        need = ["report_name","year","page","page_text_final","company"]
        miss = [c for c in need if c not in df.columns]
        if miss: raise SystemExit(f"Missing columns in {p}: {miss}")
        sub = df[need].rename(columns={"page_text_final":"text"}).copy()
        sub["year"] = pd.to_numeric(sub["year"], errors="coerce").astype("Int64")
        sub["text"] = sub["text"].fillna("").astype(str)
        frames.append(sub)
    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus[corpus["text"].str.strip().str.len()>0].reset_index(drop=True)
    if corpus.empty: raise SystemExit("All documents empty after filtering.")
    return corpus

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-runs", type=int, default=12, help="Cap number of configs to try")
    ap.add_argument("--save-best", action="store_true", help="Save artifacts for the best config")
    ap.add_argument("--with-coherence", action="store_true", help="Compute Cv (slower; needs gensim)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    # ---- Load corpus ----
    corpus = load_corpus(args.inputs)
    print(f"[INFO] Loaded {len(corpus)} pages from {len(args.inputs)} file(s).")

    # ---- Imports ----
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics import silhouette_score
        import umap
        import hdbscan
    except Exception as e:
        raise SystemExit("Install deps: pip install 'bertopic[all]' 'sentence-transformers>=2.2.2' umap-learn hdbscan scikit-learn")

    # Optional coherence
    HAVE_COH = False
    if args.with_coherence:
        try:
            from gensim.corpora import Dictionary
            from gensim.models.coherencemodel import CoherenceModel
            HAVE_COH = True
        except Exception:
            print("[WARN] gensim not available; skipping coherence.")

    # ---- Candidate grids (kept small on purpose) ----
    embeddings = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]           # mpnet is slower, often better
    min_topic_sizes = [20, 30, 40]
    umap_neighbors = [15, 30]
    umap_components = [5, 10]
    hdbscan_min_samples = [1, 5]

    # Build grid & limit
    grid = list(itertools.product(embeddings, min_topic_sizes, umap_neighbors, umap_components, hdbscan_min_samples))
    random.shuffle(grid)
    grid = grid[:args.max-runs if hasattr(args, "max-runs") else args.max_runs] if hasattr(args, "max-runs") else grid[:args.max_runs]  # defensive
    grid = grid[:args.max_runs]

    # ---- Encode once per embedding model to save time ----
    cached_embs: Dict[str, np.ndarray] = {}
    results = []
    best = {"score": -1, "idx": None}

    for i, (emb_name, mts, nnei, ncomp, minsamp) in enumerate(grid, 1):
        print(f"\n[RUN {i}/{len(grid)}] emb={emb_name}, min_topic_size={mts}, "
              f"umap_neighbors={nnei}, umap_components={ncomp}, hdbscan_min_samples={minsamp}")

        # Embeddings (cache per model)
        if emb_name not in cached_embs:
            print(f"[EMB] Encoding with {emb_name} (batch={args.batch_size}) …")
            st = SentenceTransformer(emb_name)
            cached_embs[emb_name] = st.encode(
                corpus["text"].tolist(),
                show_progress_bar=True,
                batch_size=args.batch_size
            )
        embs = cached_embs[emb_name]

        # Build reducers/clusterers explicitly for reproducibility
        reducer = umap.UMAP(
            n_neighbors=nnei,
            n_components=ncomp,
            metric="cosine",
            random_state=args.seed
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mts,
            min_samples=minsamp,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False
        )

        bt = BERTopic(
            embedding_model=SentenceTransformer(emb_name),
            umap_model=reducer,
            hdbscan_model=clusterer,
            min_topic_size=mts,
            verbose=False
        )

        start = time.time()
        topics, probs = bt.fit_transform(corpus["text"].tolist(), embs)
        dur = time.time() - start
        info = bt.get_topic_info()

        # Build wordlists for diversity
        rows = []
        for tid in info["Topic"].tolist():
            if tid == -1: continue
            words = top_terms_from_topic(bt.get_topic(tid), 10)
            rows.append(words)
        diversity = topic_diversity(rows, topn=10)

        # Silhouette (ignore noise)
        lab = np.array(topics)
        mask = lab != -1
        sil = float("nan")
        if mask.sum() > 50 and len(set(lab[mask])) > 1:
            sil = float(silhouette_score(np.asarray(embs)[mask], lab[mask]))

        # Optional coherence (Cv) using top words
        coh = float("nan")
        if HAVE_COH and rows:
            # tokenized docs (very light)
            texts = [str(t).lower().split() for t in corpus["text"].tolist()]
            dictionary = Dictionary(texts)
            try:
                cm = CoherenceModel(topics=rows, texts=texts, dictionary=dictionary, coherence="c_v")
                coh = float(cm.get_coherence())
            except Exception:
                pass

        rec = {
            "run": i,
            "embedding": emb_name,
            "min_topic_size": mts,
            "umap_neighbors": nnei,
            "umap_components": ncomp,
            "hdbscan_min_samples": minsamp,
            "topic_diversity": diversity,
            "silhouette": sil,
            "coherence_cv": coh,
            "train_seconds": round(dur, 2),
            "n_topics": int((info["Topic"] != -1).sum())
        }
        results.append(rec)
        print(f"[SCORES] diversity={diversity:.3f}  silhouette={sil if np.isnan(sil) else round(sil,3)}  "
              f"coh={coh if np.isnan(coh) else round(coh,3)}  topics={rec['n_topics']}  time={rec['train_seconds']}s")

        # Simple ranking: maximize (diversity + 0.5*silhouette + 0.5*coh_nan_to_0)
        score = diversity + 0.5*(0 if np.isnan(sil) else sil) + 0.5*(0 if np.isnan(coh) else coh)
        if score > best["score"]:
            best.update(score=score, idx=len(results)-1, params=rec, model=bt, topics=topics, embs=embs)

    # Save leaderboard
    out_csv = os.path.join(args.out_dir, "bertopic_tuning_results.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n[DONE] Wrote leaderboard → {out_csv}")

    # Save best artifacts if requested
    if args.save_best and best["idx"] is not None:
        bestdir = os.path.join(args.out_dir, "best")
        os.makedirs(bestdir, exist_ok=True)
        # Save best model
        try:
            best["model"].save(os.path.join(bestdir, "bertopic_model"))
        except Exception:
            pass
        # Save doc→topic and topic tables
        info = best["model"].get_topic_info()
        info.to_csv(os.path.join(bestdir, "topics_bertopic_info.csv"), index=False)

        # top terms
        rows=[]
        for tid in info["Topic"].tolist():
            if tid == -1: continue
            words = top_terms_from_topic(best["model"].get_topic(tid), 15)
            rows.append({"model":"BERTopic","topic_id":int(tid),"top_terms":", ".join(words)})
        pd.DataFrame(rows).to_csv(os.path.join(bestdir, "topics_bertopic.csv"), index=False)

        # doc→topic
        doc = load_corpus(args.inputs)  # same order
        doc["topic_id"] = best["topics"]
        doc.to_csv(os.path.join(bestdir, "doc_topics_bertopic.csv"), index=False)

        # prevalence
        grp = doc.groupby(["company","year","topic_id"], dropna=False).size().reset_index(name="pages")
        grp["year_pages"] = grp.groupby(["company","year"])["pages"].transform("sum")
        grp["share"] = grp["pages"] / grp["year_pages"]
        grp.sort_values(["company","year","topic_id"]).to_csv(os.path.join(bestdir, "topic_prevalence_bertopic.csv"), index=False)

        # metrics row for best
        pd.DataFrame([best["params"]]).to_csv(os.path.join(bestdir, "best_params.json.csv"), index=False)
        with open(os.path.join(bestdir, "best_params.json"), "w") as f:
            json.dump(best["params"], f, indent=2)
        print(f"[BEST] Saved best artifacts under {bestdir}")

if __name__ == "__main__":
    main()
