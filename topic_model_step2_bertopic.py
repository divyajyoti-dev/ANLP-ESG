#!/usr/bin/env python3
"""
BERTopic (mini, macOS-friendly)
- Inputs: one or more .../corpus_pages_cleaned.csv
- Outputs (in --out-dir):
    topics_bertopic.csv
    topics_bertopic_info.csv
    doc_topics_bertopic.csv
    topic_prevalence_bertopic.csv
    metrics_bertopic.csv
    (optional) bertopic_model/  if --save-model is passed
"""

import argparse, os, re, random, numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--emb", default="all-MiniLM-L6-v2")   # compact model
    ap.add_argument("--min-topic-size", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Ensure reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    def infer_company(p: str) -> str:
        m = re.search(r"/processed/([^/]+)_layout_all/", p.replace("\\","/"))
        return m.group(1) if m else "Unknown"

    # ---------- Load corpora ----------
    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        if "company" not in df.columns:
            df["company"] = infer_company(p)
        need = ["report_name","year","page","page_text_final","company"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise SystemExit(f"Missing columns in {p}: {miss}")
        sub = df[need].rename(columns={"page_text_final": "text"}).copy()
        sub["year"] = pd.to_numeric(sub["year"], errors="coerce").astype("Int64")
        sub["text"] = sub["text"].fillna("").astype(str)
        frames.append(sub)

    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus[corpus["text"].str.strip().str.len() > 0].reset_index(drop=True)
    if corpus.empty:
        raise SystemExit("All documents empty after filtering.")
    print(f"[INFO] Loaded {len(corpus)} pages from {len(args.inputs)} file(s).")

    # ---------- BERTopic ----------
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics import silhouette_score
    except Exception:
        raise SystemExit("Install deps: pip install 'bertopic[all]' 'sentence-transformers>=2.2.2' umap-learn hdbscan scikit-learn")

    print(f"[BT] Embedding with {args.emb} (batch={args.batch_size}) …")
    st = SentenceTransformer(args.emb)
    embs = st.encode(
        corpus["text"].tolist(),
        show_progress_bar=True,
        batch_size=args.batch_size
    )

    print(f"[BT] Fitting BERTopic (min_topic_size={args.min_topic_size}) …")
    bt = BERTopic(
        embedding_model=st,
        min_topic_size=args.min_topic_size,
        verbose=True
    )
    topics, probs = bt.fit_transform(corpus["text"].tolist(), embs)

    if args.save_model:
        bt.save(os.path.join(args.out_dir, "bertopic_model"))

    # ---------- Save outputs ----------
    info = bt.get_topic_info()
    info.to_csv(os.path.join(args.out_dir, "topics_bertopic_info.csv"), index=False)

    # Top terms per topic (handle both DataFrame and list-of-tuples APIs)
    def top_terms_from_topic(topic_obj, k=15):
        # Newer BERTopic often returns list[ (word, score), ... ]
        if isinstance(topic_obj, list):
            return [w for (w, _score) in topic_obj[:k]]
        # Some versions return a DataFrame with 'Word' column
        if hasattr(topic_obj, "head") and "Word" in getattr(topic_obj, "columns", []):
            return topic_obj["Word"].head(k).astype(str).tolist()
        # Fallback: try to coerce to strings
        return [str(x) for x in list(topic_obj)[:k]]

    rows = []
    for tid in info["Topic"].tolist():
        if tid == -1:
            continue
        topic_obj = bt.get_topic(tid)
        words = ", ".join(top_terms_from_topic(topic_obj, 15))
        rows.append({"model": "BERTopic", "topic_id": int(tid), "top_terms": words})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "topics_bertopic.csv"), index=False)

    # Document→topic mapping
    doc_bt = corpus.copy()
    doc_bt["topic_id"] = topics
    doc_bt.to_csv(os.path.join(args.out_dir, "doc_topics_bertopic.csv"), index=False)

    # Topic prevalence by year/company
    def prevalence(df: pd.DataFrame, topic_col: str) -> pd.DataFrame:
        grp = df.groupby(["company", "year", topic_col], dropna=False).size().reset_index(name="pages")
        grp["year_pages"] = grp.groupby(["company", "year"])["pages"].transform("sum")
        grp["share"] = grp["pages"] / grp["year_pages"]
        return grp.sort_values(["company", "year", topic_col])

    prevalence(doc_bt, "topic_id").to_csv(
        os.path.join(args.out_dir, "topic_prevalence_bertopic.csv"), index=False
    )

    # ---------- Metrics ----------
    def topic_diversity(word_lists, topn=10):
        flat = [w for lst in word_lists for w in lst[:topn]]
        return len(set(flat)) / max(1, len(flat))

    tw = pd.read_csv(os.path.join(args.out_dir, "topics_bertopic.csv"))
    wordlists = [[w.strip() for w in str(s).split(",")] for s in tw["top_terms"].tolist()]
    diversity = topic_diversity(wordlists, topn=10)

    lab = np.array(topics)
    mask = lab != -1
    sil = float("nan")
    if mask.sum() > 1 and len(set(lab[mask])) > 1:
        sil = float(silhouette_score(np.asarray(embs)[mask], lab[mask]))

    pd.DataFrame([{"model": "BERTopic", "topic_diversity": diversity, "silhouette": sil}]) \
        .to_csv(os.path.join(args.out_dir, "metrics_bertopic.csv"), index=False)

    print("[BT] Done. Wrote outputs to:", args.out_dir)

if __name__ == "__main__":
    main()
