#!/usr/bin/env python3
"""
Step 1: Baseline Topic Modeling for ESG corpus (macOS/Windows safe)
- Inputs: one or more .../corpus_pages_cleaned.csv files
- Models: LDA (CountVectorizer), NMF (TfidfVectorizer)
- Outputs (in --out-dir):
    topics_lda.csv, doc_topics_lda.csv, topic_prevalence_lda.csv, metrics_lda.csv
    topics_nmf.csv, doc_topics_nmf.csv, topic_prevalence_nmf.csv, metrics_nmf.csv

Run examples:
  python topic_model_step1.py --inputs data/processed/google_layout_all/corpus_pages_cleaned.csv --out-dir reports/topics_google --n-topics 12
  python topic_model_step1.py --inputs data/processed/amazon_layout_all/corpus_pages_cleaned.csv data/processed/microsoft_layout_all/corpus_pages_cleaned.csv --out-dir reports/topics_amzn_msft --n-topics 12
"""

import argparse, os, re, sys
from typing import List
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

def main():
    # ---------- CLI ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Path(s) to corpus_pages_cleaned.csv")
    ap.add_argument("--out-dir", required=True, help="Where to write outputs")
    ap.add_argument("--n-topics", type=int, default=12)
    ap.add_argument("--max-features", type=int, default=6000)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load & unify ----------
    def infer_company(p: str) -> str:
        m = re.search(r"/processed/([^/]+)_layout_all/", p.replace("\\", "/"))
        return m.group(1) if m else "Unknown"

    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        if "company" not in df.columns:
            df["company"] = infer_company(p)
        need = ["report_name","year","page","page_text_final","company"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing columns in {p}: {missing}")
        sub = df[need].rename(columns={"page_text_final":"text"}).copy()
        sub["year"] = pd.to_numeric(sub["year"], errors="coerce").astype("Int64")
        sub["text"] = sub["text"].fillna("").astype(str)
        frames.append(sub)

    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus[corpus["text"].str.strip().str.len() > 0].reset_index(drop=True)
    if corpus.empty:
        raise SystemExit("All documents are empty after filtering.")
    print(f"[INFO] Loaded {len(corpus)} pages from {len(args.inputs)} file(s).")

    # ---------- Helpers ----------
    def prevalence(df: pd.DataFrame, topic_col: str) -> pd.DataFrame:
        grp = df.groupby(["company","year",topic_col], dropna=False).size().reset_index(name="pages")
        grp["year_pages"] = grp.groupby(["company","year"])["pages"].transform("sum")
        grp["share"] = grp["pages"] / grp["year_pages"]
        return grp.sort_values(["company","year",topic_col])

    def topic_diversity(word_lists: List[List[str]], topn: int = 10) -> float:
        flat = [w for lst in word_lists for w in lst[:topn]]
        return len(set(flat)) / max(1, len(flat))

    def write_metrics(model_name: str, top_words: List[List[str]]):
        metrics = {
            "model": model_name,
            "topic_diversity": topic_diversity(top_words, topn=10),
            "coherence_cv": np.nan  # reserved; add gensim later if desired
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(args.out_dir, f"metrics_{model_name.lower()}.csv"), index=False)
        print(f"[METRICS] {model_name} → {metrics}")

    # ---------- LDA ----------
    print("[LDA] Vectorizing (bag-of-words)…")
    cv = CountVectorizer(stop_words="english", max_df=args.max_df, min_df=args.min_df, max_features=args.max_features)
    X_counts = cv.fit_transform(corpus["text"])
    terms_counts = cv.get_feature_names_out()

    print(f"[LDA] Fitting (n_topics={args.n_topics})…")
    lda = LatentDirichletAllocation(n_components=args.n_topics, random_state=args.seed, learning_method="batch")
    theta = lda.fit_transform(X_counts)   # doc→topic
    beta = lda.components_               # topic→term

    def lda_top_terms(k, topn=15):
        idx = beta[k].argsort()[-topn:][::-1]
        return [terms_counts[j] for j in idx]

    lda_topics_rows, lda_wordlists = [], []
    for k in range(args.n_topics):
        words = lda_top_terms(k, 15)
        lda_wordlists.append(words)
        lda_topics_rows.append({"model":"LDA","topic_id":k,"top_terms":", ".join(words)})
    pd.DataFrame(lda_topics_rows).to_csv(os.path.join(args.out_dir, "topics_lda.csv"), index=False)

    doc_topics_lda = corpus.copy()
    doc_topics_lda["topic_id"] = pd.DataFrame(theta).idxmax(axis=1)
    doc_topics_lda.to_csv(os.path.join(args.out_dir, "doc_topics_lda.csv"), index=False)
    prevalence(doc_topics_lda, "topic_id").to_csv(os.path.join(args.out_dir, "topic_prevalence_lda.csv"), index=False)

    write_metrics("LDA", lda_wordlists)

    # ---------- NMF ----------
    print("[NMF] Vectorizing (tf-idf)…")
    tv = TfidfVectorizer(stop_words="english", max_df=args.max_df, min_df=args.min_df, max_features=args.max_features)
    X_tfidf = tv.fit_transform(corpus["text"])
    terms_tfidf = tv.get_feature_names_out()

    print(f"[NMF] Fitting (n_topics={args.n_topics})…")
    nmf = NMF(n_components=args.n_topics, random_state=args.seed, init="nndsvd", max_iter=400)
    W = nmf.fit_transform(X_tfidf)   # doc→topic
    H = nmf.components_              # topic→term

    def nmf_top_terms(k, topn=15):
        idx = H[k].argsort()[-topn:][::-1]
        return [terms_tfidf[j] for j in idx]

    nmf_topics_rows, nmf_wordlists = [], []
    for k in range(args.n_topics):
        words = nmf_top_terms(k, 15)
        nmf_wordlists.append(words)
        nmf_topics_rows.append({"model":"NMF","topic_id":k,"top_terms":", ".join(words)})
    pd.DataFrame(nmf_topics_rows).to_csv(os.path.join(args.out_dir, "topics_nmf.csv"), index=False)

    doc_topics_nmf = corpus.copy()
    doc_topics_nmf["topic_id"] = pd.DataFrame(W).idxmax(axis=1)
    doc_topics_nmf.to_csv(os.path.join(args.out_dir, "doc_topics_nmf.csv"), index=False)
    prevalence(doc_topics_nmf, "topic_id").to_csv(os.path.join(args.out_dir, "topic_prevalence_nmf.csv"), index=False)

    write_metrics("NMF", nmf_wordlists)

    print("[DONE] Wrote:")
    print("  topics_lda.csv, doc_topics_lda.csv, topic_prevalence_lda.csv, metrics_lda.csv")
    print("  topics_nmf.csv, doc_topics_nmf.csv, topic_prevalence_nmf.csv, metrics_nmf.csv")

if __name__ == "__main__":
    # Proper guard to avoid multiprocessing bootstrap errors on macOS/Windows
    main()