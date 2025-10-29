#!/usr/bin/env python3
"""
Universal ESG Topic Modeling Pipeline
- Loads one or many cleaned corpora: data/processed/*_layout_all/corpus_pages_cleaned.csv
- Runs LDA (baseline) and optional BERTopic
- Writes tidy CSVs for topics and doc-topic assignments
- Aggregates per-year and per-company topic prevalence

Usage examples:
  # Single company (Meta)
  python topic_pipeline.py --inputs data/processed/meta_layout_all/corpus_pages_cleaned.csv --out-dir reports/topics_meta

  # Apple + Meta together (compare trends)
  python  topic_pipeline.py --inputs data/processed/Apple_layout_all/corpus_pages_cleaned.csv data/processed/meta_layout_all/corpus_pages_cleaned.csv --out-dir reports/topics_apple_meta

  # Include BERTopic
  python  topic_pipeline.py --inputs data/processed/meta_layout_all/corpus_pages_cleaned.csv --out-dir reports/topics_meta --bertopic
"""

import argparse, os, re, sys, json, pathlib
from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True,
                help="Path(s) to corpus_pages_cleaned.csv for one or more companies.")
ap.add_argument("--out-dir", required=True, help="Directory to write results.")
ap.add_argument("--n-topics", type=int, default=12, help="LDA number of topics.")
ap.add_argument("--max-features", type=int, default=6000, help="Max vocab size for LDA.")
ap.add_argument("--min-df", type=int, default=5, help="Min doc freq for LDA vocab.")
ap.add_argument("--max-df", type=float, default=0.9, help="Max doc freq for LDA vocab.")
ap.add_argument("--bertopic", action="store_true", help="Also run BERTopic (requires extra deps).")
ap.add_argument("--emb", default="all-MiniLM-L6-v2", help="SentenceTransformer name for BERTopic.")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ---------- Load & unify ----------
def infer_company_from_path(p: str) -> str:
    # expects .../processed/<Company>_layout_all/corpus_pages_cleaned.csv
    m = re.search(r"/processed/([^/]+)_layout_all/", p.replace("\\","/"))
    return m.group(1) if m else "Unknown"

dfs = []
for p in args.inputs:
    df = pd.read_csv(p)
    if "company" not in df.columns:
        df["company"] = infer_company_from_path(p)
    # Standardize column names if needed
    if "page_text_final" not in df.columns:
        # fallback to "text" if someone passes a different file
        text_col = "page_text_final" if "page_text_final" in df.columns else "text"
    else:
        text_col = "page_text_final"
    needed = ["report_name","year","page",text_col,"company"]
    missing = [c for c in ["report_name","year","page"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required cols in {p}: {missing}")
    df = df.rename(columns={text_col: "text"})
    df["company"] = df["company"].astype(str)
    dfs.append(df[["report_name","year","page","text","company"]])

corpus = pd.concat(dfs, ignore_index=True)
corpus["text"] = corpus["text"].fillna("").astype(str)
corpus["year"] = pd.to_numeric(corpus["year"], errors="coerce").astype("Int64")

# Filter out empty docs
corpus = corpus[corpus["text"].str.strip().str.len() > 0].reset_index(drop=True)
if corpus.empty:
    raise SystemExit("All documents are empty after filtering.")

# ---------- LDA (baseline) ----------
print("[LDA] Vectorizing…")
vect = CountVectorizer(
    stop_words="english",
    max_df=args.max_df,
    min_df=args.min_df,
    max_features=args.max_features
)
X = vect.fit_transform(corpus["text"])

print(f"[LDA] Fitting LDA (n_topics={args.n_topics})…")
lda = LatentDirichletAllocation(n_components=args.n_topics, random_state=args.seed, learning_method="batch")
lda_doc_topic = lda.fit_transform(X)
terms = vect.get_feature_names_out()

def top_terms_for_topic(topic_idx: int, topn: int = 15) -> List[str]:
    comp = lda.components_[topic_idx]
    idx = comp.argsort()[-topn:][::-1]
    return [terms[j] for j in idx]

# Topics table
lda_topics = []
for k in range(args.n_topics):
    lda_topics.append({
        "model": "LDA",
        "topic_id": k,
        "top_terms": ", ".join(top_terms_for_topic(k, 15))
    })
lda_topics_df = pd.DataFrame(lda_topics)
lda_topics_df.to_csv(os.path.join(args.out_dir, "topics_lda.csv"), index=False)

# Doc→topic assignments (argmax)
lda_assign = pd.DataFrame(lda_doc_topic).idxmax(axis=1)
doc_topics_lda = corpus.copy()
doc_topics_lda["topic_id"] = lda_assign
doc_topics_lda.to_csv(os.path.join(args.out_dir, "doc_topics_lda.csv"), index=False)

# Year/company prevalence
def prevalence(df: pd.DataFrame, topic_col: str) -> pd.DataFrame:
    grp = df.groupby(["company","year",topic_col], dropna=False, observed=True).size().reset_index(name="pages")
    # normalize within (company, year)
    grp["year_pages"] = grp.groupby(["company","year"])["pages"].transform("sum")
    grp["share"] = grp["pages"] / grp["year_pages"]
    return grp.sort_values(["company","year",topic_col])

prev_lda = prevalence(doc_topics_lda, "topic_id")
prev_lda.to_csv(os.path.join(args.out_dir, "topic_prevalence_lda.csv"), index=False)

print("[LDA] Done. Wrote topics_lda.csv, doc_topics_lda.csv, topic_prevalence_lda.csv")

# ---------- BERTopic (optional) ----------
if args.bertopic:
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        import umap  # noqa
        import hdbscan  # noqa
    except Exception as e:
        print("[BERTopic] Missing dependencies. Install: pip install 'bertopic[all]' 'sentence-transformers>=2.2.2' umap-learn hdbscan")
        sys.exit(1)

    print(f"[BERTopic] Embedding model: {args.emb}")
    st_model = SentenceTransformer(args.emb)
    bt = BERTopic(embedding_model=st_model, language="english", verbose=True)
    topics, probs = bt.fit_transform(corpus["text"].tolist())

    info = bt.get_topic_info()
    info.to_csv(os.path.join(args.out_dir, "topics_bertopic_info.csv"), index=False)

    # Per-topic top terms
    rows = []
    for tid in info["Topic"].tolist():
        if tid == -1:
            continue
        terms_df = bt.get_topic(tid)
        top_terms = ", ".join(terms_df["Word"].head(15).tolist())
        rows.append({"model":"BERTopic","topic_id":tid,"top_terms":top_terms})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "topics_bertopic.csv"), index=False)

    # Doc→topic
    doc_bt = corpus.copy()
    doc_bt["topic_id"] = topics
    doc_bt.to_csv(os.path.join(args.out_dir, "doc_topics_bertopic.csv"), index=False)

    # Prevalence by year/company
    prev_bt = prevalence(doc_bt, "topic_id")
    prev_bt.to_csv(os.path.join(args.out_dir, "topic_prevalence_bertopic.csv"), index=False)

    # Save model (for later inspection/visualization)
    bt.save(os.path.join(args.out_dir, "bertopic_model"))
    print("[BERTopic] Done. Wrote topics_bertopic.csv, doc_topics_bertopic.csv, topic_prevalence_bertopic.csv and saved model.")
