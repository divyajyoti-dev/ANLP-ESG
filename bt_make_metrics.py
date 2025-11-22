#!/usr/bin/env python3
# Rebuild metrics_bertopic.csv from existing BERTopic outputs.
import argparse, os, pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="Folder containing topics_bertopic.csv and doc_topics_bertopic.csv")
ap.add_argument("--emb", default="all-MiniLM-L6-v2")
args = ap.parse_args()

topics_path = os.path.join(args.dir, "topics_bertopic.csv")
doc_path    = os.path.join(args.dir, "doc_topics_bertopic.csv")
if not (os.path.exists(topics_path) and os.path.exists(doc_path)):
    raise SystemExit("Need topics_bertopic.csv AND doc_topics_bertopic.csv in --dir")

tw = pd.read_csv(topics_path)
def topic_diversity(word_lists, topn=10):
    flat=[w for lst in word_lists for w in lst[:topn]]
    return len(set(flat))/max(1,len(flat))
wordlists = [[w.strip() for w in str(s).split(",")] for s in tw["top_terms"].tolist()]
diversity = topic_diversity(wordlists, topn=10)

# silhouette (embed docs and use topic_id labels, ignoring -1 if present)
doc = pd.read_csv(doc_path)
labels = doc["topic_id"].to_numpy()
mask = labels != -1
sil = float("nan")
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import silhouette_score
    text = doc["text"] if "text" in doc.columns else None
    if text is None:
        # join back to corpus if stored separately; otherwise skip silhouette
        raise RuntimeError("No 'text' column found for embeddings; silhouette will be NaN.")
    st = SentenceTransformer(args.emb)
    embs = st.encode(text.astype(str).tolist(), show_progress_bar=True, batch_size=64)
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        sil = float(silhouette_score(np.asarray(embs)[mask], labels[mask]))
except Exception:
    pass

out = pd.DataFrame([{"model":"BERTopic","topic_diversity":diversity,"silhouette":sil}])
out.to_csv(os.path.join(args.dir, "metrics_bertopic.csv"), index=False)
print("Wrote", os.path.join(args.dir, "metrics_bertopic.csv"))
