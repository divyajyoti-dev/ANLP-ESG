#!/usr/bin/env python3
# compare_lda_bert_metrics.py
# Robust: parses many LDA schemas, cleans tokens, compares vs BERTopic.

import os, re, ast
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from bertopic import BERTopic

COMPANIES = ["apple","google","amazon","intel","meta","microsoft","nvidia"]
TOP_K = 10

# ---------- text loaders ----------
def load_docs(company):
    path = f"data/processed/{company}_layout_all/corpus_pages_cleaned.csv"
    if not os.path.exists(path):
        return [], []
    df = pd.read_csv(path, usecols=["page_text_final"])
    docs = df["page_text_final"].astype(str).tolist()
    # simple tokenize: lowercase + split on non-letters
    tokens = [re.findall(r"[a-z]+", d.lower()) for d in docs]
    # drop empties
    tokens = [t for t in tokens if len(t) >= 3]
    return docs, tokens

# ---------- cleaning helpers ----------
def _ensure_token_list(x):
    """Return a clean list[str] of tokens from various input types."""
    if isinstance(x, (list, tuple)):
        cand = []
        for y in x:
            if isinstance(y, str):
                cand.extend(re.findall(r"[a-z]+", y.lower()))
            else:
                cand.append(str(y).lower())
        return [t for t in cand if t]
    if isinstance(x, str):
        # try python-list string first
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                val = ast.literal_eval(s)
                return _ensure_token_list(val)
            except Exception:
                pass
        # else split on separators or non-letters
        parts = re.split(r"[;,|\t]| {2,}", s)
        toks = []
        for p in parts:
            toks.extend(re.findall(r"[a-z]+", p.lower()))
        return [t for t in toks if t]
    return []

def clean_topics(topics, k=TOP_K):
    """topics -> list[list[str]] trimmed, deduped per-topic, no empties."""
    out = []
    for t in topics:
        toks = _ensure_token_list(t)
        # keep order but remove duplicates
        seen, uniq = set(), []
        for tok in toks:
            if tok and tok not in seen:
                seen.add(tok); uniq.append(tok)
        uniq = uniq[:k]
        if uniq:
            out.append(uniq)
    # need at least 2 topics for C_v to be meaningful
    return out if len(out) >= 2 else []

# ---------- LDA loaders (many schemas) ----------
def _from_long_topic_word(df):
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("topic")
    wcol = cols.get("word") or cols.get("term")
    if tcol and wcol:
        tops = []
        for t in sorted(df[tcol].dropna().unique()):
            words = df[df[tcol]==t][wcol].dropna().astype(str).tolist()
            tops.append(words)
        return tops
    return None

def _from_wide_top_words(df):
    term_cols = [c for c in df.columns if re.fullmatch(r"(term|top|word)_?\d+", c, re.I)]
    if term_cols:
        term_cols = sorted(term_cols, key=lambda s: int(re.findall(r"\d+", s)[0]))
        tops = []
        for _, row in df.iterrows():
            tops.append([row[c] for c in term_cols if pd.notnull(row[c])])
        return tops
    return None

def _from_concat_or_list_column(df):
    # look for a single column containing delimited or list-like strings
    for name in df.columns:
        lname = name.lower()
        if lname in {"top_words","topterms","keywords","terms","topic_words","topwords"}:
            return df[name].dropna().astype(str).tolist()
    # fallback: if there are exactly 2 columns: topic_id + a combined column
    if df.shape[1] == 2:
        cands = [c for c in df.columns if df[c].dtype == object]
        if cands:
            return df[cands[0]].dropna().astype(str).tolist()
    return None

def lda_top_words(company):
    # primary file seen in your repo:
    paths = [
        f"reports/topics_{company}/topics_lda.csv",           # your file
        f"reports/topics_{company}/topic_terms.csv",          # alt
        f"reports/topics_{company}/lda_topic_terms.csv"       # alt
    ]
    path = next((p for p in paths if os.path.exists(p)), None)
    if not path:
        return []

    df = pd.read_csv(path)

    # try parsers in order
    for parser in (_from_long_topic_word, _from_wide_top_words):
        tops = parser(df.copy())
        if tops:
            return clean_topics(tops, k=TOP_K)

    # try list/concat column
    colvals = _from_concat_or_list_column(df.copy())
    if colvals:
        # colvals could be per-topic strings -> clean each into tokens
        tops = [colvals] if isinstance(colvals, str) else colvals
        tops = [ _ensure_token_list(s) for s in tops ]
        return clean_topics(tops, k=TOP_K)

    # worst-case heuristic: treat each row as a topic, all object columns as terms
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        tops = []
        for _, row in df[obj_cols].iterrows():
            vals = []
            for c in obj_cols:
                vals.extend(_ensure_token_list(row[c]))
            if vals:
                tops.append(vals)
        return clean_topics(tops, k=TOP_K)

    return []

# ---------- BERTopic loader ----------
def bert_top_words(company):
    pref = f"reports/topics_{company}_bt_refined/bertopic_model"
    fb   = f"reports/topics_{company}_bt/bertopic_model"
    path = pref if os.path.exists(pref) else fb
    if not os.path.exists(path):
        return []
    model = BERTopic.load(path)
    topics = model.get_topics()
    tops = []
    for tid, words in topics.items():
        if tid == -1:
            continue
        tops.append([w for (w, _) in words])
    return clean_topics(tops, k=TOP_K)

# ---------- metrics ----------
def compute_coherence(tokens, topic_word_lists):
    if not topic_word_lists:
        return None
    dictionary = Dictionary(tokens)
    if len(dictionary) == 0:
        return None
    cm = CoherenceModel(topics=topic_word_lists, texts=tokens, dictionary=dictionary, coherence='c_v')
    return cm.get_coherence()

def topic_diversity(topic_word_lists):
    if not topic_word_lists:
        return None
    flat = [w for t in topic_word_lists for w in t[:TOP_K]]
    denom = len(topic_word_lists) * TOP_K
    if denom == 0:
        return None
    return len(set(flat)) / denom

# ---------- main ----------
def main():
    rows = []
    for c in COMPANIES:
        docs, tokens = load_docs(c)
        if not tokens:
            continue

        lda_tops  = lda_top_words(c)
        bert_tops = bert_top_words(c)

        lda_cv   = compute_coherence(tokens, lda_tops)   if lda_tops  else None
        lda_div  = topic_diversity(lda_tops)             if lda_tops  else None
        bert_cv  = compute_coherence(tokens, bert_tops)  if bert_tops else None
        bert_div = topic_diversity(bert_tops)            if bert_tops else None

        rows.append({
            "company": c,
            "lda_coherence": lda_cv,
            "lda_diversity": lda_div,
            "bert_coherence": bert_cv,
            "bert_diversity": bert_div
        })

        print(f"✓ {c}: "
              f"LDA(cv={None if lda_cv is None else round(lda_cv,3)}, "
              f"div={None if lda_div is None else round(lda_div,3)}) | "
              f"BERT(cv={None if bert_cv is None else round(bert_cv,3)}, "
              f"div={None if bert_div is None else round(bert_div,3)})")

    out = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/model_metrics_summary_final.csv", index=False)
    print("\nSaved → reports/model_metrics_summary_final.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
