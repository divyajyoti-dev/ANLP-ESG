#!/usr/bin/env python3
# make_esg_segments_all.py
import glob, pandas as pd, os

paths = sorted(glob.glob("data/processed/*_layout_all/corpus_pages_cleaned.csv"))
frames = []
for p in paths:
    try:
        df = pd.read_csv(p, usecols=["page_text_final"])
    except Exception:
        continue
    df = df.rename(columns={"page_text_final":"text"})
    df["text"] = (df["text"].astype(str)
                          .str.replace(r"\s+"," ", regex=True)
                          .str.strip())
    df = df[df["text"].str.split().str.len().ge(5)]
    frames.append(df)

out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["text"])
os.makedirs("data/processed", exist_ok=True)
out.to_csv("data/processed/esg_segments_all.csv", index=False)
print(f"Wrote {len(out)} rows → data/processed/esg_segments_all.csv")
