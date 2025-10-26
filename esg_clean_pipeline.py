
"""
FINAL APPLE + UNIVERSAL ESG CLEANER
Removes repeating headers/footers + year-specific Apple footers (2015–2025).
"""

import re, os, math, json, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image
import pytesseract

# ---------------- CLI ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--company", required=True)
parser.add_argument("--base", default="data/processed")
parser.add_argument("--ocr", action="store_true")
args = parser.parse_args()

BASE = Path(args.base) / f"{args.company}_layout_all"
OUT = BASE / "corpus_pages_cleaned.csv"
PAT = BASE / "auto_hf_patterns.json"

# ---------------- HARDCODED APPLE FOOTERS ----------------
APPLE_FOOTERS = {
    2015: [r"(?mi)\d*\s*environmental\s+responsibility\s+report\s*\|\s*2015\s+progress\s+report.*fy2014"],
    2016: [r"(?mi)\d*\s*environmental\s+responsibility\s+report\s*\|\s*2016\s+progress\s+report.*fy2015"],
    2017: [r"(?mi)\d*\s*environmental\s+responsibility\s+report\s*\|\s*2017\s+progress\s+report.*fy2016"],
    2018: [r"(?mi)\d*\s*environmental\s+responsibility\s+report\s*\|\s*2018\s+progress\s+report.*fy2017"],
    2019: [r"(?mi)introduction\s+climate\s+change\s+resources\s+smarter\s+chemistry\s+better\s+together\s+2019\s+environmental\s+responsibility\s+report"],
    2020: [
        r"(?mi)introduction\s+climate\s+change\s+resources\s+smarter\s+chemistry\s+appendix",
        r"(?mi)2020\s+environmental\s+progress\s+report",
        r"(?mi)apple\.com/environment"
    ],
    2021: [r"(?mi)introduction\s+climate\s+change\s+resources\s+smarter\s+chemistry\s+appendix\s+environmental\s+progress\s+report"],
    2022: [r"(?mi)introduction\s+environment\s+our\s+people\s+suppliers\s+customers\s+communities\s+governance\s+appendix\s+apple’s\s+2022\s+esg\s+report"],
    2023: [r"(?mi)introduction\s+climate\s+change\s+resources\s+smarter\s+chemistry\s+engagement\s+and\s+advocacy\s+appendix\s+2023\s+environmental\s+progress\s+report"],
    2024: [r"(?mi)contents\s+introduction\s+environmental\s+initiatives\s+engagement\s+and\s+advocacy\s+data\s+appendix\s+2024\s+environmental\s+progress\s+report"],
    2025: [r"(?mi)contents\s+introduction\s+environmental\s+initiatives\s+engagement\s+and\s+advocacy\s+data\s+green\s+bond\s+impact\s+report\s+appendix\s+2025\s+environmental\s+progress\s+report"],
}

# ---------------- LOAD ----------------
def safe_read(path): return pd.read_csv(path) if path.exists() else pd.DataFrame()
pages = safe_read(BASE / "pages.csv")
blocks = safe_read(BASE / "blocks.csv")

if pages.empty:
    raise ValueError(f"No pages.csv found under {BASE}")
if "report_name" not in pages.columns:
    pages["report_name"] = pages["year"].apply(lambda y: f"report_{int(y)}.pdf")

# ---------------- AUTO MINER ----------------
def _norm(s):
    s = re.sub(r"\s+", " ", s.strip().lower())
    return re.sub(r"\b\d{3,}\b", "<num>", s)
def _flex(s):
    s = re.escape(s).replace(r"\ ", r"\s+").replace(re.escape("<num>"), r"\d+")
    return rf"(?mi)^\s*{s}\s*$"

def mine_hf(df_pages, df_blocks):
    out = {}
    geom = not df_blocks.empty and all(c in df_blocks.columns for c in ["y0","x0"])
    for rep, grp in df_pages.groupby("report_name"):
        top, bot = [], []
        if geom:
            for (y, p), bdf in df_blocks[df_blocks["report_name"]==rep].groupby(["year","page"]):
                bdf = bdf.sort_values(["y0","x0"])
                top.append(_norm(" ".join(bdf.head(2)["text"].fillna(""))))
                bot.append(_norm(" ".join(bdf.tail(2)["text"].fillna(""))))
        else:
            for _, r in grp.iterrows():
                raw = str(r.get("text_all") or "")
                lines = [ln for ln in raw.splitlines() if ln.strip()]
                if not lines: continue
                top.append(_norm(" ".join(lines[:2])))
                bot.append(_norm(" ".join(lines[-2:])))
        n = max(1,len(grp))
        min_support = max(3, math.ceil(0.25*n))
        tc, bc = Counter(top), Counter(bot)
        tkeep = [x for x,c in tc.items() if c>=min_support and len(x)>8]
        bkeep = [x for x,c in bc.items() if c>=min_support and len(x)>8]
        out[rep] = {"header":[_flex(x) for x in tkeep],
                    "footer":[_flex(x) for x in bkeep]}
    return out

# ---------------- CLEANERS ----------------
def fix_ligatures(t): return t.replace("ﬁ","fi").replace("ﬂ","fl").replace("\u00a0"," ")
def merge_hyphens(t): return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
def unwrap(t):
    lines=t.split("\n"); out=[]
    for i,l in enumerate(lines):
        out.append(l.rstrip())
        if i<len(lines)-1: out.append(" " if l and l[-1] not in ".!?)" else "\n")
    return re.sub(r"\n{3,}","\n\n","".join(out))
def normspace(t):
    t=re.sub(r"[ \t]{2,}"," ",t)
    t=re.sub(r"\s+\n","\n",t)
    return t.strip()
def strip_patterns(t,pats):
    for p in pats:
        try:t=re.sub(p,"",t)
        except re.error: continue
    return t

def clean_text(text, year, pats):
    if not isinstance(text, str): return ""
    t = fix_ligatures(text)
    t = merge_hyphens(t)
    t = unwrap(t)
    t = normspace(t)

    base_patterns = [
        r"(?mi)^\s*\d+\s*$",              # bare page numbers
        r"(?mi)\bpage\s*\d+\b",
        r"(?mi)©\s*\d{4}.*apple",
        r"(?mi)https?://\S+",
    ]
    all_pats = base_patterns + pats.get("header",[]) + pats.get("footer",[])

    # Append hardcoded Apple footer if applicable
    if year in APPLE_FOOTERS:
        all_pats += APPLE_FOOTERS[year]

    # Remove them all
    t = strip_patterns(t, all_pats)
    t = normspace(t)
    return t.strip()

def ocr_fallback(img):
    try: return pytesseract.image_to_string(Image.open(img))
    except: return ""

# ---------------- MAIN ----------------
print(f"[info] Running cleaner for {args.company}")
patterns = mine_hf(pages, blocks)
with open(PAT, "w") as f: json.dump(patterns, f, indent=2)

records=[]
for _,r in pages.iterrows():
    year=int(r["year"]); page=int(r["page"])
    text = str(r.get("text_all") or "")
    if len(text)<200 and args.ocr:
        img = BASE/"page_images"/f"{r['report_name']}_{page}.png"
        text = ocr_fallback(img)
    pats = patterns.get(r["report_name"], {})
    final = clean_text(text, year, pats)
    records.append({"report_name":r["report_name"],"year":year,"page":page,"page_text_final":final})

out=pd.DataFrame(records)
out.to_csv(OUT,index=False)
print(f"[done] Cleaned corpus saved → {OUT}")
