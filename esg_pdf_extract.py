#!/usr/bin/env python3
"""
FAST ESG PDF → Page text extractor (simple)
-------------------------------------------
- Parallel per-PDF with --workers
- Sharding with --shards / --shard-index
- Optional bitmap OCR (off by default); run only if text layer short
- Lower default DPI (200)
- Per-PDF outputs + auto-concat

Outputs:
  out_dir/per_pdf/<pdf>/pages.csv
  out_dir/pages.csv (concat)

Usage example:
  python esg_pdf_extract.py \
    --download-dir data/raw/apple_all \
    --out data/processed/apple_pages_all \
    --company Apple \
    --workers 6 --dpi 200 --min-text-chars 200
"""

import os, re, csv, sys, json, math, hashlib, argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor, as_completed

import fitz
import pandas as pd
from tqdm import tqdm

try:
    from pdf2image import convert_from_path
    from PIL import Image
    HAVE_IMG = True
except Exception:
    HAVE_IMG = False

try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False

YEAR_RE = re.compile(r"(20\d{2}|201[5-9])")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sha1_path(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def infer_year_from_filename(fname: str):
    m = YEAR_RE.search(fname)
    return int(m.group(1)) if m else None

def bitmap_ocr_page(pdf_path: str, page_num: int, dpi: int = 200) -> str:
    if not (HAVE_IMG and HAVE_TESS):
        return ""
    try:
        imgs = convert_from_path(pdf_path, dpi=dpi, first_page=page_num+1, last_page=page_num+1)
        if imgs:
            return pytesseract.image_to_string(imgs[0])
        return ""
    except Exception:
        return ""

def extract_pdf(pdf_path: str, company: str, bitmap_ocr: bool, dpi: int, min_text_chars: int, out_dir: str) -> str:
    per_pdf_dir = os.path.join(out_dir, "per_pdf", os.path.splitext(os.path.basename(pdf_path))[0])
    ensure_dir(per_pdf_dir)

    doc = fitz.open(pdf_path)
    file_id = sha1_path(pdf_path)
    fname = os.path.basename(pdf_path)
    year = infer_year_from_filename(fname)

    rows: List[Dict[str, Any]] = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        row = {
            "company": company,
            "year": year,
            "pdf_file": fname,
            "file_sha1": file_id,
            "page": i,
            "text": text.strip(),
            "bitmap_ocr_text": ""
        }
        if bitmap_ocr and len(text.strip()) < min_text_chars:
            row["bitmap_ocr_text"] = bitmap_ocr_page(pdf_path, i, dpi=dpi)
        rows.append(row)

    outp = os.path.join(per_pdf_dir, "pages.csv")
    pd.DataFrame(rows).to_csv(outp, index=False)
    return outp

def concat_per_pdf(out_dir: str):
    per_dir = os.path.join(out_dir, "per_pdf")
    dfs = []
    for root, dirs, files in os.walk(per_dir):
        if "pages.csv" in files:
            dfs.append(pd.read_csv(os.path.join(root, "pages.csv")))
    if dfs:
        outp = os.path.join(out_dir, "pages.csv")
        pd.concat(dfs, ignore_index=True).to_csv(outp, index=False)
        print(f"[CONCAT] wrote {outp} ({len(dfs)} parts)")

def shard(items, shards, idx):
    if shards <= 1: return items
    n = len(items)
    per = math.ceil(n / shards)
    start = idx * per
    end = min(n, start + per)
    return items[start:end]

def main():
    ap = argparse.ArgumentParser(description="FAST ESG PDF → Page text extractor")
    ap.add_argument("--download-dir", required=True, help="Folder with PDFs")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--company", default="Apple")
    ap.add_argument("--bitmap-ocr", action="store_true", help="Enable bitmap OCR (off by default)")
    ap.add_argument("--dpi", type=int, default=200, help="DPI for bitmap OCR")
    ap.add_argument("--min-text-chars", type=int, default=200, help="Only bitmap OCR when text layer short")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="Skip PDFs already processed")
    args = ap.parse_args()

    ensure_dir(args.out)
    pdfs = sorted([p for p in os.listdir(args.download_dir) if p.lower().endswith(".pdf")])
    if not pdfs:
        print(f"No PDFs found in {args.download_dir}.", file=sys.stderr)
        sys.exit(1)

    # shard
    pdfs = shard(pdfs, args.shards, args.shard_index)

    # resume
    if args.resume:
        keep = []
        for f in pdfs:
            per_pdf_dir = os.path.join(args.out, "per_pdf", os.path.splitext(f)[0])
            if not os.path.exists(os.path.join(per_pdf_dir, "pages.csv")):
                keep.append(f)
        pdfs = keep

    if not pdfs:
        print("[INFO] Nothing to do after resume/shard filtering.")
        concat_per_pdf(args.out)
        return

    def _task(fname):
        return extract_pdf(
            pdf_path=os.path.join(args.download_dir, fname),
            company=args.company,
            bitmap_ocr=args.bitmap_ocr,
            dpi=args.dpi,
            min_text_chars=args.min_text_chars,
            out_dir=args.out
        )

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_task, f): f for f in pdfs}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                fut.result()
                print(f"[OK] {f}")
            except Exception as e:
                print(f"[WARN] {f} failed: {e}")

    concat_per_pdf(args.out)

if __name__ == "__main__":
    main()
