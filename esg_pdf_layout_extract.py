#!/usr/bin/env python3
"""
FAST Layout-aware ESG PDF extractor (Apple-first)
-------------------------------------------------
Speed optimizations:
  - Parallel per-PDF with --workers (Mac-safe: top-level worker, no pickling issues)
  - Sharding with --shards / --shard-index
  - NO figure clip OCR (huge speed win)
  - Fallback OCR only when text layer is short (--min-text-chars)
  - Lower default DPI for fallback OCR (200)

Outputs (per PDF and concatenated):
  out_dir/per_pdf/<pdf_base>/pages.csv
  out_dir/per_pdf/<pdf_base>/blocks.csv
  out_dir/per_pdf/<pdf_base>/figures.csv
  out_dir/per_pdf/<pdf_base>/table_index.csv
  out_dir/pages.csv, blocks.csv, figures.csv, table_index.csv  (concatenated)

System deps:
  brew install tesseract poppler ghostscript
Python deps:
  pip install "PyMuPDF==1.26.5" "pytesseract==0.3.13" "pdf2image==1.17.0" \
              "opencv-python==4.10.0.84" "pillow==10.4.0" "pandas==2.2.3" \
              "tqdm==4.66.5" "pyarrow==17.0.0" "camelot-py[cv]==0.11.0"
"""

import os
import re
import argparse
import hashlib
import math
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Optional Camelot (tables)
try:
    import camelot
    HAVE_CAMELOT = True
except Exception:
    HAVE_CAMELOT = False

YEAR_RE = re.compile(r"(20\d{2}|201[5-9])")


# ---------------------- helpers ----------------------
def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()


def sha1_text(s: str) -> str:
    return sha1_bytes((s or "").encode("utf-8", "ignore"))


def pil_from_pixmap(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


def _ocr_try_modes(img: Image.Image, psms=(6, 11, 12)) -> str:
    """Try a few Tesseract PSMs and keep the longest."""
    best = ""
    for p in psms:
        try:
            txt = pytesseract.image_to_string(img, config=f"--psm {p}").strip()
        except Exception:
            txt = ""
        if len(txt) > len(best):
            best = txt
    return best


def page_ocr_fallback(page: fitz.Page, dpi: int = 200) -> str:
    """OCR a rasterized page (only when text layer is short)."""
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = pil_from_pixmap(pix)
        return _ocr_try_modes(img)
    except Exception:
        return ""


# ---------------------- extraction primitives ----------------------
def extract_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    """Text blocks with bbox + avg font size, sorted in reading order."""
    out: List[Dict[str, Any]] = []
    d = page.get_text("dict")
    block_id = 0
    for b in d.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        x0, y0, x1, y1 = b["bbox"]
        lines = b.get("lines", [])
        spans_txt, sizes = [], []
        for ln in lines:
            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if t and t.strip():
                    spans_txt.append(t)
                    sizes.append(sp.get("size", 0.0))
        if not spans_txt:
            continue
        txt = " ".join(spans_txt).strip()
        avg_size = float(np.mean(sizes)) if sizes else 0.0
        out.append({
            "block_id": block_id,
            "bbox_x0": x0, "bbox_y0": y0, "bbox_x1": x1, "bbox_y1": y1,
            "avg_font_size": avg_size,
            "text": txt,
            "text_hash": sha1_text(txt),
        })
        block_id += 1
    out.sort(key=lambda r: (round(r["bbox_y0"], 1), round(r["bbox_x0"], 1)))
    return out


def extract_images(page: fitz.Page) -> List[Dict[str, Any]]:
    """Return image regions with bbox + basic metadata (no OCR)."""
    imgs = []
    for idx, im in enumerate(page.get_images(full=True)):
        xref = im[0]
        try:
            bbox = page.get_image_bbox(xref)
        except Exception:
            bbox = fitz.Rect(0, 0, 0, 0)
        ihash, ext = "", ""
        try:
            raw = page.parent.extract_image(xref)
            ihash = sha1_bytes(raw.get("image", b""))
            ext = raw.get("ext", "")
        except Exception:
            pass
        imgs.append({
            "img_idx": idx, "xref": xref,
            "bbox_x0": bbox.x0, "bbox_y0": bbox.y0, "bbox_x1": bbox.x1, "bbox_y1": bbox.y1,
            "img_sha1": ihash, "img_ext": ext
        })
    return imgs


def nearest_caption(blocks: List[Dict[str, Any]], fig_rect: fitz.Rect,
                    below_px: float = 180, side_pad: float = 40) -> str:
    """Heuristic: nearest text block below the figure, horizontally overlapping."""
    cand = []
    for b in blocks:
        r = fitz.Rect(b["bbox_x0"], b["bbox_y0"], b["bbox_x1"], b["bbox_y1"])
        x_overlap = not (r.x1 < fig_rect.x0 + side_pad or r.x0 > fig_rect.x1 - side_pad)
        below = r.y0 >= fig_rect.y1 and (r.y0 - fig_rect.y1) < below_px
        if x_overlap and below:
            dy = r.y0 - fig_rect.y1
            cand.append((dy, b.get("avg_font_size", 0.0), b["text"]))
    if not cand:
        return ""
    cand.sort(key=lambda t: (t[0], t[1]))
    return cand[0][2].strip()


# ---------------------- tables via Camelot ----------------------
def extract_tables_with_camelot(pdf_path: str, page_num1: int, out_dir: str) -> List[Dict[str, Any]]:
    """Try 'lattice' first; if nothing, try 'stream'. Save to tables/*.csv."""
    meta = []
    if not HAVE_CAMELOT:
        return meta

    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    page_str = str(page_num1)

    def _save(tables, flavor):
        for i, t in enumerate(tables):
            fname = f"table_p{page_num1}_{flavor}_{i}.csv"
            fp = os.path.join(tables_dir, fname)
            try:
                t.to_csv(fp, index=False)
                meta.append({"page": page_num1, "flavor": flavor,
                             "rows": t.shape[0], "cols": t.shape[1],
                             "csv_file": f"tables/{fname}"})
            except Exception:
                pass

    # Lattice first
    try:
        tl = camelot.read_pdf(pdf_path, pages=page_str, flavor="lattice")
        if len(tl) > 0:
            _save(tl, "lattice")
            return meta
    except Exception:
        pass

    # Stream fallback
    try:
        ts = camelot.read_pdf(pdf_path, pages=page_str, flavor="stream")
        if len(ts) > 0:
            _save(ts, "stream")
    except Exception:
        pass

    return meta


# ---------------------- main per-PDF processing ----------------------
def process_pdf(pdf_path: str, company: str, out_dir: str,
                dpi_ocr: int = 200, min_text_chars: int = 200) -> Dict[str, str]:
    """
    Process a single PDF and write per-PDF CSVs. Returns paths written.
    """
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    per_pdf_dir = os.path.join(out_dir, "per_pdf", pdf_base)
    os.makedirs(per_pdf_dir, exist_ok=True)

    pages_rows, blocks_rows, figs_rows, table_rows = [], [], [], []

    with fitz.open(pdf_path) as doc:
        pdf_name = os.path.basename(pdf_path)
        m = YEAR_RE.search(pdf_name)
        year = int(m.group(1)) if m else None

        for pno in range(len(doc)):
            page = doc[pno]
            pnum1 = pno + 1

            # --- blocks + text_all
            blocks = extract_blocks(page)
            for b in blocks:
                blocks_rows.append({
                    "company": company, "year": year, "pdf_file": pdf_name, "page": pno, **b
                })
            text_all = " ".join([b["text"] for b in blocks]).strip()

            # --- figures (no clip OCR; caption only)
            imgs = extract_images(page)
            for im in imgs:
                rect = fitz.Rect(im["bbox_x0"], im["bbox_y0"], im["bbox_x1"], im["bbox_y1"])
                cap = nearest_caption(blocks, rect)
                figs_rows.append({
                    "company": company, "year": year, "pdf_file": pdf_name, "page": pno,
                    **im, "caption": cap, "ocr_text": ""  # no per-figure OCR
                })

            # --- tables (lattice first, stream fallback)
            table_meta = extract_tables_with_camelot(pdf_path, pnum1, per_pdf_dir)
            for tm in table_meta:
                table_rows.append({"company": company, "year": year, "pdf_file": pdf_name, **tm})

            # --- page-level fallback OCR only if text_all short
            page_ocr_text = ""
            if len(text_all) < min_text_chars:
                page_ocr_text = page_ocr_fallback(page, dpi=dpi_ocr)

            pages_rows.append({
                "company": company, "year": year, "pdf_file": pdf_name, "page": pno,
                "char_count": len(text_all), "word_count": len(text_all.split()),
                "text_all": text_all,
                "page_ocr_fallback": page_ocr_text
            })

    # write per-PDF outputs
    pp = os.path.join(per_pdf_dir, "pages.csv")
    pb = os.path.join(per_pdf_dir, "blocks.csv")
    pf = os.path.join(per_pdf_dir, "figures.csv")
    pt = os.path.join(per_pdf_dir, "table_index.csv")
    pd.DataFrame(pages_rows).to_csv(pp, index=False)
    pd.DataFrame(blocks_rows).to_csv(pb, index=False)
    pd.DataFrame(figs_rows).to_csv(pf, index=False)
    pd.DataFrame(table_rows).to_csv(pt, index=False)
    return {"pages": pp, "blocks": pb, "figures": pf, "table_index": pt}


def concat_per_pdf(out_dir: str):
    """Concatenate all per-PDF CSVs into single files in out_dir."""
    per_dir = os.path.join(out_dir, "per_pdf")
    categories = {
        "pages": "pages.csv",
        "blocks": "blocks.csv",
        "figures": "figures.csv",
        "table_index": "table_index.csv",
    }
    for name, fname in categories.items():
        dfs = []
        for root, _, files in os.walk(per_dir):
            if fname in files:
                dfs.append(pd.read_csv(os.path.join(root, fname)))
        if dfs:
            outp = os.path.join(out_dir, f"{name}.csv")
            pd.concat(dfs, ignore_index=True).to_csv(outp, index=False)
            print(f"[CONCAT] wrote {outp} ({len(dfs)} parts)")


def shard(items, shards, idx):
    if shards <= 1:
        return items
    n = len(items)
    per = math.ceil(n / shards)
    start = idx * per
    end = min(n, start + per)
    return items[start:end]


# ---------- TOP-LEVEL WORKER (must be picklable for macOS spawn) ----------
def run_one(fname: str, pdf_dir: str, company: str, out_dir: str,
            dpi: int, min_text_chars: int):
    return process_pdf(
        pdf_path=os.path.join(pdf_dir, fname),
        company=company,
        out_dir=out_dir,
        dpi_ocr=dpi,
        min_text_chars=min_text_chars
    )


# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True, help="Directory with PDFs")
    ap.add_argument("--company", default="Apple")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--dpi", type=int, default=200, help="DPI for fallback OCR (200 is faster)")
    ap.add_argument("--min-text-chars", type=int, default=200,
                    help="Only OCR page if text_all shorter than this")
    ap.add_argument("--workers", type=int, default=6, help="Parallel workers (per-PDF)")
    ap.add_argument("--shards", type=int, default=1, help="Split the PDF list into N shards")
    ap.add_argument("--shard-index", type=int, default=0, help="Shard index [0..N-1]")
    ap.add_argument("--resume", action="store_true",
                    help="Skip PDFs already processed into per_pdf/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pdfs = sorted([f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf")])
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.pdf_dir}")

    # shard
    pdfs = shard(pdfs, args.shards, args.shard_index)

    # resume: skip PDFs that already have per_pdf/<pdf_base>/pages.csv
    if args.resume:
        keep = []
        for f in pdfs:
            pdf_base = os.path.splitext(f)[0]
            per_pdf_dir = os.path.join(args.out_dir, "per_pdf", pdf_base)
            if not os.path.exists(os.path.join(per_pdf_dir, "pages.csv")):
                keep.append(f)
        pdfs = keep

    if not pdfs:
        print("[INFO] Nothing to do after resume/shard filtering.")
        concat_per_pdf(args.out_dir)
        return

    # parallel per-PDF
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                run_one, f, args.pdf_dir, args.company, args.out_dir, args.dpi, args.min_text_chars
            ): f for f in pdfs
        }
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                fut.result()
                print(f"[OK] {f}")
            except Exception as e:
                print(f"[WARN] {f} failed: {e}")

    # concat
    concat_per_pdf(args.out_dir)


if __name__ == "__main__":
    main()
