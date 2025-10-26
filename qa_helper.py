#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA HELPER — ESG Corpus Viewer
-----------------------------
Utility to load the cleaned corpus (corpus_pages.csv)
and extract text for manual verification against PDFs.

Usage examples:
    python qa_helper.py --year 2020 --page 12
    python qa_helper.py --year 2020 --report Apple_Environmental_Progress_Report_2020.pdf
"""

import pandas as pd
import argparse
from pathlib import Path

# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description="QA helper for viewing cleaned ESG text")
parser.add_argument("--company", default="apple", help="Company name (default=apple)")
parser.add_argument("--year", type=int, help="Year of report (e.g. 2020)")
parser.add_argument("--page", type=int, help="Page number to inspect")
parser.add_argument("--report", type=str, help="Optional full report name to filter by")
parser.add_argument("--base", default="data/processed", help="Base path of processed folder")
args = parser.parse_args()

# ------------------ LOAD ------------------
BASE = Path(args.base) / f"{args.company}_layout_all"
CORPUS_PATH = BASE / "corpus_pages.csv"

if not CORPUS_PATH.exists():
    raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")

df = pd.read_csv(CORPUS_PATH)

# ------------------ QUERY ------------------
def get_page_text(year: int, page: int, report: str = None):
    """Return cleaned page text for given year & page."""
    mask = (df["year"] == year) & (df["page"] == page)
    if report:
        mask &= df["report_name"].str.contains(report, case=False, na=False)
    match = df.loc[mask, "page_text_final"]
    if match.empty:
        return f"[No text found for year={year}, page={page}, report={report}]"
    return str(match.iloc[0])

def get_all_pages(year: int, report: str = None):
    """Return all pages for a year (optionally restricted to one report)."""
    mask = (df["year"] == year)
    if report:
        mask &= df["report_name"].str.contains(report, case=False, na=False)
    sub = df.loc[mask, ["year", "page", "report_name", "page_text_final"]].sort_values("page")
    return sub

# ------------------ CLI ACTION ------------------
if args.page:
    text = get_page_text(args.year, args.page, args.report)
    print(f"\n=== YEAR {args.year} | PAGE {args.page} ===\n")
    print(text)
elif args.year:
    subset = get_all_pages(args.year, args.report)
    print(f"\n=== REPORT TEXTS for {args.company.upper()} {args.year} ===\n")
    for _, row in subset.iterrows():
        print(f"\n--- Page {int(row.page)} ---")
        print(row.page_text_final[:2000])  # first 2000 chars for readability
        print("\n")
else:
    print("Please specify --year (and optionally --page or --report)")
