#!/usr/bin/env python3
"""
build_gold_dataset.py

Generate GOLD labels for CED annotation by adjudicating
annotator1 + annotator2 labels using the rule:

    Evidence (2) > Claim (1) > Other (3)

Input CSV must contain:
- sentence_id
- company
- year
- page
- sentence_text
- label_annotator1
- label_annotator2

Output:
    ced_gold_sentences.csv
"""

import pandas as pd
import numpy as np
import argparse


def adjudicate_labels(row):
    """
    Apply adjudication rule to a row.

    Rules:
    1. If both annotators missing → return NaN
    2. If only one present → use that
    3. If both present:
            if any = 2 → Evidence
            elif any = 1 → Claim
            else       → Other
    """
    l1 = row["label_annotator1"]
    l2 = row["label_annotator2"]

    # Case 1: both missing
    if pd.isna(l1) and pd.isna(l2):
        return np.nan

    # Case 2: only one labeled
    if pd.isna(l1):
        return int(l2)
    if pd.isna(l2):
        return int(l1)

    # Case 3: both present → apply hierarchy Evidence > Claim > Other
    l1, l2 = int(l1), int(l2)
    labels = {l1, l2}

    if 2 in labels:
        return 2
    elif 1 in labels:
        return 1
    else:
        return 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to combined annotation CSV file"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="ced_gold_sentences.csv",
        help="Output CSV file"
    )
    args = parser.parse_args()

    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)

    # Apply adjudication
    print("Adjudicating labels...")
    df["label_gold"] = df.apply(adjudicate_labels, axis=1)

    # Drop rows where no gold label could be assigned
    gold_df = df.dropna(subset=["label_gold"]).copy()
    gold_df["label_gold"] = gold_df["label_gold"].astype(int)

    # Select relevant columns
    gold_out = gold_df[
        ["sentence_id", "company", "year", "page", "sentence_text", "label_gold"]
    ].sort_values(["company", "year", "page", "sentence_id"])

    # Save
    gold_out.to_csv(args.out, index=False)
    print(f"Gold dataset saved to {args.out}")
    print(f"Total gold sentences: {len(gold_out)}")


if __name__ == "__main__":
    main()
