#!/usr/bin/env python3
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to combined annotation CSV file",
    )
    parser.add_argument(
        "--col1",
        type=str,
        default="label_annotator1",
        help="Column name for annotator 1 labels",
    )
    parser.add_argument(
        "--col2",
        type=str,
        default="label_annotator2",
        help="Column name for annotator 2 labels",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load data
    df = pd.read_csv(args.csv)

    # 2. Keep only rows where both annotators provided a label
    df = df.dropna(subset=[args.col1, args.col2]).copy()

    # 3. Convert to int (if stored as floats/strings)
    df[args.col1] = df[args.col1].astype(int)
    df[args.col2] = df[args.col2].astype(int)

    a1 = df[args.col1]
    a2 = df[args.col2]

    print(f"Loaded {len(df)} rows with labels from BOTH annotators.")
    print("\nLabel distribution (annotator 1):")
    print(a1.value_counts().sort_index())
    print("\nLabel distribution (annotator 2):")
    print(a2.value_counts().sort_index())

    # 4. Compute Cohen's kappa
    kappa = cohen_kappa_score(a1, a2)
    print("\n=== Cohen's Kappa ===")
    print(f"kappa = {kappa:.4f}")

    # 5. Confusion matrix for more insight
    labels_sorted = sorted(set(a1.unique()) | set(a2.unique()))
    cm = confusion_matrix(a1, a2, labels=labels_sorted)
    print("\nConfusion matrix (rows = annotator 1, cols = annotator 2):")
    print("labels:", labels_sorted)
    print(cm)


if __name__ == "__main__":
    main()
