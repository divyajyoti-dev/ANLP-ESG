#!/usr/bin/env python3
"""
Step 3: Aggregate metrics from Step 1 (LDA/NMF) and Step 2 (BERTopic)
- Inputs are two folders:
  --step1-dir (has metrics_lda.csv, metrics_nmf.csv, topic_prevalence_*.csv)
  --step2-dir (has metrics_bertopic.csv, topic_prevalence_bertopic.csv)
- Output:
  model_metrics_summary.csv
"""
import argparse, os, pandas as pd

def maybe(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1-dir", required=True)  # e.g., reports/topics_all_companies
    ap.add_argument("--step2-dir", required=True)  # e.g., reports/topics_all_bt
    ap.add_argument("--out", default="model_metrics_summary.csv")
    args = ap.parse_args()

    rows = []
    for name in ["lda","nmf"]:
        p = os.path.join(args.step1_dir, f"metrics_{name}.csv")
        df = maybe(p)
        if df is not None:
            df["model"] = name.upper()
            rows.append(df)

    bt = maybe(os.path.join(args.step2_dir, "metrics_bertopic.csv"))
    if bt is not None:
        bt["model"] = "BERTopic"
        rows.append(bt)

    if not rows:
        raise SystemExit("No metrics found. Check --step1-dir and --step2-dir.")

    out = pd.concat(rows, ignore_index=True)
    # Reorder columns nicely if present
    cols = [c for c in ["model","topic_diversity","coherence_cv","silhouette"] if c in out.columns] + \
           [c for c in out.columns if c not in ["model","topic_diversity","coherence_cv","silhouette"]]
    out = out[cols]
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Step 3: Aggregate metrics from Step 1 (LDA/NMF) and Step 2 (BERTopic)
- Inputs are two folders:
  --step1-dir (has metrics_lda.csv, metrics_nmf.csv, topic_prevalence_*.csv)
  --step2-dir (has metrics_bertopic.csv, topic_prevalence_bertopic.csv)
- Output:
  model_metrics_summary.csv
"""
import argparse, os, pandas as pd

def maybe(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1-dir", required=True)  # e.g., reports/topics_all_companies
    ap.add_argument("--step2-dir", required=True)  # e.g., reports/topics_all_bt
    ap.add_argument("--out", default="model_metrics_summary.csv")
    args = ap.parse_args()

    rows = []
    for name in ["lda","nmf"]:
        p = os.path.join(args.step1_dir, f"metrics_{name}.csv")
        df = maybe(p)
        if df is not None:
            df["model"] = name.upper()
            rows.append(df)

    bt = maybe(os.path.join(args.step2_dir, "metrics_bertopic.csv"))
    if bt is not None:
        bt["model"] = "BERTopic"
        rows.append(bt)

    if not rows:
        raise SystemExit("No metrics found. Check --step1-dir and --step2-dir.")

    out = pd.concat(rows, ignore_index=True)
    # Reorder columns nicely if present
    cols = [c for c in ["model","topic_diversity","coherence_cv","silhouette"] if c in out.columns] + \
           [c for c in out.columns if c not in ["model","topic_diversity","coherence_cv","silhouette"]]
    out = out[cols]
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
