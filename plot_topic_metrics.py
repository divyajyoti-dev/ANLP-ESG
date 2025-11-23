#!/usr/bin/env python3
# plot_topic_metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("reports/model_metrics_summary_final.csv")

# Replace None with NaN for plotting
df = df.where(pd.notnull(df), None)

companies = df["company"].tolist()
x = range(len(companies))

# 1) Coherence bar chart
plt.figure(figsize=(10,6))
lda_vals  = [v if v is not None else 0 for v in df["lda_coherence"]]
bert_vals = [v if v is not None else 0 for v in df["bert_coherence"]]
plt.bar([i-0.2 for i in x], lda_vals,  width=0.4, label="LDA",  alpha=0.6)
plt.bar([i+0.2 for i in x], bert_vals, width=0.4, label="BERTopic", alpha=0.8)
plt.xticks(x, companies, rotation=45, ha="right")
plt.ylabel("Topic Coherence (C_v)")
plt.title("Coherence Comparison: LDA vs BERTopic")
plt.legend()
plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/coherence_comparison.png", dpi=200)
plt.close()

# 2) Diversity bar chart
plt.figure(figsize=(10,6))
lda_vals  = [v if v is not None else 0 for v in df["lda_diversity"]]
bert_vals = [v if v is not None else 0 for v in df["bert_diversity"]]
plt.bar([i-0.2 for i in x], lda_vals,  width=0.4, label="LDA",  alpha=0.6)
plt.bar([i+0.2 for i in x], bert_vals, width=0.4, label="BERTopic", alpha=0.8)
plt.xticks(x, companies, rotation=45, ha="right")
plt.ylabel("Topic Diversity")
plt.title("Topic Diversity Comparison: LDA vs BERTopic")
plt.legend()
plt.tight_layout()
plt.savefig("reports/diversity_comparison.png", dpi=200)
plt.close()

print("Saved → reports/coherence_comparison.png and reports/diversity_comparison.png")
