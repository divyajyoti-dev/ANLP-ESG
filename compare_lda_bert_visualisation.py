import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("reports/model_metrics_summary.csv")   # or paste your table
fig, ax = plt.subplots(1,2, figsize=(10,4))

# Coherence
df.plot(x="company", y=["lda_coherence","bert_coherence"],
        kind="bar", ax=ax[0], title="Topic Coherence (LDA vs BERT)")
ax[0].set_ylabel("Coherence Score")

# Diversity
df.plot(x="company", y=["lda_diversity","bert_diversity"],
        kind="bar", ax=ax[1], title="Topic Diversity (LDA vs BERT)")
ax[1].set_ylabel("Topic Diversity")

plt.suptitle("Model Performance Comparison Across Companies", fontsize=14)
plt.tight_layout()
plt.show()
