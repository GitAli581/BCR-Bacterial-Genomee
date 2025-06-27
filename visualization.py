import pandas as pd
import matplotlib.pyplot as plt
import os

# ─── Loading of the Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("input.csv", low_memory=False).fillna("")
df.columns = df.columns.str.lower()  # lowercase the column names
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
print("CSV columns:", df.columns.tolist())

# ─── The output directory creation ─────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

# ─── the bar chart for the common name of the host  ─────────────────────────────────────────────
if "host common name" in df.columns:
    host_counts = df["host common name"].value_counts()
    host_counts.plot(kind="bar", title="Genomes by Host", ylabel="Count")
    plt.tight_layout()
    plt.savefig("figures/host_bar_chart.png")
    plt.clf()

# ─── the isolation source's bar chart ─────────────────────────────────────────────
if "isolation source" in df.columns:
    iso_counts = df["isolation source"].value_counts()
    iso_counts.plot(kind="bar", title="Genomes by Isolation Source", ylabel="Count")
    plt.tight_layout()
    plt.savefig("figures/source_bar_chart.png")
    plt.clf()

# ─── pie chart for the genomes status ────────────────────────────────────────────────
if "genome status" in df.columns:
    status_counts = df["genome status"].value_counts()
    status_counts.plot(kind="pie", autopct="%1.1f%%", title="Genome Status")
    plt.ylabel("")  # hide y-axis label for pie
    plt.tight_layout()
    plt.savefig("figures/genome_status_pie.png")
    plt.clf()

# ─── heat map table ─────────────────────────────
if "host common name" in df.columns and "isolation source" in df.columns:
    pivot = pd.crosstab(df["host common name"], df["isolation source"])
    pivot.to_csv("figures/host_vs_source_table.csv")  # Save raw data
    pivot.plot(kind="bar", stacked=True, figsize=(10, 6), title="Host vs Isolation Source")
    plt.tight_layout()
    plt.savefig("figures/heatmap_host_source.png")
    plt.clf()

columns_to_plot = ["host common name", "isolation source", "genome status"]

for column in columns_to_plot:
    if column in df.columns:
        counts = df[column].value_counts().head(20)
        counts.plot(kind="bar", title=column)
        plt.tight_layout()
        plt.savefig(f"figures/{column.replace(' ', '_')}_barplot.png")
        plt.clf()



print("✅ Visualizations saved to /figures")