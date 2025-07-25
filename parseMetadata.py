import pandas as pd
import os
import sys
import argparse
import bz2
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


# ─── Utility related work ────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Categorization Tool")
    parser.add_argument("input_file", help="Input .csv/.tsv (optionally .bz2 compressed) file")
    parser.add_argument("--output-dir", default="categorized_output", help="Directory for category outputs")
    parser.add_argument("--combo-dir", default="categorized_combinations", help="Combo filters directory")
    return parser.parse_args()

def read_input_file(input_file):
    if input_file.endswith(".bz2"):
        with bz2.open(input_file, "rt") as f:
            if input_file.endswith(".tsv.bz2"):
                return pd.read_csv(f, sep="\t").fillna("")
            else:
                return pd.read_csv(f).fillna("")
    elif input_file.endswith(".tsv"):
        return pd.read_csv(input_file, sep="\t").fillna("")
    else:
        return pd.read_csv(input_file).fillna("")

def clean_name(name: str) -> str:
    return name.replace("(", "").replace(")", "").replace(" ", "_").replace("-", "_").replace("/", "_")

# ─── Reporting on Metadata and its validation  ────────────────────────────────────────────────
def drop_uninformative_columns(df, threshold=0.90):
    dropped = []
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() > threshold * len(df):
            dropped.append(col)
    return df.drop(columns=dropped), dropped

def metadata_completeness_report(df):
    report = []
    for col in df.columns:
        filled = df[col].astype(bool).sum()
        pct = round(100 * filled / len(df), 2)
        unique = df[col].nunique()
        top_vals = df[col].value_counts().head(5).to_dict()
        report.append({"Column": col, "% Filled": pct, "# Unique": unique, "Top 5 Values": top_vals})
    return pd.DataFrame(report)

def validate_column_presence(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    return missing

def calculate_quality_score(row, key_columns):
    present = sum(bool(row[col]) for col in key_columns if col in row)
    return int(100 * present / len(key_columns))

def calculate_mixs_compliance(row, mixs_fields):
    present = sum(bool(row.get(col, "").strip()) for col in mixs_fields)
    return int(100 * present / len(mixs_fields)) if mixs_fields else 0

def apply_temporal_binning(df):
    if 'collection_date' in df.columns:
        df['collection_year'] = pd.to_datetime(df['collection_date'], errors='coerce').dt.year
        df['collection_decade'] = (df['collection_year'] // 10 * 10).fillna("Unknown")

# ─── Normalization of ontology  ───────────────────────────────────────────────
def normalize_column_ontology(df, columns):
    for col in columns:
        if col in df.columns:
            df[f"{col}_normalized"] = df[col].str.lower().str.strip()

# ─── Columns for multi clustering ──────────────────────────────────────────────────────────
def cluster_combined_text(df):
    text_cols = df.select_dtypes(include='object').columns
    df_combined = df[text_cols].astype(str).apply(lambda row: " ".join(row.values), axis=1)
    tfidf = TfidfVectorizer(max_features=200, stop_words='english')
    X = tfidf.fit_transform(df_combined)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    return model.fit_predict(X)

# ─── the new improved script ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    df = read_input_file(args.input_file)
    df_lower = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.combo_dir, exist_ok=True)

    # Drop uninformative columns
    df, dropped_cols = drop_uninformative_columns(df)
    with open(os.path.join(args.output_dir, "dropped_columns.txt"), "w") as f:
        for col in dropped_cols:
            f.write(f"{col}\n")

    # Validate required metadata fields
    required_cols = ["Host Common Name", "Isolation Source", "Host Group"]
    missing = validate_column_presence(df, required_cols)
    with open(os.path.join(args.output_dir, "missing_required_columns.txt"), "w") as f:
        for col in missing:
            f.write(f"{col}\n")

    # Metadata completeness summary
    report = metadata_completeness_report(df)
    report.to_csv(os.path.join(args.output_dir, "metadata_completeness_report.csv"), index=False)

    # Apply temporal binning
    apply_temporal_binning(df)

    # Ontology normalization placeholder
    normalize_column_ontology(df, required_cols)

    # Quality and MIxS compliance scoring
    mixs_fields = required_cols + ["Habitat Type", "Genome Status", "Genome Type"]
    df['metadata_quality_score'] = df.apply(lambda row: calculate_quality_score(row, required_cols), axis=1)
    df['mixs_compliance_score'] = df.apply(lambda row: calculate_mixs_compliance(row, mixs_fields), axis=1)

    # Clustering across all text
    df['all_columns_cluster'] = cluster_combined_text(df)

    # Save updated dataframe with all new features
    df.to_csv(os.path.join(args.output_dir, "improved_metadata.csv"), index=False)

    print("✅ The new improvements have been completed:", args.output_dir)

if __name__ == "__main__":
    main()

# ─── Category definitions (use the exact CSV column names)  ─────────────────────────────────────────────────────
category_rules = {
    "Host Common Name": {
        "Human": ["human", "homo sapiens", "h. sapiens"],
        "Animal": ["mouse", "rat", "pig", "cow", "animal", "mammal", "bird", "fish"],
        "Insect": ["fly", "mosquito", "bee", "wasp", "insect", "drosophila"],
        "Plant": ["plant", "leaf", "root", "stem", "rhizosphere"],
        "Fungus": ["fungus", "yeast", "mold"],
        "No Host Environmental": ["none", "", "environment"]
    },
    "Isolation Source": {
        "Soil": ["soil", "compost", "mud"],
        "Aquatic": ["aquatic", "freshwater", "marine", "ocean", "sea", "lake", "river", "water"],
        "Extreme": ["hot", "thermo", "spring", "acid", "high temperature", "halophile", "alkaline"],
        "Plant-associated": ["plant", "root", "leaf", "stem", "phyllosphere", "rhizosphere"],
        "Human Body Site": ["gut", "skin", "oral", "nasal", "feces", "vagina"],
        "Airborne": ["air", "aerosol", "dust"],
        "Hospital Clinical": ["hospital", "clinic", "nosocomial", "isolate"]
    },
    "Host Group": {
        "Pathogen": ["pathogen", "pathogenic"],
        "Symbiont": ["symbiont", "endosymbiont", "mutualist", "commensal"],
        "Parasite": ["parasite", "parasitic"],
        "Free-living": ["free-living", "freeliving", "environmental"],
        "Opportunistic": ["opportunistic"],
        "Commensal": ["commensal"]
    },
    "Genome Status": {
        "Complete": ["complete"],
        "Draft": ["draft"],
        "Scaffold Contig": ["scaffold", "contig"]
    },
    "Genome Type": {
        "Chromosome": ["chromosome"],
        "Plasmid": ["plasmid"],
        "Metagenome": ["metagenome"]
    },
    "any_column": {
        "Probiotic": ["probiotic"],
        "Industrial": ["biofuel", "enzyme", "bioreactor"],
        "Bioremediation": ["pollution", "bioremediation", "toxic"],
        "Antibiotic Producer": ["antibiotic", "secondary metabolite"]
    },
    "Temperature Preference": {
        "Thermophile": ["thermo", "hot", "heat", "high temperature"],
        "Mesophile": ["mesophile", "moderate temperature"],
        "Psychrophile": ["cold", "psychro", "low temperature"]
    },
    "Oxygen Requirement": {
        "Aerobic": ["aerobic", "obligate aerobe"],
        "Anaerobic": ["anaerobic", "obligate anaerobe"],
        "Facultative Anaerobe": ["facultative anaerobe"],
        "Microaerophile": ["microaerophile"],
        "Aerotolerant": ["aerotolerant"]
    },
    "Gram Stain": {
        "Gram-positive": ["gram positive", "g+"],
        "Gram-negative": ["gram negative", "g-"]
    },
    "Spore Formation": {
        "Spore-forming": ["spore", "sporulate", "spore-forming"],
        "Non-spore-forming": ["non-spore"]
    },
    "Motility": {
        "Motile": ["motile", "flagella", "flagellum"],
        "Non-motile": ["non-motile"]
    },
    "Habitat Type": {
        "Terrestrial": ["soil", "land", "mud", "compost"],
        "Aquatic": ["freshwater", "marine", "sea", "lake", "river", "aquatic"],
        "Host-associated": ["gut", "skin", "feces", "oral", "vaginal", "plant", "leaf", "root"],
        "Extreme": ["hot spring", "halophile", "acid", "alkaline", "cold", "volcano"],
        "Airborne": ["air", "aerosol", "dust"]
    },
    "Continent": {
        "Asia": ["china", "japan", "india", "iran", "iraq"],
        "Africa": ["nigeria", "egypt", "kenya", "south africa"],
        "Europe": ["france", "germany", "uk", "italy", "spain"],
        "North America": ["canada", "usa", "mexico"],
        "South America": ["brazil", "argentina", "chile"],
        "Oceania": ["australia", "new zealand"]
    },
    "Application": {
        "Probiotic": ["probiotic"],
        "Bioremediation": ["bioremediation", "oil", "toxic", "pollution"],
        "Industrial Enzyme": ["enzyme", "biofuel", "bioreactor", "cellulase", "amylase"],
        "Antibiotic Producer": ["antibiotic", "metabolite", "secondary metabolite"],
        "Agricultural Use": ["fertilizer", "rhizobium", "plant growth"]
    },
    "Virulence Traits": {
        "Toxin-producing": ["toxin", "enterotoxin", "cytotoxin"],
        "Biofilm-forming": ["biofilm", "adhesion", "fimbriae"],
        "Antibiotic Resistant": ["resistance", "beta-lactamase", "multidrug", "bla", "mecA"]
    }
}

# ─── New utilizaitons  ────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Genome categorization tool")
    parser.add_argument("input_file", help="Input .csv/.tsv (optionally .bz2 compressed) file")
    parser.add_argument("--output-dir", default="categorized_output", help="Directory for category outputs")
    parser.add_argument("--combo-dir", default="categorized_combinations", help="Directory for combo filters")
    return parser.parse_args()

def read_input_file(input_file):
    if input_file.endswith(".bz2"):
        with bz2.open(input_file, "rt") as f:
            if input_file.endswith(".tsv.bz2"):
                return pd.read_csv(f, sep="\t").fillna("")
            else:
                return pd.read_csv(f).fillna("")
    elif input_file.endswith(".tsv"):
        return pd.read_csv(input_file, sep="\t").fillna("")
    else:
        return pd.read_csv(input_file).fillna("")

def clean_name(name: str) -> str:
    return name.replace("(", "").replace(")", "").replace(" ", "_").replace("-", "_").replace("/", "_")

# ─── the main part ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    combo_dir = args.combo_dir

    df = read_input_file(input_file)
    df_lower = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    os.makedirs(output_dir, exist_ok=True)

    # ─── The categorization (no need to change the logic behind this)  ───────────────────────────────────────────────────────
    for column, subcats in category_rules.items():
        column_folder = clean_name(column)
        if column == "any_column":
            for cat, keywords in subcats.items():
                mask = df_lower.apply(lambda row: any(k in str(cell) for cell in row for k in keywords), axis=1)
                filtered = df[mask]
                if not filtered.empty:
                    out_dir = os.path.join(output_dir, column_folder)
                    os.makedirs(out_dir, exist_ok=True)
                    filtered.to_csv(os.path.join(out_dir, f"{clean_name(cat)}.csv"), index=False)
        elif column == "Genome Type":
            cols_to_check = ["Chromosome", "Plasmids", "Contigs"]
            for cat, keywords in subcats.items():
                mask = df_lower[cols_to_check].apply(lambda row: any(k in str(row[col]) for col in cols_to_check for k in keywords), axis=1)
                filtered = df[mask]
                if not filtered.empty:
                    out_dir = os.path.join(output_dir, column_folder)
                    os.makedirs(out_dir, exist_ok=True)
                    filtered.to_csv(os.path.join(out_dir, f"{clean_name(cat)}.csv"), index=False)
        else:
            if column not in df.columns:
                continue
            for cat, keywords in subcats.items():
                mask = df_lower[column].apply(lambda x: any(k in x for k in keywords))
                filtered = df[mask]
                if not filtered.empty:
                    out_dir = os.path.join(output_dir, column_folder)
                    os.makedirs(out_dir, exist_ok=True)
                    filtered.to_csv(os.path.join(out_dir, f"{clean_name(cat)}.csv"), index=False)

    print("The Categorization is complete :D.", output_dir)

    # ─── The filters for combinations ───────────────────────────────────────────────────
    os.makedirs(combo_dir, exist_ok=True)
    combinations = [
        {"Host Common Name": "Human", "Isolation Source": "Hospital Clinical", "Host Group": "Pathogen"},
        {"Habitat Type": "Soil", "Application": "Bioremediation", "Genome Status": "Complete"}
    ]
    for combo in combinations:
        masks = []
        for column, subcat in combo.items():
            subcat_file = os.path.join(output_dir, clean_name(column), f"{clean_name(subcat)}.csv")
            if os.path.exists(subcat_file):
                df_sub = pd.read_csv(subcat_file)
                masks.append(set(df_sub.index))
        if masks:
            common_indices = set.intersection(*masks)
            if common_indices:
                df_combo = df.loc[list(common_indices)]
                name = "__".join(f"{clean_name(k)}_{clean_name(v)}" for k, v in combo.items())
                df_combo.to_csv(os.path.join(combo_dir, f"{name}.csv"), index=False)
    print("The Combination filtering is finally completed :D See:", combo_dir)

    # ─── Most frequent words analyzed ───────────────────────────────────────────────
    print("\n🔍 Analyzing top words by column:")
    min_word_len = 4
    stopwords = set(["from", "with", "and", "the", "this", "that", "which", "for", "site", "data", "sample"])
    for col in df.columns:
        if df[col].dtype == object:
            all_words = " ".join(df[col].astype(str)).lower().split()
            filtered = [w.strip(".,:;()[]") for w in all_words if len(w) >= min_word_len and w not in stopwords]
            common_words = Counter(filtered).most_common(10)
            if common_words:
                print(f"\nTop words in '{col}':")
                for word, count in common_words:
                    print(f"  {word}: {count}")

    # ─── The inference of the host groups ─────────────────────────────────────────────────
    print("\n🔧 Inferring missing 'Host Group' values...")
    def infer_host_group(row):
        source = row.get("Isolation Source", "").lower()
        host = row.get("Host Common Name", "").lower()
        if "sick" in source or "hospital" in source:
            return "Parasitic"
        elif "healthy" in source or ("human" in host and "sick" not in source):
            return "Commensal"
        elif "water" in source or "environment" in source:
            return "Environmental"
        return "Unknown"

    if "Host Group" in df.columns:
        missing_mask = df["Host Group"].str.strip() == ""
        df.loc[missing_mask, "Host Group"] = df[missing_mask].apply(infer_host_group, axis=1)
    df.to_csv(os.path.join(output_dir, "host_group_inferred.csv"), index=False)
    print("The inference host group is saved :D")

    # ─── The clustering of the text columns ─────────────────────────────────────
    print("\n🚀 Running TF-IDF + KMeans now demonstrates the latent categories have been discovered")
    text_cols = [col for col in df.columns if df[col].dtype == object and df[col].nunique() > 10]
    for col in text_cols:
        print(f"🔬 This column is being clustered: {col}")
        text_data = df[col].astype(str)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        X = vectorizer.fit_transform(text_data)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df[f"{col}_cluster"] = kmeans.fit_predict(X)
        cluster_out = os.path.join(output_dir, f"clustered_{clean_name(col)}.csv")
        df[[col, f"{col}_cluster"]].to_csv(cluster_out, index=False)
        print(f"  ➤ Saved: {cluster_out}")

    print("\n The tasks are completed!! :).")

if __name__ == "__main__":
    main()

