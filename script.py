import pandas as pd
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# ─── Category definitions (use the exact CSV column names) ──────────────────
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

def clean_name(name: str) -> str:
    return name.replace("(", "").replace(")", "").replace(" ", "_").replace("-", "_").replace("/", "_")

# ─── Loading up the data ──────────────────
df = pd.read_csv("input.csv").fillna("")
df_lower = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# ─── The categorization (no need to change the logic behind this)  ──────────────────
os.makedirs("categorized_output", exist_ok=True)
for column, subcats in category_rules.items():
    column_folder = clean_name(column)
    if column == "any_column":
        for cat, keywords in subcats.items():
            mask = df_lower.apply(lambda row: any(k in str(cell) for cell in row for k in keywords), axis=1)
            filtered = df[mask]
            if not filtered.empty:
                out_dir = os.path.join("categorized_output", column_folder)
                os.makedirs(out_dir, exist_ok=True)
                filtered.to_csv(os.path.join(out_dir, f"{clean_name(cat)}.csv"), index=False)
    elif column == "Genome Type":
        cols_to_check = ["Chromosome", "Plasmids", "Contigs"]
        for cat, keywords in subcats.items():
            mask = df_lower[cols_to_check].apply(lambda row: any(k in str(row[col]) for col in cols_to_check for k in keywords), axis=1)
            filtered = df[mask]
            if not filtered.empty:
                out_dir = os.path.join("categorized_output", column_folder)
                os.makedirs(out_dir, exist_ok=True)
                filtered.to_csv(os.path.join(out_dir, f"{clean_name(cat)}.csv"), index=False)
    else:
        if column not in df.columns:
            continue
        for cat, keywords in subcats.items():
            mask = df_lower[column].apply(lambda x: any(k in x for k in keywords))
            filtered = df[mask]
            if not filtered.empty:
                out_dir = os.path.join("categorized_output", column_folder)
                os.makedirs(out_dir, exist_ok=True)
                filtered.to_csv(os.path.join(out_dir, f"{clean_name(cat)}.csv"), index=False)

print("The Categorization is complete :D. Check under the folders under 'categorized_output/'")

# ─── The filters for combinations ──────────────────
combo_dir = "categorized_combinations"
os.makedirs(combo_dir, exist_ok=True)
combinations = [
    {"Host Common Name": "Human", "Isolation Source": "Hospital Clinical", "Host Group": "Pathogen"},
    {"Habitat Type": "Soil", "Application": "Bioremediation", "Genome Status": "Complete"}
]
for combo in combinations:
    masks = []
    for column, subcat in combo.items():
        subcat_file = os.path.join("categorized_output", clean_name(column), f"{clean_name(subcat)}.csv")
        if os.path.exists(subcat_file):
            df_sub = pd.read_csv(subcat_file)
            masks.append(set(df_sub.index))
    if masks:
        common_indices = set.intersection(*masks)
        if common_indices:
            df_combo = df.loc[list(common_indices)]
            name = "__".join(f"{clean_name(k)}_{clean_name(v)}" for k, v in combo.items())
            df_combo.to_csv(os.path.join(combo_dir, f"{name}.csv"), index=False)
print("The Combination filtering is finally completed :D. Check out 'categorized_combinations/'")

# ─── Most frequent words analyzed  ──────────────────
print("\n🔍 Every column is being analyzed for the most frequent words *sigh*")
min_word_len = 4
stopwords = set(["from", "with", "and", "the", "this", "that", "which", "for", "site", "data", "sample"])
word_summary = {}
for col in df.columns:
    if df[col].dtype == object:
        all_words = " ".join(df[col].astype(str)).lower().split()
        filtered = [w.strip(".,:;()[]") for w in all_words if len(w) >= min_word_len and w not in stopwords]
        common_words = Counter(filtered).most_common(10)
        word_summary[col] = common_words
for col, words in word_summary.items():
    print(f"\nTop words in '{col}':")
    for word, count in words:
        print(f"  {word}: {count}")

# ─── The inference of the host groups ──────────────────
print("\n🔧 Inferring missing 'Host Group'")
def infer_host_group(row):
    source = row.get("Isolation Source", "").lower()
    host = row.get("Host Common Name", "").lower()
    if "sick" in source or "hospital" in source:
        return "Parasitic"
    elif "healthy" in source or ("human" in host and not "sick" in source):
        return "Commensal"
    elif "water" in source or "environment" in source:
        return "Environmental"
    else:
        return "Unknown"
if "Host Group" in df.columns:
    missing_mask = df["Host Group"].str.strip() == ""
    df.loc[missing_mask, "Host Group"] = df[missing_mask].apply(infer_host_group, axis=1)
df.to_csv(os.path.join("categorized_output", "host_group_inferred.csv"), index=False)
print("✅ Inferred Host Group values saved to: categorized_output/host_group_inferred.csv")

# ─── The clustering of the text columns ──────────────────
print("\n🚀 Running TF-IDF + KMeans now demonstrates the latent categories have been discovered")
text_cols = [col for col in df.columns if df[col].dtype == object and df[col].nunique() > 10]
for col in text_cols:
    print(f"\n🔬 This column is being clustered: {col}")
    text_data = df[col].astype(str)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    X = vectorizer.fit_transform(text_data)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df[f"{col}_cluster"] = kmeans.fit_predict(X)
    # Save clustered outputs
    cluster_out = os.path.join("categorized_output", f"clustered_{clean_name(col)}.csv")
    df[[col, f"{col}_cluster"]].to_csv(cluster_out, index=False)
    print(f"  ➤ Saved clustered output to: {cluster_out}")
print("The clustering is now complete ✅.")

print("\n💡 Tip: You can use the output for the clustering to see if there are any structures that are hidden inside of the data and come up with a new class of metadatas")

