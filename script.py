import pandas as pd
import os

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
    # We'll handle these three columns together below
    "Genome Type": {
        "Chromosome": ["chromosome"],
        "Plasmid": ["plasmid"],
        "Metagenome": ["metagenome"]
    },
    # Search any column for these
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
    """Remove parentheses, replace spaces & hyphens with underscores."""
    return name.replace("(", "").replace(")", "").replace(" ", "_").replace("-", "_")

# ─── Loading and the preprocessing  ───────────────────────────────────────────────────────
df = pd.read_csv("input.csv").fillna("")                  # original data
df_lower = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)  

# ─── Preparing the outputs  ──────────────────────────────────────────────────────────
base_dir = "categorized_output"
os.makedirs(base_dir, exist_ok=True)

# ─── The write outs ──────────────────────────────────────────────────
for column, subcats in category_rules.items():
    column_folder = clean_name(column)
    
    # 1) Searching all the columns hopefully this works
    if column == "any_column":
        for cat, keywords in subcats.items():
            mask = df_lower.apply(
                lambda row: any(k in str(cell) for cell in row for k in keywords),
                axis=1
            )
            filtered = df[mask]
            if not filtered.empty:
                out_dir = os.path.join(base_dir, column_folder)
                os.makedirs(out_dir, exist_ok=True)
                filtered.to_csv(os.path.join(out_dir, f"{clean_name(cat)}.csv"), index=False)
    
    # 2) genome type spanning three different columns
    elif column == "Genome Type":
        cols_to_check = ["Chromosome", "Plasmids", "Contigs"]
        for cat, keywords in subcats.items():
            mask = df_lower[cols_to_check].apply(
                lambda row: any(k in str(row[col]) for col in cols_to_check for k in keywords),
                axis=1
            )
            filtered = df[mask]
            if not filtered.empty:
                out_dir = os.path.join(base_dir, column_folder)
                os.makedirs(out_dir, exist_ok=True)
                filtered.to_csv(os.path.join(out_dir, f"{clean_name(cat)}.csv"), index=False)

    # 3) one standard column
    else:
        if column not in df.columns:
            continue  # skip if your CSV is missing that header
        for cat, keywords in subcats.items():
            mask = df_lower[column].apply(lambda x: any(k in x for k in keywords))
            filtered = df[mask]
            if not filtered.empty:
                out_dir = os.path.join(base_dir, column_folder)
                os.makedirs(out_dir, exist_ok=True)
                filtered.to_csv(os.path.join(out_dir, f"{clean_name(cat)}.csv"), index=False)

print("✅ Categorization complete. See folders under 'categorized_output/'.")


# ─── Combo filtering lol ──────────────────

combinations = [
    {
        "Host Common Name": "Human",
        "Isolation Source": "Hospital Clinical",
        "Host Group": "Pathogen"
    },
    {
        "Habitat Type": "Soil",
        "Application": "Bioremediation",
        "Genome Status": "Complete"
    }
]

combo_dir = "categorized_combinations"
os.makedirs(combo_dir, exist_ok=True)

combo_dir = "categorized_combinations"
os.makedirs(combo_dir, exist_ok=True)

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

print("✅ Combination filtering complete. See 'categorized_combinations/'.")
