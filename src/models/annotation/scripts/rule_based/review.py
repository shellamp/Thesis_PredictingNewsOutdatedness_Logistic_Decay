import json
import pandas as pd
from pathlib import Path

# === CONFIG ===
INPUT_PATH = Path("/Users/sheillaschool/Documents/final/Thesis_PredictingNewsOutdatedness_Logistic_Decay/model/labelling/data/rulebased.json")
OUTPUT_PATH = Path("/Users/sheillaschool/Documents/final/Thesis_PredictingNewsOutdatedness_Logistic_Decay/model/labelling/data/rulebased_review.json")
RANDOM_SEED = 42

# === LOAD DATA ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data, orient="index")

# === FILTER ONLY LABELED ===
df_labeled = df[df["label"].notna()].copy()

# === SEPARATE AND SAMPLE EACH CLASS ===
df_label_0 = df_labeled[df_labeled["label"] == 0].sample(n=60, random_state=RANDOM_SEED)
df_label_1 = df_labeled[df_labeled["label"] == 1].sample(n=60, random_state=RANDOM_SEED)

# === COMBINE AND SHUFFLE ===
df_sampled = pd.concat([df_label_0, df_label_1]).sample(frac=1, random_state=RANDOM_SEED).copy()

# === ADD EMPTY REVIEW COLUMN ===
df_sampled["reviewed_label"] = None

# === SAVE ===
output_dict = df_sampled.to_dict(orient="index")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output_dict, f, indent=2, ensure_ascii=False)

print(f"✅ Saved 120 reviewed samples (60 per class) to: {OUTPUT_PATH}")
