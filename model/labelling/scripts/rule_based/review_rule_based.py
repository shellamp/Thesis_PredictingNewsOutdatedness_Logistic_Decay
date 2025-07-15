import json
import pandas as pd
from pathlib import Path

# === CONFIG ===
INPUT_PATH = Path("/Users/sheillaschool/Documents/final/Thesis_PredictingNewsOutdatedness_Logistic_Decay/model/finetune/seed_dataset/rulebased.json")
OUTPUT_PATH = Path("/Users/sheillaschool/Documents/final/Thesis_PredictingNewsOutdatedness_Logistic_Decay/model/finetune/seed_dataset/rule_based_review.json")
RANDOM_SEED = 42

# === LOAD DATA ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data, orient="index")

# === FILTER ONLY LABELED ===
df_labeled = df[df["label"].notna()].copy()

# === SAMPLE 100 ===
df_sampled = df_labeled.sample(n=100, random_state=RANDOM_SEED).copy()

# === ADD COLUMN FOR MANUAL REVIEW ===
df_sampled["reviewed_label"] = None

# === SAVE ===
output_dict = df_sampled.to_dict(orient="index")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output_dict, f, indent=2, ensure_ascii=False)

print(f"✅ Saved 100 reviewed samples to: {OUTPUT_PATH}")
