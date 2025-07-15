import pandas as pd
import json
import spacy
from pathlib import Path

# === CONFIG ===
INPUT_PATH = Path("/Users/sheillaschool/Documents/final/Thesis_PredictingNewsOutdatedness_Logistic_Decay/data/main_data/finetuning/unlabeled_finetuning.json")
OUTPUT_PATH = Path("/Users/sheillaschool/Documents/final/Thesis_PredictingNewsOutdatedness_Logistic_Decay/model/finetune/seed_dataset/rulebased.json")

# === LOAD SPACY ===
nlp = spacy.load("en_core_web_sm")

# === FUNCTIONS ===

def load_json_as_dataframe(file_path: Path) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame.from_dict(data, orient="index")

def is_ambiguous(text: str) -> bool:
    doc = nlp(str(text).strip())
    tokens = [token for token in doc if token.is_alpha]
    has_verb = any(token.pos_ == "VERB" for token in doc)
    return len(tokens) < 5 or not has_verb

def detect_phrase(text: str) -> str:
    doc = nlp(str(text).strip())
    if any(token.tag_ == "MD" and token.head.pos_ == "VERB" for token in doc):
        return "future"
    elif any(token.tag_ in ("VBP", "VBZ") for token in doc):
        return "ongoing"
    elif any(token.tag_ in ("VBD", "VBN") for token in doc):
        return "past"
    return "unknown"

def assign_label(row) -> tuple[int | None, str]:
    if pd.notnull(row.get("label")):
        return int(row["label"]), "existing"

    if is_ambiguous(row.get("title", "")):
        return None, "ambiguous_skipped"

    t = row.get("t")
    title = str(row.get("title", "")).lower()
    summary = str(row.get("summary", "")).lower()
    phrase_type = detect_phrase(title)

    if t is None or t < 0:
        return None, "invalid_t"

    if t <= 3:
        return 1, "rule_1"
    if phrase_type == "ongoing" and t <= 10:
        return 1, "rule_2"
    if phrase_type == "future" and t > 90:
        return 0, "rule_3"
    if "today" in summary and t > 0:
        return 0, "rule_4"

    PAST_EVENT_KEYWORDS = [
        "olympics", "covid", "president biden", "president obama",
        "brexit", "tokyo 2020", "world cup",
        "election 2020", "pandemic", "lockdown", "covid-19"
    ]

    if any(kw in title for kw in PAST_EVENT_KEYWORDS) and t > 30:
        return 0, "rule_5"

    if t > 365:
        return 0, "rule_6"

    return None, "no_rule_applied"

def save_dataframe_to_json(df: pd.DataFrame, file_path: Path) -> None:
    df_dict = df.to_dict(orient="index")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(df_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ File saved: {file_path.name}")

def main():
    df = load_json_as_dataframe(INPUT_PATH)

    # === Apply rule-based labeling ===
    df[["label", "label_comment"]] = df.apply(
        lambda row: pd.Series(assign_label(row)), axis=1
    )

    # === Reporting ===
    print("\n🔎 Label comment counts:")
    print(df["label_comment"].value_counts(dropna=False))

    print("\n🧮 Label value counts:")
    print(df["label"].value_counts(dropna=False))

    print(f"\n✅ Total rows: {len(df)}")
    print(f"  Label 0: {(df['label'] == 0).sum()}")
    print(f"  Label 1: {(df['label'] == 1).sum()}")
    print(f"  Unlabeled: {df['label'].isnull().sum()}")

    # === Save ===
    save_dataframe_to_json(df, OUTPUT_PATH)

if __name__ == "__main__":
    main()
