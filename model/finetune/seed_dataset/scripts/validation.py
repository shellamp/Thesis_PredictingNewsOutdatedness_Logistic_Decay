import json
import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIG ===
INPUT_PATH = Path("/Users/sheillaschool/Documents/final/Thesis_PredictingNewsOutdatedness_Logistic_Decay/model/finetune/seed_dataset/data/rule_based_review.json")
API_KEY_FILE = Path("wandb-api-key.txt")
CSV_OUTPUT_PATH = Path("./validation_metrics.csv")
CONF_MATRIX_CSV = Path("./confusion_matrix.csv")

# === LOGIN TO WANDB ===
with open(API_KEY_FILE, "r") as f:
    wandb.login(key=f.read().strip())

# === LOAD JSON FILE SAFELY ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data, orient="index")

# === FILTER VALID ROWS ===
df = df[df["label"].notnull() & df["reviewed_label"].notnull()].copy()
y_true = df["reviewed_label"].astype(int)
y_pred = df["label"].astype(int)

# === GET CLASSIFICATION REPORT ===
report = classification_report(y_true, y_pred, output_dict=True)
print("\n📊 Classification Report:")
print(json.dumps(report, indent=2))

# === SAVE CLASSIFICATION METRICS TO CSV ===
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(CSV_OUTPUT_PATH)
print(f"\n✅ Classification report saved to: {CSV_OUTPUT_PATH.name}")

# === SAVE CONFUSION MATRIX TO CSV ===
conf_matrix = confusion_matrix(y_true, y_pred)
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["actual_0", "actual_1"],
    columns=["pred_0", "pred_1"]
)
conf_matrix_df.to_csv(CONF_MATRIX_CSV)
print(f"✅ Confusion matrix saved to: {CONF_MATRIX_CSV.name}")

# === LOG TO WANDB ===
run = wandb.init(project="news-outdatedness-validation", name="rulebased_vs_reviewed_table")

# Log metrics as key-value (for quick view)
wandb.log({
    "precision_0": report["0"]["precision"],
    "recall_0": report["0"]["recall"],
    "f1_0": report["0"]["f1-score"],
    "support_0": report["0"]["support"],

    "precision_1": report["1"]["precision"],
    "recall_1": report["1"]["recall"],
    "f1_1": report["1"]["f1-score"],
    "support_1": report["1"]["support"],

    "macro_avg_f1": report["macro avg"]["f1-score"],
    "accuracy": report["accuracy"]
})

# Log classification report as table
wandb_table = wandb.Table(dataframe=df_report.reset_index().rename(columns={"index": "metric"}))
wandb.log({"classification_report_table": wandb_table})

# Log confusion matrix as table
wandb_conf_table = wandb.Table(dataframe=conf_matrix_df.reset_index().rename(columns={"index": "actual"}))
wandb.log({"confusion_matrix_table": wandb_conf_table})

wandb.finish()
