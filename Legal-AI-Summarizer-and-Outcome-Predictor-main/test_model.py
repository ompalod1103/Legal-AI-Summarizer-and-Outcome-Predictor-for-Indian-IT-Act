# test_model.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ================================
# 1) LOAD MODEL + THRESHOLD
# ================================
MODEL_PATH = "models/binary_outcome_model.joblib"
TEST_PATH  = "scraped_it_cases/test.csv"   # adjust if needed

print(f"Loading model bundle: {MODEL_PATH}")
bundle = joblib.load(MODEL_PATH)

# Keys as saved by train_models2.py
pipe      = bundle["pipeline"]               # ColumnTransformer + Calibrated LR
sec_cols  = bundle["sec_cols"]               # list[str]
threshold = bundle["decision_threshold"]     # float

print(f"Loaded. Threshold = {threshold:.3f}")
print(f"Section flags ({len(sec_cols)}): {sec_cols[:8]}{' ...' if len(sec_cols) > 8 else ''}")

# ================================
# 2) LOAD TEST DATA
# ================================
print(f"\nLoading test data: {TEST_PATH}")
df = pd.read_csv(TEST_PATH)

# Ground-truth labels 
y_str = None
if "who_won" in df.columns:
    y_str = df["who_won"].astype(str)



def best_text_column(df_):
    """
    Choose the best available text source to feed into the model.
    The training script built 'text' from title + operative_snippet
    after cleaning. For evaluation, we’ll fall back to:
      1) existing 'text'
      2) 'combined_text' (created by cleaner)
      3) title + operative_snippet
      4) full_text / body (if present)
    """
    if "text" in df_.columns and df_["text"].notna().any():
        return df_["text"].fillna("")
    if "combined_text" in df_.columns and df_["combined_text"].notna().any():
        return df_["combined_text"].fillna("")
    if {"title", "operative_snippet"}.issubset(df_.columns):
        return (df_["title"].fillna("") + " " + df_["operative_snippet"].fillna("")).str.strip()
    for candidate in ["full_text", "body", "judgment_text", "content"]:
        if candidate in df_.columns and df_[candidate].notna().any():
            return df_[candidate].fillna("")
    # join all string-like columns
    str_cols = [c for c in df_.columns if df_[c].dtype == "object"]
    if str_cols:
        return df_[str_cols].fillna("").agg(" ".join, axis=1)
    return pd.Series([""] * len(df_), index=df_.index)

# Create the feature frame the pipeline expects
X = pd.DataFrame(index=df.index)
X["text"] = best_text_column(df)

# Ensure all section flags exist
for c in sec_cols:
    if c not in df.columns:
        X[c] = 0
    else:
        X[c] = df[c].fillna(0).astype(int)

# ================================
# 4) PREDICT
# ================================
print("\nRunning predictions...")
proba_pet = pipe.predict_proba(X)[:, 1]
pred_lab  = np.where(proba_pet >= threshold, "PETITIONER", "RESPONDENT")

# ================================
# 5) METRICS
# ================================
print("\n================ TEST RESULTS ================\n")
if y_str is not None:
    # Binary versions for metrics that need 0/1
    y_true_bin = (y_str == "PETITIONER").astype(int)
    y_pred_bin = (pred_lab == "PETITIONER").astype(int)

    acc      = accuracy_score(y_str, pred_lab)
    prec     = precision_score(y_str, pred_lab, pos_label="PETITIONER")
    rec      = recall_score(y_str, pred_lab, pos_label="PETITIONER")
    f1       = f1_score(y_str, pred_lab, pos_label="PETITIONER")
    macro_f1 = f1_score(y_str, pred_lab, average="macro")
    auc      = roc_auc_score(y_true_bin, proba_pet)

    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision(+1):   {prec:.4f}")
    print(f"Recall(+1):      {rec:.4f}")
    print(f"F1(+1):          {f1:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"ROC-AUC:         {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_str, pred_lab))

    cm = confusion_matrix(y_str, pred_lab, labels=["PETITIONER", "RESPONDENT"])
    cm_df = pd.DataFrame(
        cm,
        index=["True PETITIONER", "True RESPONDENT"],
        columns=["Pred PETITIONER", "Pred RESPONDENT"]
    )
    print("\nConfusion Matrix:")
    print(cm_df)
else:
    print("No ground-truth column 'who_won' found in test.csv — skipping metrics.")

# ================================
# 6) CALIBRATION SUMMARY
# ================================
print("\nCalibration summary:")
print(f"Mean predicted prob (PETITIONER): {np.mean(proba_pet):.4f}")
print(f"Std  predicted prob (PETITIONER): {np.std(proba_pet):.4f}")

# ================================
# 7) SAMPLE OUTPUT PREVIEW
# ================================
preview_cols = []
if "case_id" in df.columns: preview_cols.append("case_id")
if "title"   in df.columns: preview_cols.append("title")
if "date"    in df.columns: preview_cols.append("date")

out = pd.DataFrame({
    **({c: df[c] for c in preview_cols}),
    "pred_label": pred_lab,
    "proba_petitioner": proba_pet
})
print("\nSample predictions (first 10):")
print(out.head(10).to_string(index=False))

print("\n Test evaluation complete.")
