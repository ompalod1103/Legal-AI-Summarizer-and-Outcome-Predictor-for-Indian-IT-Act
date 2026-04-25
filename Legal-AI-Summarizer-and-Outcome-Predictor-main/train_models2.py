import json
import joblib
import numpy as np
import pandas as pd
import re
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

#  paths
CANDIDATE_DIRS = [Path("scraped_it_cases"), Path("."), Path("/mnt/data")]

def find_data_dir():
    for d in CANDIDATE_DIRS:
        if (d / "train.csv").exists() and (d / "valid.csv").exists():
            return d
    raise FileNotFoundError(
        "Could not find train.csv and valid.csv in: " + ", ".join(map(str, CANDIDATE_DIRS))
    )

DATA_DIR = find_data_dir()
TRAIN = DATA_DIR / "train.csv"
VALID = DATA_DIR / "valid.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BIN_MODEL_PATH   = MODEL_DIR / "binary_outcome_model.joblib"
VECTORIZER_INFO  = MODEL_DIR / "feature_info.txt"
META_PATH        = MODEL_DIR / "binary_outcome_meta.json"

#  switches 
SECTION_PREFIX = "sec_"
THRESH_OBJECTIVE  = "macro_f1"  # choose from: "macro_f1", "youden"

#  legal/role stopwords
# These terms often leak party identity or add spurious signal
ROLE_TERMS = {
    "petitioner","respondent","applicant","appellant","complainant","defendant","accused",
    "state","union","learned","counsel","advocate","versus","vs","v","uoi","opp",
    "prosecution","defence","deponent","party","parties","person","persons"
}

# Extend with general high-frequency function words that clutter explanations
GENERIC_EXTRA = {
    "shall","may","thereof","herein","therein","thereby","whereas","hereby",
    "submitted","observed","stated","filed","appeared","prayed","ordered",
    "disposed","allowed","dismissed","quashed","heard","perused"
}

CUSTOM_STOPWORDS = ROLE_TERMS | GENERIC_EXTRA

ROLE_REGEX = re.compile(r"\b(" + "|".join(sorted(ROLE_TERMS)) + r")\b", flags=re.IGNORECASE)

#  helpers
def ensure_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only PETITIONER vs RESPONDENT rows; add target=1 if PETITIONER."""
    mask = df["who_won"].isin(["PETITIONER", "RESPONDENT"])
    out = df.loc[mask].copy()
    out["target"] = (out["who_won"] == "PETITIONER").astype(int)
    return out

def build_text_col(df: pd.DataFrame) -> pd.Series:
    # Prefer a unified free-text column if it exists; otherwise combine title+operative snippet
    if "text" in df.columns:
        t = df["text"].fillna("").astype(str)
    else:
        t = (
            df.get("title", "").fillna("").astype(str) + " " +
            df.get("operative_snippet", "").fillna("").astype(str)
        )
    t = t.str.replace(r"\s+", " ", regex=True).str.strip()
    # Remove role words that leak labels
    t = t.apply(lambda s: ROLE_REGEX.sub(" ", s))
    return t

def get_section_cols(df_columns):
    return sorted([c for c in df_columns if c.startswith(SECTION_PREFIX)])

def make_feature_frame(df: pd.DataFrame, sec_cols: list) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["text"] = df["text"].astype(str)
    for c in sec_cols:
        X[c] = df[c].astype(int)
    return X

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def choose_threshold(y_true, proba, objective="macro_f1"):
    """Grid-search threshold on valid set."""
    thresholds = np.unique(np.clip(np.r_[0.05, np.linspace(0.1, 0.9, 81), 0.95], 0, 1))
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        y_hat = (proba >= t).astype(int)
        if objective == "youden":
            tp = np.sum((y_true == 1) & (y_hat == 1))
            fn = np.sum((y_true == 1) & (y_hat == 0))
            tn = np.sum((y_true == 0) & (y_hat == 0))
            fp = np.sum((y_true == 0) & (y_hat == 1))
            tpr = tp / (tp + fn + 1e-9)
            tnr = tn / (tn + fp + 1e-9)
            score = tpr + tnr - 1
        else:
            score = macro_f1(y_true, y_hat)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t), float(best_score)

#  load & align 
train_df = ensure_binary(pd.read_csv(TRAIN))
valid_df = ensure_binary(pd.read_csv(VALID))

if len(train_df) == 0 or len(valid_df) == 0:
    raise RuntimeError("No rows left after filtering to PETITIONER/RESPONDENT.")

# Align section columns
sec_cols = sorted(set(get_section_cols(train_df.columns)) | set(get_section_cols(valid_df.columns)))
for c in sec_cols:
    if c not in train_df: train_df[c] = 0
    if c not in valid_df: valid_df[c] = 0

# Build text & drop empties
train_df["text"] = build_text_col(train_df)
valid_df["text"] = build_text_col(valid_df)
train_df = train_df[train_df["text"].str.len() > 0].copy()
valid_df = valid_df[valid_df["text"].str.len() > 0].copy()

X_train_df = make_feature_frame(train_df, sec_cols)
X_valid_df = make_feature_frame(valid_df, sec_cols)
y_train = train_df["target"].values
y_valid = valid_df["target"].values

#  vectorizers 
# Drop char-ngrams (they tended to add noise); strengthen word n-grams and pruning.
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.97,
    max_features=70000,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # letters-only tokens of len>=2
    stop_words= sorted(CUSTOM_STOPWORDS)
)

ct = ColumnTransformer(
    transformers=[
        ("txt_word", tfidf_word, "text"),
        ("sec_hot", "passthrough", sec_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=True
)

# ========================== model (with simple C tuning) ==========================
def make_pipe(C=1.0):
    base_clf = LogisticRegression(
        solver="liblinear",
        max_iter=4000,
        class_weight="balanced",   # robust to imbalance without undersampling
        C=C
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=cv)
    return Pipeline([("features", ct), ("clf", clf)])

C_CANDIDATES = [0.5, 1.0, 2.0, 4.0]
best = {"C": None, "score": -1.0, "threshold": 0.5, "pipe": None}

for C in C_CANDIDATES:
    pipe = make_pipe(C)
    pipe.fit(X_train_df, y_train)
    valid_proba = pipe.predict_proba(X_valid_df)[:, 1]
    # pick threshold on valid set
    t, _ = choose_threshold(y_valid, valid_proba, objective=THRESH_OBJECTIVE)
    preds = (valid_proba >= t).astype(int)
    score = macro_f1(y_valid, preds)
    if score > best["score"]:
        best.update({"C": C, "score": score, "threshold": t, "pipe": pipe})

pipe = best["pipe"]
best_t = best["threshold"]

#  final evaluation @ best threshold 
valid_proba = pipe.predict_proba(X_valid_df)[:, 1]
valid_pred_opt = (valid_proba >= best_t).astype(int)

print(f"\n======== FINAL EVAL @C={best['C']} / threshold={best_t:.2f} ({THRESH_OBJECTIVE}) ========")
print(classification_report(y_valid, valid_pred_opt, target_names=["RESPONDENT", "PETITIONER"]))
try:
    auc = roc_auc_score(y_valid, valid_proba)
    print(f"ROC-AUC: {auc:.3f}")
except Exception:
    pass
print("Confusion matrix:\n", confusion_matrix(y_valid, valid_pred_opt))
print(f"Macro-F1: {macro_f1(y_valid, valid_pred_opt):.3f}")

# save 
joblib.dump({"pipeline": pipe, "sec_cols": sec_cols, "decision_threshold": best_t}, BIN_MODEL_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "decision_threshold": best_t,
        "threshold_objective": THRESH_OBJECTIVE,
        "role_terms": sorted(list(ROLE_TERMS)),
        "custom_stopwords_size": len(CUSTOM_STOPWORDS),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "C": best["C"],
        "sec_cols": sec_cols
    }, f, indent=2)
print(f"\nSaved binary model → {BIN_MODEL_PATH}")
print(f"Saved meta → {META_PATH}")

#  explainability 
try:
    fitted_ct = pipe.named_steps["features"]
    word_v = fitted_ct.named_transformers_["txt_word"]
    word_names = np.array(word_v.get_feature_names_out())
    text_feature_names = word_names  

    clf_cal = pipe.named_steps["clf"]
    coefs = []
    for cc in getattr(clf_cal, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None)
        if est is not None and hasattr(est, "coef_"):
            coefs.append(est.coef_.ravel())
    if not coefs:
        base = getattr(clf_cal, "base_estimator", None) or getattr(clf_cal, "base_estimator_", None)
        if base is not None and hasattr(base, "coef_"):
            coefs.append(base.coef_.ravel())

    if coefs:
        coef_avg = np.mean(np.stack(coefs, axis=0), axis=0)
        top_k = 20
        top_pos_idx = np.argsort(coef_avg)[:-top_k-1:-1]
        top_neg_idx = np.argsort(coef_avg)[:top_k]

        with open(VECTORIZER_INFO, "w", encoding="utf-8") as f:
            f.write("Top + features (→ PETITIONER):\n")
            for i in top_pos_idx:
                f.write(f"+ {text_feature_names[i]}\t{coef_avg[i]:.4f}\n")
            f.write("\nTop - features (→ RESPONDENT):\n")
            for i in top_neg_idx:
                f.write(f"- {text_feature_names[i]}\t{coef_avg[i]:.4f}\n")
        print(f"Wrote quick feature introspection → {VECTORIZER_INFO}")
    else:
        print("(Skipped quick feature introspection: no LR coefs found.)")
except Exception as e:
    print(f"(Skipped quick feature introspection: {e})")
