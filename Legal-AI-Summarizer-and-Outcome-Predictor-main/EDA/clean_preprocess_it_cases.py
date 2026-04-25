import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV  = "scraped_it_cases/it_act_dataset.csv"
OUT_DIR  = Path("scraped_it_cases")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
CLEAN_ALL_CSV      = OUT_DIR / "it_act_clean_all.csv"       
INCONCLUSIVE_CSV   = OUT_DIR / "it_act_inconclusive.csv"    
TRAINABLE_CSV      = OUT_DIR / "it_act_trainable.csv"       
SPLIT_TRAIN_CSV    = OUT_DIR / "train.csv"
SPLIT_VALID_CSV    = OUT_DIR / "valid.csv"
SPLIT_TEST_CSV     = OUT_DIR / "test.csv"

# IT sections 
IT_SECTIONS_SET = {
    "43","65","66","66A","66B","66C","66D","66E","67","67A","67B",
    "68","69","69A","70","72","72A","73","74","79"
}

# Columns required
REQUIRED_COLS = [
    "title", "url", "court_meta", "date", "outcome_label", "outcome_phrase",
    "it_sections", "text_length", "operative_snippet"
]

def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    # De-duplicate
    before = len(df)
    df = df.drop_duplicates(subset=["url"]).copy()
    df = df.drop_duplicates(subset=["title"]).copy()
    print(f"Loaded {before} rows → after dedupe: {len(df)}")
    return df

def normalize_whitespace(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_it_sections(s: str) -> str:
    """
    Normalize section codes to canonical tokens (e.g., '66A', '67B').
    Keep only sections in IT_SECTIONS_SET. Return 'Unknown' if none.
    """
    if not isinstance(s, str) or not s.strip():
        return "Unknown"
    parts = re.split(r"[,\;/\|\s]+", s)
    keep = set()
    for p in parts:
        token = p.strip().upper()
        # Remove trailing punctuation like '.' or ','
        token = re.sub(r"[^\w]", "", token)
        if token in IT_SECTIONS_SET:
            keep.add(token)
    return ", ".join(sorted(keep)) if keep else "Unknown"

def derive_who_won(outcome_label: str) -> str:
    """
    Crisp 'who won' target for classifier demo.
    PETITIONER: petition allowed / conviction set aside / partly allowed
    RESPONDENT: petition dismissed / conviction upheld
    NEUTRAL: disposed, remanded, bail decisions etc.
    """
    if outcome_label in {"PETITION_ALLOWED", "PARTLY_ALLOWED", "CONVICTION_SET_ASIDE"}:
        return "PETITIONER"
    if outcome_label in {"PETITION_DISMISSED", "CONVICTION_UPHELD"}:
        return "RESPONDENT"
    return "NEUTRAL"

def add_section_flags(df: pd.DataFrame) -> pd.DataFrame:
    for sec in sorted(IT_SECTIONS_SET):
        col = f"sec_{sec}"
        df[col] = df["it_sections"].apply(lambda s: int(isinstance(s, str) and sec in s.split(", ")))
    return df

def parse_year_from_date(s: str) -> float:
    # Best-effort year extraction to enable temporal analysis/splits if needed
    if not isinstance(s, str): return float("nan")
    m = re.search(r"(19|20)\d{2}", s)
    return float(m.group(0)) if m else float("nan")

def main():
    df = load_raw(RAW_CSV)

    # --- Basic text cleaning ---
    for c in ["title", "court_meta", "outcome_phrase", "it_sections", "operative_snippet"]:
        df[c] = df[c].apply(normalize_whitespace)

    # --- Drop very small texts 
    before = len(df)
    df = df[(df["text_length"].fillna(0) >= 600) & df["operative_snippet"].notna()]
    print(f"Drop short/empty: {before} → {len(df)}")

    # --- Create combined text (title + operative snippet)
    df["combined_text"] = (
        df["title"].fillna("") + " " + df["operative_snippet"].fillna("")
    ).str.strip()


    # --- Normalize sections & add one-hot flags ---
    df["it_sections"] = df["it_sections"].apply(clean_it_sections)
    df = add_section_flags(df)


    df["year"]     = df["date"].apply(parse_year_from_date)
    df["who_won"]  = df["outcome_label"].apply(derive_who_won)

    # Save the full cleaned table (including INCONCLUSIVE)
    df.to_csv(CLEAN_ALL_CSV, index=False)
    print(f"Saved cleaned (all rows) → {CLEAN_ALL_CSV} ({len(df)} rows)")

    # --- Split out INCONCLUSIVE as “unlabeled/hold-out pool” (do NOT discard) ---
    inconc = df[df["outcome_label"] == "INCONCLUSIVE"].copy()
    if not inconc.empty:
        inconc.to_csv(INCONCLUSIVE_CSV, index=False)
        print(f"Saved INCONCLUSIVE pool → {INCONCLUSIVE_CSV} ({len(inconc)} rows)")


    trainable = df[df["outcome_label"] != "INCONCLUSIVE"].copy()

    
    trainable.to_csv(TRAINABLE_CSV, index=False)
    print(f"Saved trainable table → {TRAINABLE_CSV} ({len(trainable)} rows)")

 
    # Use who_won for stratification 
    if len(trainable) >= 200:
     
        split_df = trainable[trainable["who_won"].notna()].copy()
        tr, te = train_test_split(split_df, test_size=0.2, random_state=42, stratify=split_df["who_won"])
        tr, va = train_test_split(tr, test_size=0.125, random_state=42, stratify=tr["who_won"])  # 0.8*0.125=0.10
        tr.to_csv(SPLIT_TRAIN_CSV, index=False)
        va.to_csv(SPLIT_VALID_CSV, index=False)
        te.to_csv(SPLIT_TEST_CSV,  index=False)
        print(f"Saved splits → train:{len(tr)}  valid:{len(va)}  test:{len(te)}")
    else:
        print("Skip stratified splits (not enough rows).")

    # --- Quick label inventory ---
    print("\nLabel counts (outcome_label):")
    print(df["outcome_label"].value_counts())
    print("\nwho_won distribution:")
    print(df["who_won"].value_counts())

if __name__ == "__main__":
    main()
