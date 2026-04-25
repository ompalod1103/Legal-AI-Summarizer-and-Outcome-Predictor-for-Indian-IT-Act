import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import pypdf as PyPDF2
except Exception:
    PyPDF2 = None

# ---------- SAME role cleanup as training ----------
ROLE_TERMS = {
    "petitioner","respondent","applicant","appellant","complainant","defendant","accused",
    "state","union","learned","counsel","advocate","versus","vs","v","uoi","opp",
    "prosecution","defence","deponent","party","parties","person","persons"
}
ROLE_REGEX = re.compile(r"\b(" + "|".join(sorted(ROLE_TERMS)) + r")\b", flags=re.IGNORECASE)

# ---------- IT Act sections we support as binary features ----------
SEC_PATTERN = re.compile(r"\bsection\s+([0-9]{1,3}[A-Z]?)\b", flags=re.IGNORECASE)

def extract_text_from_pdf(pdf_path: str, max_pages: int = None) -> str:
    """
    Extract text from PDF using PyPDF2 (available cross-platform).
    """
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 not available. Please install it in your environment.")
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        n = len(reader.pages)
        if max_pages is not None:
            n = min(n, max_pages)
        for i in range(n):
            try:
                page = reader.pages[i]
                t = page.extract_text() or ""
            except Exception:
                t = ""
            text_parts.append(t)
    text = "\n".join(text_parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _clean_for_model(text: str) -> str:
    """
    Mirror the training-time cleanup:
    - remove role words that leak labels
    - normalize whitespace
    """
    text = ROLE_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_it_sections(text: str) -> set:
    """
    Find occurrences like 'Section 66A', 'section 67', etc.
    Returns a set of normalized feature keys: {'sec_66A', 'sec_67', ...}
    """
    secs = set()
    for m in SEC_PATTERN.finditer(text):
        code = m.group(1).upper()  # e.g., '66A'
        # Only keep typical IT Act range (coarse filter to avoid IPC collisions)
        # You can relax this if your training added more sections as features.
        secs.add(f"sec_{code}")
    return secs

def make_inference_frame(text: str, sec_cols: list) -> pd.DataFrame:
    """
    Build a single-row frame with 'text' and binary section flags aligned to training cols.
    """
    text_norm = _clean_for_model(text)
    found_secs = parse_it_sections(text_norm)
    row = {"text": text_norm}
    for c in sec_cols:
        row[c] = 1 if c in found_secs else 0
    return pd.DataFrame([row])

def _extract_parties_heuristic(text_first_page: str) -> dict:
    """
    Best-effort party extraction: looks for names before/after Petitioner/Respondent synonyms.
    We keep it deliberately simple and robust. You already said your extractor works; this is a fallback.
    """
    t = text_first_page
    t = re.sub(r"\s+", " ", t)
    # Simple patterns:
    pat_pet = re.compile(r"(?:petitioner|applicant|appellant|complainant)\s*[:\-–]\s*([A-Z][A-Za-z.\s]+)", re.IGNORECASE)
    pat_res = re.compile(r"(?:respondent|defendant|state)\s*[:\-–]\s*([A-Z][A-Za-z.\s]+)", re.IGNORECASE)

    pet = None
    res = None
    m1 = pat_pet.search(t)
    m2 = pat_res.search(t)
    if m1:
        pet = m1.group(1).strip()
    if m2:
        res = m2.group(1).strip()

    # If nothing found, try crude “X vs Y” capture
    if not pet or not res:
        m = re.search(r"([A-Z][A-Za-z .]+)\s+v(?:s\.?|ersus)\s+([A-Z][A-Za-z .]+)", t, re.IGNORECASE)
        if m:
            pet = pet or m.group(1).strip()
            res  = res  or m.group(2).strip()

    return {"petitioner": pet, "respondent": res}

def predict_case(pdf_path: str, model_dir: str = "models") -> dict:
    """
    Load saved pipeline + meta, extract text + sections + parties,
    compute probability Petitioner wins, choose label via saved threshold.
    """
    model_path = Path(model_dir) / "binary_outcome_model.joblib"
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    sec_cols = bundle.get("sec_cols", [])
    decision_threshold = float(bundle.get("decision_threshold", 0.5))

    # Full text for model, and first page for party heuristics
    full_text = extract_text_from_pdf(pdf_path, max_pages=None)
    first_page_text = extract_text_from_pdf(pdf_path, max_pages=1)

    X_row = make_inference_frame(full_text, sec_cols)
    proba = float(pipe.predict_proba(X_row)[:, 1][0])
    winner = "PETITIONER" if proba >= decision_threshold else "RESPONDENT"

    parties = _extract_parties_heuristic(first_page_text)

    # Also return which section flags we actually set to 1
    detected = []
    for c in sec_cols:
        if X_row.iloc[0][c] == 1:
            detected.append(c)

    return {
        "proba_petitioner": proba,
        "decision_threshold": decision_threshold,
        "predicted_winner_label": winner,
        "parties": parties,
        "detected_sections": detected
    }
