#Narrative V2
# narrative_explainer.py
import re
from typing import Dict, Any, List, Tuple

# short labels you can tweak/extend
_SECTION_TAG = {
    # CHAPTER I — PRELIMINARY
 
    "1":  "Short title, extent, commencement—no penalty.",
    "2":  "Definitions—no penalty.",

    # -------------------------
    # CHAPTER II — DIGITAL SIGNATURES
    # -------------------------
    "3":  "Electronic signatures legally recognized—no penalty.",
    "3A": "Electronic signature techniques recognized—no penalty.",
    "4":  "Electronic records legally recognized—no penalty.",
    "5":  "Legal recognition of electronic signatures—no penalty.",
    "6":  "Use of electronic records by government—no penalty.",
    "6A": "Delivery of services by electronic means—no penalty.",
    "7":  "Retention of electronic records—no penalty.",
    "7A": "Audit of documents maintained electronically—no penalty.",
    "8":  "Publication of rules/regulations—no penalty.",
    "9":  "Sections not applicable to negotiable instruments, wills, etc.—no penalty.",
    "10": "Power to make rules regarding digital signature—no penalty.",
    "10A":"E-contracts recognition—no penalty.",

    # -----------------------------------------------------
    # CHAPTER III — ELECTRONIC GOVERNANCE (No penalties)
    # -----------------------------------------------------
    "11": "Attribution of e-records—no penalty.",
    "12": "Acknowledgement of receipt—no penalty.",
    "13": "Dispatch & receipt of e-records—no penalty.",
    "14": "Secure electronic records—no penalty.",
    "15": "Secure digital signatures—no penalty.",
    "16": "Security procedure—no penalty.",

    # -----------------------------------------------------
    # CHAPTER IV — CERTIFYING AUTHORITIES (Penal provisions start from 34 onward)
    # -----------------------------------------------------
    "17": "Controller of Certifying Authorities—no penalty.",
    "18": "Functions of Controller—no penalty.",
    "19": "Recognition of foreign Certifying Authorities—no penalty.",
    "20": "Licence to issue DSC—no penalty.",
    "21": "Application for licence—no penalty.",
    "22": "Renewal of licence—no penalty.",
    "23": "Procedure for grant/refusal of licence—no penalty.",
    "24": "Suspension of licence—no penalty.",
    "25": "Revocation of licence—no penalty.",
    "26": "Notice to subscriber—no penalty.",
    "27": "Certifying Authority to follow security guidelines.",
    "28": "Validity period of certificate—no penalty.",
    "29": "Trustworthy systems—no penalty.",
    "30": "Certifying Authority duties—no penalty.",
    "31": "Display of licence—no penalty.",
    "32": "Surrender of licence—no penalty.",
    "33": "Disclosure—no penalty.",

    # PENAL PROVISIONS
    "34": "Penalty for publishing false Digital Signature Certificate.",
    "35": "Performance audits—no penalty.",
    "36": "Penalty: Contravention by Certifying Authority.",
    "37": "Power to investigate by Controller.",
    "38": "Access to computers/files for investigation.",
    "39": "Right to investigate—no penalty.",

    # -----------------------------------------------------
    # CHAPTER V — DIGITAL SIGNATURES
    # -----------------------------------------------------
    "40": "Issue of digital signature certificate—no penalty.",
    "41": "Suspension of DSC—no penalty.",
    "42": "Revocation of DSC—no penalty.",

    # -----------------------------------------------------
    # **REAL PENALTIES START HERE**
    # -----------------------------------------------------

    # CIVIL COMPENSATION
    "43":  "Compensation for unauthorized access, damage, virus spreading, disruption etc.",
    "43A": "Compensation for failure to protect sensitive personal data.",
    "44":  "Penalty for failure to furnish information, returns, documents.",
    "45":  "Penalty for contravention of rules.",
    "46":  "Power of adjudication—no penalty.",
    "47":  "Factors for adjudging compensation—no penalty.",

    # CYBER OFFENCES (Criminal)
    "65":  "Tampering with computer source code—imprisonment up to 3 years or fine up to ₹2 lakh or both.",
    "66":  "Computer-related offences—imprisonment up to 3 years or fine up to ₹5 lakh or both.",
    "66A": "Struck down by Supreme Court (Shreya Singhal, 2015).",
    "66B": "Receiving stolen computer resource—imprisonment up to 3 years or fine up to ₹1 lakh or both.",
    "66C": "Identity theft—imprisonment up to 3 years and fine up to ₹1 lakh.",
    "66D": "Cheating by personation using computer—imprisonment up to 3 years and fine up to ₹1 lakh.",
    "66E": "Violation of privacy—imprisonment up to 3 years or fine up to ₹2 lakh or both.",
    "67":  "Publishing/transmitting obscene material—3 years + ₹5 lakh (first); 5 years + ₹10 lakh (subsequent).",
    "67A": "Publishing sexually explicit material—up to 5 years + ₹10 lakh.",
    "67B": "Child sexual content—up to 7 years + ₹10 lakh.",
    "67C": "Intermediaries to preserve and retain information—noncompliance punishable.",
    "68":  "Failure to comply with Controller's directions—up to 2 years + ₹1 lakh.",
    "69":  "Interception/monitoring/decryption—non-compliance up to 7 years.",
    "69A": "Blocking orders—non-compliance punishable up to 7 years.",
    "69B": "Monitoring of traffic data—non-compliance punishable.",
    "70":  "Protected systems—unauthorized access punishable up to 10 years.",
    "70A": "National nodal agency—no penalty.",
    "70B": "CERT-In—reporting non-compliance punishable.",
    "71":  "Misrepresentation—up to 2 years + ₹1 lakh.",
    "72":  "Breach of confidentiality/privacy—up to 2 years or fine up to ₹1 lakh or both.",
    "72A": "Disclosure of information in breach of contract—up to 3 years or ₹5 lakh or both.",
    "73":  "Publishing DSC with false particulars—up to 2 years or ₹1 lakh or both.",
    "74":  "Publication for fraudulent purpose—up to 2 years or ₹1 lakh or both.",
    "75":  "Extraterritorial application—no penalty.",
    "76":  "Confiscation of computer/resource used in offence.",
    "77":  "Compounding of offences—no penalty.",
    "77A":"Compensation instead of prosecution.",
    "77B":"Offences to be cognizable & bailable.",
    "78":  "Power to investigate—no penalty.",

    # -----------------------------------------------------
    # CHAPTER XII — INTERMEDIARIES + MISC.
    # -----------------------------------------------------
    "79":  "Intermediary safe harbour—liable if due diligence not followed.",
    "80":  "Power of police/others to enter, search, arrest without warrant.",
    "81":  "Act to have overriding effect—no penalty.",
    "82":  "Protection of action taken in good faith—no penalty.",
    "83":  "Offences by companies—liability clarification.",
    "84":  "Offences outside India—no penalty.",
    "84A":"Modes & methods of encryption—no penalty.",
    "85":  "Offences by companies—directors liable.",
    "86":  "Removal of difficulties—no penalty.",
    "87":  "Power of Central Government to make rules.",
    "88":  "Cyber Regulations Advisory Committee.",
    "89":  "Power to make regulations.",
    "90":  "Rules/regulations to be laid before Parliament."
}

_STOPWORDS = set("""
a an the and or of to for from with by on into as is are was were be being been
it this that those these such there here their which who whom where when while
shall may can could would should applicant petitioner respondent appellant
complainant defendant state court high supreme india indian year section act
order judgment exhibit annexure para page
""".split())

def _clean_token(t: str) -> str:
    t = str(t).strip()
    for p in ("txt_word__", "txt_char__", "sec_hot__", "sec_"):
        if t.startswith(p):
            t = t[len(p):]
    return t

def _grouped_terms(exp: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str], float]:
    """Return pos_words, neg_words, pos_secs, neg_secs, margin"""
    g = exp.get("grouped") or {}
    posw = [ _clean_token(w) for w,_ in g.get("pos",{}).get("words",[]) ]
    negw = [ _clean_token(w) for w,_ in g.get("neg",{}).get("words",[]) ]
    poss = [ _clean_token(s).replace("SECTION ","").upper() for s,_ in g.get("pos",{}).get("sections",[]) ]
    negs = [ _clean_token(s).replace("SECTION ","").upper() for s,_ in g.get("neg",{}).get("sections",[]) ]
    return posw, negw, poss, negs, float(exp.get("margin",0.0))

def _keep_readable(words: List[str], k: int = 6) -> List[str]:
    out = []
    for w in words:
        wl = w.lower()
        if len(wl) < 5:         # drop very short words
            continue
        if wl in _STOPWORDS:    # drop legal/function words
            continue
        if not re.search(r"[a-z]", wl):
            continue
        out.append(w)
        if len(out) >= k:
            break
    return out

def _pretty_section_list(secs: List[str], k: int = 3) -> List[str]:
    nice = []
    for s in secs:
        tag = _SECTION_TAG.get(s, "").strip()
        nice.append(f"Section {s}" + (f" ({tag})" if tag else ""))
        if len(nice) >= k:
            break
    return nice

def _decide_direction(predicted_label: str = None,
                      proba: float = None,
                      threshold: float = None,
                      margin: float = None) -> str:
    """
    Prefer UI decision (label or proba/threshold).
    Fall back to margin sign if needed.
    """
    if predicted_label in {"PETITIONER", "RESPONDENT"}:
        return predicted_label
    if proba is not None and threshold is not None:
        return "PETITIONER" if proba >= threshold else "RESPONDENT"
    if margin is not None:
        return "PETITIONER" if margin > 0 else "RESPONDENT"
    return "RESPONDENT"

def make_plain_english_explanation(
    exp: Dict[str, Any],
    proba: float = None,
    threshold: float = None,
    predicted_label: str = None,
    detected_sections: List[str] = None,
    max_terms: int = 5
) -> Dict[str, Any]:
    """
    Convert explainer.explain_single output into short, plain-English text.
    Crucially, the 'direction' now aligns with the actual predicted label (or proba vs threshold).
    """
    posw, negw, poss, negs, margin = _grouped_terms(exp)
    direction = _decide_direction(predicted_label, proba, threshold, margin)

    pos_words = _keep_readable(posw, k=max_terms)
    neg_words = _keep_readable(negw, k=max_terms)

    # Prefer the UI-detected sections (from your predictor), then fall back to local section attributions
    ui_secs = list(detected_sections or [])
    nice_ui_secs = _pretty_section_list(ui_secs, k=4) if ui_secs else []
    pos_secs  = _pretty_section_list(poss, k=3)
    neg_secs  = _pretty_section_list(negs, k=3)

    lines = []
    if proba is not None and threshold is not None:
        pct = round(proba * 100, 1)
        thr_pct = round(threshold * 100, 1)
        lines.append(
            f"Based on the model’s probability **{pct}%** versus the decision threshold **{thr_pct}%**, "
            f"the predicted winner is **{direction}**."
        )
    else:
        lines.append(f"The local explanation supports **{direction}** overall.")

    # Concrete legal anchors first (sections)
    if nice_ui_secs:
        lines.append("Key statutory anchors detected in the case:")
        for s in nice_ui_secs:
            lines.append(f"- {s}")

    # Then the model’s strongest textual cues on both sides
    if pos_words:
        lines.append("Textual cues that pushed **PETITIONER** upward:")
        lines.append("  - " + ", ".join(f"“{w}”" for w in pos_words) + ".")
    if pos_secs and not nice_ui_secs:
        lines.append("Sections that pushed **PETITIONER** upward:")
        for s in pos_secs:
            lines.append(f"  - {s}")

    if neg_words:
        lines.append("Textual cues that pushed **RESPONDENT** upward:")
        lines.append("  - " + ", ".join(f"“{w}”" for w in neg_words) + ".")
    if neg_secs and not nice_ui_secs:
        lines.append("Sections that pushed **RESPONDENT** upward:")
        for s in neg_secs:
            lines.append(f"  - {s}")

    if len(lines) == 1:
        lines.append("No specific phrases or sections dominated; the model relied on broader context.")

    return {"summary_md": "\n\n".join(lines)}
