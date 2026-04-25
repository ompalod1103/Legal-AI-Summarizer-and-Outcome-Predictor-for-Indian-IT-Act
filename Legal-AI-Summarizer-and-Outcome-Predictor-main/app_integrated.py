#app-integrated v3
import os
import re
import joblib
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Phase 1 RAG
from summarizer import summarize_pdf

# Phase 2 (predictor + penalties + explainer)
from case_predictor import predict_case, extract_text_from_pdf, make_inference_frame
from penalties_retriever import retrieve_penalties_for_sections
from narrative_explainer import make_plain_english_explanation
from explainer import explain_single

load_dotenv()
st.set_page_config(page_title="Legal AI – FIRAC + Outcome Predictor", layout="wide")
st.title("⚖️ Legal AI Workbench")
st.caption("FIRAC summarization • Case outcome prediction • IT Act penalties • Local explanation")

tab_sum, tab_pred = st.tabs(["📑 FIRAC Summarizer", "🧠 Case Outcome Predictor"])

# ----------------------------
# Tab 1: Summarizer (Phase 1)
# ----------------------------
with tab_sum:
    st.subheader("Upload a case PDF to generate a FIRAC summary")
    up = st.file_uploader("Upload PDF", type=["pdf"], key="sum_pdf")

    if up:
        if st.button("Generate FIRAC Summary", type="primary"):
            with st.spinner("Summarizing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(up.getbuffer())
                    tmp_path = tmp.name
                try:
                    summary = summarize_pdf(tmp_path)
                    st.markdown("### 📄 FIRAC Summary")
                    st.write(summary)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

# -------------------------------------
# Tab 2: Predictor (Phase 2)
# -------------------------------------
with tab_pred:
    st.subheader("Predict outcome for a case and view relevant IT-Act penalties")
    up2 = st.file_uploader("Upload PDF", type=["pdf"], key="pred_pdf")

    if up2:
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Analyzing case & running model..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
                    tmp2.write(up2.getbuffer())
                    tmp2_path = tmp2.name
                try:
                    # 1) Predict
                    pred = predict_case(tmp2_path, model_dir="models")
                    p_pet = float(pred["proba_petitioner"])
                    thr = float(pred["decision_threshold"])
                    label = pred["predicted_winner_label"]

                    # --- Robust parties extraction ---
                    pet_name = None
                    res_name = None
                    if isinstance(pred.get("parties"), dict):
                        parties = pred["parties"]
                        pet_name = (
                            parties.get("petitioner")
                            or parties.get("applicant")
                            or parties.get("appellant")
                            or parties.get("complainant")
                        )
                        res_name = (
                            parties.get("respondent")
                            or parties.get("defendant")
                            or parties.get("state")
                        )
                    if not pet_name or not res_name:
                        if label == "PETITIONER":
                            pet_name = pet_name or pred.get("winner_name")
                            res_name = res_name or pred.get("loser_name")
                        else:
                            res_name = res_name or pred.get("winner_name")
                            pet_name = pet_name or pred.get("loser_name")

                    # 2) Headline
                    st.markdown("### 📊 Prediction")
                    c1, c2, c3 = st.columns([1.2, 2, 2])
                    with c1:
                        st.metric("Predicted winner (label)", label)
                    with c2:
                        st.progress(min(max(p_pet, 0.0), 1.0), text=f"P(Petitioner wins) = {p_pet:.3f}")
                    with c3:
                        st.write(f"**Decision threshold:** {thr:.2f}")

                    # Parties card
                    st.markdown("#### 👥 Parties detected (best-effort from first page)")
                    st.write(
                        f"- **Petitioner / Applicant / Appellant / Complainant:** `{pet_name or 'unknown'}`\n"
                        f"- **Respondent / Defendant / State:** `{res_name or 'unknown'}`"
                    )

                    # 3) Narrative reason
                    pct = round(p_pet * 100, 1)
                    verdict_word = "above" if p_pet >= thr else "below"
                    st.info(
                        f"The model estimates a **{pct}% probability that the Petitioner wins**. "
                        f"This is **{verdict_word}** the decision threshold of **{round(thr*100,1)}%**, "
                        f"so the predicted winner is **{label}**. "
                        f"Note: many orders title the Petitioner as **Applicant/Appellant**—we treat them equivalently."
                    )

                    # 4) Local explanation
                    bundle = joblib.load("models/binary_outcome_model.joblib")
                    pipe   = bundle["pipeline"]
                    text   = extract_text_from_pdf(tmp2_path)
                    X_row  = make_inference_frame(text, bundle.get("sec_cols", []))
                    exp    = explain_single(pipe, X_row, top_k=12)

                    st.markdown("### 🔎 Why this prediction?")
                    st.caption("We compute linear contribution weights from the calibrated Logistic Regression and convert them into a readable explanation.")
                    try:
                        nar = make_plain_english_explanation(exp, proba=p_pet, threshold=thr, predicted_label=label)
                        st.write(nar["summary_md"])
                    except Exception as e:
                        st.warning(f"Could not build the plain-English explanation: {e}")

                    # 5) IT-Act penalties
                    secs = pred["detected_sections"]
                    st.markdown("### 🏛️ Relevant penalties (Information Technology Act, 2000)")
                    if secs:
                        pen = retrieve_penalties_for_sections(secs, collapse_unknown=True)
                        for line in pen["penalties"]:
                            st.write(f"- {line}")
                        st.caption("Note: concise demo summaries for learning; verify against the statute before real-world use.")
                    else:
                        st.write("No IT-Act sections detected in the document for penalty lookup.")

                finally:
                    try:
                        os.remove(tmp2_path)
                    except Exception:
                        pass
