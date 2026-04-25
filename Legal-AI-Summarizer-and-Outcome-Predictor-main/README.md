# ⚖️ Legal AI: FIRAC Summarizer & Case Outcome Predictor (IT Act, 2000)
<img width="615" height="451" alt="image" src="https://github.com/user-attachments/assets/123e0209-813f-4319-9cc9-86d83b6fea19" />

## 📌 Overview
This project is a **domain-specific Legal AI system** focused on Indian cyber law cases governed by the **Information Technology Act, 2000**.  
It combines **Retrieval-Augmented Generation (RAG)** and **Explainable Machine Learning** to assist legal research by:

- Generating structured **FIRAC summaries** of judgments
- Predicting **case outcomes (Petitioner vs Respondent)**
- Explaining predictions in **plain English**
- Mapping relevant **IT Act penalties**

The system is intended for **academic and research use** and does **not replace judicial reasoning**.

---

## 🧩 System Architecture
 <img width="1536" height="1024" alt="system_overview" src="https://github.com/user-attachments/assets/3d66d5b3-7707-4484-8d57-270498f7fd98" />


The project consists of **two tightly integrated phases**:

1. **Phase 1 – FIRAC Summarizer (RAG-based)**
2. **Phase 2 – Case Outcome Predictor (Explainable ML)**

Both phases are exposed through a **single Streamlit application**.

---

## 📑 Phase 1: FIRAC Summarizer (RAG Pipeline)

### What It Does
- Produces **FIRAC-style summaries**:
  - Facts
  - Issues
  - Rules
  - Analysis
  - Conclusion
- Grounds summaries in:
  - **IT Act, 2000 (PDF knowledge base)**
  - **Uploaded Indian court judgments**

### How It Works
1. PDF text extraction
2. **Semantic chunking** of legal text
3. Vector embedding of chunks
4. Similarity-based retrieval
5. Context-aware response generation using LLMs

### Why RAG Is Used
- Prevents hallucination common in generic LLMs
- Ensures outputs remain **statute-grounded**
- Enables traceable and auditable summaries
- Improves factual consistency in legal text generation

---

## 🧠 Phase 2: Case Outcome Predictor

### What It Does
- Predicts the **likely winner** of a case:
  - **PETITIONER**
  - **RESPONDENT**
- Provides:
  - Probability score
  - Optimized decision threshold
  - Plain-English explanation
  - Relevant IT Act penalties

---

## ⚙️ Model & Feature Engineering

### Feature Construction
- **Text Features**
  - TF-IDF vectorization
  - Unigrams and bigrams
  - Legal-specific token filtering
- **Statutory Features**
  - One-hot encoding of IT Act sections (e.g., Section 66, 43A)

### Model Used
- **Calibrated Logistic Regression**
  - Stratified 5-fold cross-validation
  - Class-weight balancing
  - Probability calibration using **Platt Scaling (Sigmoid)**

### Why Logistic Regression
- High interpretability (linear coefficients)
- Stable on limited legal datasets
- Coefficients directly support explainability
- Faster and more reliable than black-box models in legal domains

---

## 📊 Evaluation & Results
<img width="442" height="339" alt="image" src="https://github.com/user-attachments/assets/41434271-2dd7-4e96-b16c-e4146756eb5a" />


### Quantitative Metrics
- Accuracy
- Macro F1-score (primary optimization metric)
- ROC-AUC
- Confusion Matrix

### Qualitative Evaluation
- Manual comparison against real judgments
- Validation of:
  - Detected IT Act sections
  - Predicted outcome alignment
  - Explanation plausibility and legal coherence

---

## 🔎 Explainable AI (XAI)

- Feature-level contribution analysis
- Identifies:
  - Influential words/phrases
  - Statutory sections affecting predictions
- Converts model reasoning into **plain-English explanations**
- Designed for **non-technical legal users**

---

## 🏛️ IT Act Penalty Mapping
- Automatically retrieves penalties for detected IT Act sections
- Section-aware and configurable
- Educational and research use only

---

## 🖥️ User Interface
- Built using **Streamlit**
- Unified application with:
  - FIRAC Summarizer tab
  - Case Outcome Predictor tab
- Supports PDF uploads
- Displays predictions, explanations, and penalties in a single view

---

## 🗂️ Project Structure
```text
app_integrated.py
ik_it_act_scraper.py
clean_preprocess_it_cases.py
train_models2.py
case_predictor.py
explainer.py
narrative_explainer.py
penalties_retriever.py
it_act_config.py
models/
scraped_it_cases/
README.md
