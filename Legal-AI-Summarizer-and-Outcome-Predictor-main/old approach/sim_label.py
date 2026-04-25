import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from utils import load_pdf_chunks
from config import IT_QUERIES , PROCEDURE_QUERIES, OUTCOME_QUERIES , SECTION_QUERIES
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


# Same embedding you use elsewhere (keep consistent!)
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

# Regex to catch explicit IT Act mentions/sections
IT_REGEX = re.compile(
    r"(Information\s+Technology\s+Act|IT\s+Act|Sec(?:tion)?\s*\d+[A-Z]?\s*of\s+the\s+IT\s*Act|Section\s*(66[A-Z]?|67[A-Z]?|72))",
    re.IGNORECASE
)

def build_it_centroid(queries: List[str]) -> np.ndarray:
    Q = embedder.embed_documents(queries)  # list -> list[vec]
    return np.mean(np.array(Q), axis=0, keepdims=True)  # shape (1, d)

def extract_additional_features(chunks):
    def sim_to(queries):
        centroid = np.mean(embedder.embed_documents(queries), axis=0, keepdims=True)
        C = embedder.embed_documents(chunks)
        return float(np.max(cosine_similarity(C, centroid)))  # top match score

    return {
        "outcome_score": sim_to(OUTCOME_QUERIES),
        "procedure_score": sim_to(PROCEDURE_QUERIES),
        "section_score": sim_to(SECTION_QUERIES)
    }

def score_chunks_against_it(
    chunks: List[str],
    it_queries: List[str],
    tau_chunk: float = 0.70,
    tau_doc: float = 0.72,
    regex_bonus: float = 0.05,
    min_pos_chunks_for_doc: int = 2,
) -> Dict:
    """
    Returns:
      {
        "chunk_scores": List[{"idx": i, "sim": float, "regex_hit": bool, "final": float}],
        "doc_score_max": float,
        "doc_score_mean_top3": float,
        "doc_label": "IT" | "NON_IT",
        "thresholds": {"chunk": tau_chunk, "doc": tau_doc}
      }
    """
    centroid = build_it_centroid(it_queries)     # (1, d)
    C = embedder.embed_documents(chunks)         # (n, d)
    sims = cosine_similarity(np.array(C), centroid).ravel()  # (n,)

    results = []
    for i, (chunk, sim) in enumerate(zip(chunks, sims)):
        hit = bool(IT_REGEX.search(chunk))
        final = sim + (regex_bonus if hit else 0.0)
        results.append({"idx": i, "sim": float(sim), "regex_hit": hit, "final": float(final)})

    # Aggregate to doc
    finals = np.array([r["final"] for r in results])
    doc_max  = float(np.max(finals)) if len(finals) else 0.0
    doc_top3 = float(np.mean(np.sort(finals)[-3:])) if len(finals) >= 3 else doc_max

    # Decision rules (modify to taste):
    #  - IT if either doc_top3 >= tau_doc OR (doc_max >= tau_doc and at least min_pos_chunks >= tau_chunk)
    pos_chunks = int(np.sum(finals >= tau_chunk))
    is_it = (doc_top3 >= tau_doc) or (doc_max >= tau_doc and pos_chunks >= min_pos_chunks_for_doc)

    return {
        "chunk_scores": results,
        "doc_score_max": doc_max,
        "doc_score_mean_top3": doc_top3,
        "doc_label": "IT" if is_it else "NON_IT",
        "thresholds": {"chunk": tau_chunk, "doc": tau_doc},
        "pos_chunks": pos_chunks
    }

def main1():
    # ------------------------------------------------------
    # Load environment variables
    # ------------------------------------------------------
    load_dotenv()

    FOLDER_PATH = "supreme_court_judgments/2024"   # folder containing all PDFs
    OUTPUT_CSV = "supreme_court_judgments/2024/classification_results.csv"

    import warnings
    warnings.filterwarnings("ignore")

    rows = []
    pdf_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".pdf")]

    for file in tqdm(pdf_files, desc="Classifying PDFs"):
        pdf_path = os.path.join(FOLDER_PATH, file)
        try:
            # ✅ Load fewer, higher-quality chunks
            chunks = load_pdf_chunks(pdf_path, max_pages=20)

            # Remove short or empty chunks (<200 chars)
            chunks = [c.strip() for c in chunks if len(c.strip()) > 200]

            # Limit to top 40 most text-dense chunks for speed
            if len(chunks) > 40:
                chunks = sorted(chunks, key=len, reverse=True)[:40]

            # Handle empty documents safely
            if not chunks:
                rows.append({
                    "pdf_name": file,
                    "doc_label": "EMPTY",
                    "doc_max_sim": 0.0,
                    "doc_top3_mean": 0.0,
                    "positive_chunks": 0,
                    "total_chunks": 0
                })
                continue

            # Run your classifier
            res = score_chunks_against_it(
                chunks=chunks,
                it_queries=IT_QUERIES,
                tau_chunk=0.70,
                tau_doc=0.70,
                regex_bonus=0.20
            )
            extra = extract_additional_features(chunks)
            rows.append({
                "pdf_name": file,
                "doc_label": res["doc_label"],
                "doc_max_sim": round(res["doc_score_max"], 3),
                "doc_top3_mean": round(res["doc_score_mean_top3"], 3),
                "positive_chunks": res["pos_chunks"],
                "total_chunks": len(chunks),
                "outcome_score": round(extra["outcome_score"], 3),
                "procedure_score": round(extra["procedure_score"], 3),
                "section_score": round(extra["section_score"], 3)
            })

        except Exception as e:
            rows.append({
                "pdf_name": file,
                "doc_label": f"ERROR: {str(e)[:25]}",
                "doc_max_sim": 0.0,
                "doc_top3_mean": 0.0,
                "positive_chunks": 0,
                "total_chunks": 0
            })

    # ------------------------------------------------------
    # Save to CSV
    # ------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved classification results to: {OUTPUT_CSV}")
    print(f"Total PDFs processed: {len(rows)}")

if __name__ == "__main__":
    main1()

# RAW Main usage:


# def main():
#     # ------------------------------------------------------
#     # Load environment variables
#     # ------------------------------------------------------
#     load_dotenv()
#     # GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#     # VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db")
#     # KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/IT_Act_2000.pdf")
#     # MAX_PAGES_TO_PROCESS = os.getenv("MAX_PAGES_TO_PROCESS", 25)
#     FOLDER_PATH = "supreme_court_judgments/2025/"   # folder containing all PDFs
#     OUTPUT_CSV = "supreme_court_judgments/2025/classification_results.csv"
#     # ------------------------------------------------------
#     rows = []
#     for file in tqdm(os.listdir(FOLDER_PATH), desc="Classifying PDFs"):
#         if not file.lower().endswith(".pdf"):
#             continue

#         pdf_path = os.path.join(FOLDER_PATH, file)
#         pdf_chunks = load_pdf_chunks(pdf_path, max_pages=25)

#         res = score_chunks_against_it(
#             chunks=pdf_chunks,
#             it_queries=IT_QUERIES,
#             tau_chunk=0.70,
#             tau_doc=0.70,
#             regex_bonus=0.20
#         )

#         rows.append({
#             "pdf_name": file,
#             "doc_label": res["doc_label"],
#             "doc_max_sim": round(res["doc_score_max"], 3),
#             "doc_top3_mean": round(res["doc_score_mean_top3"], 3),
#             "positive_chunks": res["pos_chunks"],
#             "total_chunks": len(pdf_chunks)
#         })

#     # Save to CSV
#     df = pd.DataFrame(rows)
#     df.to_csv(OUTPUT_CSV, index=False)
#     print(f"\n✅ Saved classification results to: {OUTPUT_CSV}")