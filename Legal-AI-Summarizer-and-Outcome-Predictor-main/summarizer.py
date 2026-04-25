import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils import load_pdf_chunks

# ------------------------------------------------------
# Load environment variables
# ------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------------------------------------
# Global constants
# ------------------------------------------------------
VECTOR_DB_PATH = "vector_db"
KNOWLEDGE_BASE_PATH = "data/IT_Act_2000.pdf"
MAX_PAGES_TO_PROCESS = 25

# ------------------------------------------------------
# SMART MODEL SELECTION (Free tier friendly)
# ------------------------------------------------------
def get_available_model():
    """Try models in order of preference, return first available"""
    models = [
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    llm = ChatGroq(api_key=GROQ_API_KEY, model=models[0], temperature=0.0)
    return llm

# ------------------------------------------------------
# Build Vectorstore (Cached)
# ------------------------------------------------------
def build_vectorstore():
    from langchain_community.document_loaders import PyMuPDFLoader

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(VECTOR_DB_PATH):
        print("📦 Using cached vector store")
        return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

    print("🔨 Building vector store (one-time setup)...")
    loader = PyMuPDFLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    kb_chunks = splitter.split_documents(documents)

    vector_store = Chroma.from_documents(kb_chunks, embeddings, persist_directory=VECTOR_DB_PATH)
    print(f"✅ Vector store ready with {len(kb_chunks)} chunks")
    return vector_store

# ------------------------------------------------------
# OPTIMIZED: Direct Summarization (No RAG for demos)
# ------------------------------------------------------
def summarize_without_rag(chunks: list) -> str:
    """
    Demo-friendly: Summarize directly without RAG retrieval
    Saves tokens and avoids context contamination
    """
    llm = get_available_model()

    text = "\n\n".join(chunks)

    prompt = f"""You are a legal AI assistant. Analyze this case document and create a FIRAC summary.

**STRICT RULES:**
- Only use information from the document below
- Do NOT add external legal procedures or court information
- Focus on: What happened? Legal issue? Applicable law? How applied? Outcome?

**CASE DOCUMENT:**
{text[:15000]}

**FIRAC FORMAT:**
Facts (F): [Key facts from case]
Issue (I): [Legal question]
Rule (R): [IT Act/laws cited]
Application (A): [How law applies]
Conclusion (C): [Outcome]

Generate concise FIRAC summary now:"""

    try:
        print("⚡ Generating summary...")
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e):
            return "⚠️ **Rate limit reached.** Please wait a few minutes or try again later. Free tier has daily limits."
        return f"Error: {str(e)}"

# ------------------------------------------------------
# OPTIMIZED: With RAG using modern LCEL (replaces deprecated RetrievalQA)
# ------------------------------------------------------
def summarize_with_rag(chunks: list) -> str:
    """Use RAG only for smaller documents to conserve tokens"""

    vector_store = build_vectorstore()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 5}  # Only 2 chunks
    )

    llm = get_available_model()

    prompt = PromptTemplate(
        template="""You are analyzing a legal case. Use the IT Act context ONLY to identify relevant sections.

**IT ACT REFERENCE (for section numbers only):**
{context}

**USER'S CASE:**
{question}

**GENERATE FIRAC SUMMARY OF THE CASE:**
Facts (F):
Issue (I):
Rule (R): [Cite IT Act sections from context]
Application (A):
Conclusion (C):

SUMMARY:
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Modern LCEL chain — replaces deprecated RetrievalQA
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    text = "\n\n".join(chunks)
    query = f"Summarize this case:\n\n{text[:12000]}"

    try:
        result = rag_chain.invoke(query)
        return result  # StrOutputParser returns a plain string, no subscripting needed
    except Exception as e:
        if "429" in str(e):
            return "⚠️ Rate limit reached. Switching to no-RAG mode...\n\n" + summarize_without_rag(chunks)
        return f"Error: {str(e)}"

# ------------------------------------------------------
# MAIN SUMMARIZER (Token-Optimized)
# ------------------------------------------------------
def summarize_pdf(pdf_path: str) -> str:
    """
    Demo-optimized summarizer:
    - Limits pages processed
    - Uses fastest model
    - Skips RAG for speed
    """
    chunks = load_pdf_chunks(pdf_path, max_pages=MAX_PAGES_TO_PROCESS)
    print(f"📄 Processing {len(chunks)} chunks (first {MAX_PAGES_TO_PROCESS} pages)")

    total_chars = sum(len(c) for c in chunks)
    estimated_tokens = total_chars // 4
    print(f"📊 Estimated tokens: ~{estimated_tokens}")

    try:
        if estimated_tokens > 8000:
            print("⚡ Using direct summarization (no RAG) for speed")
            return summarize_without_rag(chunks)
        else:
            print("⚡ Using RAG for enhanced accuracy")
            return summarize_with_rag(chunks)

    except Exception as e:
        print(f"⚠️ Error: {e}")
        if "429" in str(e) or "rate_limit" in str(e):
            return """⚠️ **Daily token limit reached on Groq free tier.**

**For your demo, try:**
1. Wait 10-15 minutes for rate limit reset
2. Process smaller documents (first 15-20 pages)
3. Use the demo with pre-processed examples

**Note:** Free tier allows ~5-10 large document summaries per day."""
        return f"Error during summarization: {str(e)}"
