import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

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

# ------------------------------------------------------
# Prompt Templates
# ------------------------------------------------------
STUFF_PROMPT = PromptTemplate(
    template="""
    You are an expert legal AI assistant specializing in India's IT Act, 2000. 
    Using ONLY the provided legal context below, analyze the user's document 
    and generate a structured summary in FIRAC format (Facts, Issue, Rule, Application, Conclusion).

    CONTEXT:
    {context}

    DOCUMENT:
    {question}

    FINAL ANSWER (in FIRAC format):
    """,
    input_variables=["context", "question"]
)

MAP_PROMPT = PromptTemplate(
    template="""
    Extract only the **key elements** (facts, issues, rules) from this chunk. 
    Do NOT generate full FIRAC yet.

    DOCUMENT CHUNK:
    {context}

    KEY ELEMENTS OF CHUNK:
    """,
    input_variables=["context"]
)

COMBINE_PROMPT = PromptTemplate(
    template="""
    Combine the extracted elements from all chunks into one coherent **final FIRAC summary**.

    ELEMENTS FROM CHUNKS:
    {summaries}

    FINAL FIRAC SUMMARY (Facts, Issue, Rule, Application, Conclusion):
    """,
    input_variables=["summaries"]
)

# ------------------------------------------------------
# Build Vectorstore
# ------------------------------------------------------
def build_vectorstore():
    from langchain_community.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    kb_chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(kb_chunks, embeddings, persist_directory=VECTOR_DB_PATH)

# ------------------------------------------------------
# RAG Pipeline Setup
# ------------------------------------------------------
def setup_rag_pipeline(chain_type="stuff"):
    vector_store = build_vectorstore()
    retriever = vector_store.as_retriever()

    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1)

    if chain_type == "stuff":
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": STUFF_PROMPT}
        )
    else:
        # ✅ Manual MapReduce chain instead of chain_type_kwargs
        map_chain = LLMChain(llm=llm, prompt=MAP_PROMPT)
        combine_chain = LLMChain(llm=llm, prompt=COMBINE_PROMPT)
        reduce_chain = StuffDocumentsChain(llm_chain=combine_chain, document_variable_name="summaries")

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_chain,
            document_variable_name="context",
            return_intermediate_steps=False
        )

        return RetrievalQA(
            retriever=retriever,
            combine_documents_chain=map_reduce_chain
        )

# ------------------------------------------------------
# Summarizer
# ------------------------------------------------------
def summarize_pdf(pdf_path: str) -> str:
    chunks = load_pdf_chunks(pdf_path)
    print(f"📄 Loaded {len(chunks)} chunks from {pdf_path}")

    chain_type = "stuff" if len(chunks) <= 10 else "map_reduce"
    print(f"⚡ Using {chain_type.upper()} pipeline")

    rag_chain = setup_rag_pipeline(chain_type)

    try:
        text = "\n".join(chunks)
        result = rag_chain.invoke({"query": text})
        return result["result"]
    except Exception as e:
        print(f"⚠️ Error in summarization: {e}")
        return ""





















import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

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

# ------------------------------------------------------
# Prompt Templates - FIXED FOR HALLUCINATION
# ------------------------------------------------------
STUFF_PROMPT = PromptTemplate(
    template="""
    You are an expert legal AI assistant specializing in India's IT Act, 2000. 

    **CRITICAL INSTRUCTIONS:**
    1. ONLY analyze the USER'S DOCUMENT provided below
    2. Use the LEGAL CONTEXT only to identify applicable IT Act sections
    3. DO NOT include information about Appellate Tribunals, procedures, or anything NOT in the user's document
    4. If the document doesn't mention something, DO NOT add it
    5. Generate a FIRAC summary based STRICTLY on what's in the user's document

    **LEGAL CONTEXT (for reference only):**
    {context}

    **USER'S DOCUMENT TO ANALYZE:**
    {question}

    **YOUR TASK:**
    Generate a FIRAC summary of the USER'S DOCUMENT ONLY:
    - Facts: Only facts stated in the user's document
    - Issue: Legal issues present in the user's document
    - Rule: IT Act sections that apply (reference legal context)
    - Application: How the rules apply to the user's document facts
    - Conclusion: Based only on the user's document

    FINAL FIRAC SUMMARY:
    """,
    input_variables=["context", "question"]
)

MAP_PROMPT = PromptTemplate(
    template="""
    Extract ONLY the factual information from this document chunk.
    Do NOT add any external legal knowledge or procedures.

    DOCUMENT CHUNK:
    {context}

    EXTRACTED FACTS AND ISSUES:
    """,
    input_variables=["context"]
)

COMBINE_PROMPT = PromptTemplate(
    template="""
    Combine the extracted elements into a FIRAC summary.
    
    **STRICT RULES:**
    - Only use information from the document chunks provided
    - Do NOT add information about courts, tribunals, or procedures unless explicitly mentioned
    - Focus on the specific case/document content

    DOCUMENT ELEMENTS:
    {summaries}

    FINAL FIRAC SUMMARY (Facts, Issue, Rule, Application, Conclusion):
    """,
    input_variables=["summaries"]
)

# ------------------------------------------------------
# Build Vectorstore
# ------------------------------------------------------
def build_vectorstore():
    from langchain_community.document_loaders import PyMuPDFLoader
    
    # Check if vector store already exists
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(VECTOR_DB_PATH):
        print("📦 Loading existing vector store...")
        return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    
    print("🔨 Building new vector store...")
    loader = PyMuPDFLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    kb_chunks = splitter.split_documents(documents)

    vector_store = Chroma.from_documents(kb_chunks, embeddings, persist_directory=VECTOR_DB_PATH)
    print(f"✅ Vector store created with {len(kb_chunks)} chunks")
    return vector_store

# ------------------------------------------------------
# RAG Pipeline Setup - FIXED
# ------------------------------------------------------
def setup_rag_pipeline(chain_type="stuff"):
    vector_store = build_vectorstore()
    
    # CRITICAL FIX: Reduce retrieved chunks and increase relevance threshold
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3  # Reduced from default (4-5) to get only most relevant chunks
        }
    )

    llm = ChatGroq(
    api_key=GROQ_API_KEY, 
    model="llama-3.3-70b-versatile",  # ✅ Has 8K token limit instead of 6K
    temperature=0.0,
    max_tokens=2048
    )   

    if chain_type == "stuff":
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": STUFF_PROMPT},
            return_source_documents=False  # Don't return source docs to reduce noise
        )
    else:
        map_chain = LLMChain(llm=llm, prompt=MAP_PROMPT)
        combine_chain = LLMChain(llm=llm, prompt=COMBINE_PROMPT)
        reduce_chain = StuffDocumentsChain(
            llm_chain=combine_chain, 
            document_variable_name="summaries"
        )

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_chain,
            document_variable_name="context",
            return_intermediate_steps=False
        )

        return RetrievalQA(
            retriever=retriever,
            combine_documents_chain=map_reduce_chain
        )

# ------------------------------------------------------
# Summarizer - ENHANCED
# ------------------------------------------------------
def summarize_pdf(pdf_path: str) -> str:
    chunks = load_pdf_chunks(pdf_path)
    print(f"📄 Loaded {len(chunks)} chunks from {pdf_path}")

    # Adaptive chunk sizing
    chain_type = "stuff" if len(chunks) <= 15 else "map_reduce"  # Increased threshold
    print(f"⚡ Using {chain_type.upper()} pipeline")

    rag_chain = setup_rag_pipeline(chain_type)

    try:
        # Prepare document text
        text = "\n\n".join(chunks)  # Better chunk separation
        
        # Add explicit instruction prefix
        query = f"""Analyze the following case document and create a FIRAC summary based ONLY on its contents:

{text}

Remember: Only summarize what is explicitly stated in this document. Do not add external legal procedures or information."""
        
        result = rag_chain.invoke({"query": query})
        return result["result"]
    except Exception as e:
        print(f"⚠️ Error in summarization: {e}")
        return f"Error during summarization: {str(e)}"